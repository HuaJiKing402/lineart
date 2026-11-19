//! 引导降噪：用原图作为结构先验，剔除已有线稿里的伪线。
//!
//! 输入是「原图 + 已生成的线稿」，输出是清理后的线稿。判据有两条，互为补充：
//!
//!   1. 黑顶帽（black top-hat）：抓原图里「细的深色笔画」。插画的线本来就是画上去的，
//!      顶帽能直接响应它们；而柔和渐变、平涂抖动不会产生响应。
//!   2. 彩色阶跃：跨通道梯度，覆盖顶帽抓不到的真边（浅色线、亮暗反转、色块交界）。
//!
//! 两者取最大得到「这里可以画线」的许可，再乘到墨量上：真线保持原强度，
//! 假线被清零（权重经锐化处理，避免线条变灰）。

use crate::blur;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct Params {
    /// 分辨率缩放系数（长边 / 1920）
    pub res: f32,
    /// 0..1，越大越激进
    pub strength: f32,
}

fn smoothstep(x: f32, lo: f32, hi: f32) -> f32 {
    let t = ((x - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 可分离的最小值滤波（方形结构元，边界 clamp）
fn min_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..y * w + w];
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let mut m = f32::INFINITY;
            for &v in &s[x0..=x1] {
                if v < m {
                    m = v;
                }
            }
            row[x] = m;
        }
    });
    let mut dst = vec![0.0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for x in 0..w {
            let mut m = f32::INFINITY;
            for yy in y0..=y1 {
                let v = tmp[yy * w + x];
                if v < m {
                    m = v;
                }
            }
            row[x] = m;
        }
    });
    dst
}

/// 可分离的最大值滤波
fn max_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..y * w + w];
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let mut m = f32::NEG_INFINITY;
            for &v in &s[x0..=x1] {
                if v > m {
                    m = v;
                }
            }
            row[x] = m;
        }
    });
    let mut dst = vec![0.0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for yy in y0..=y1 {
                let v = tmp[yy * w + x];
                if v > m {
                    m = v;
                }
            }
            row[x] = m;
        }
    });
    dst
}

/// 黑顶帽：闭运算 - 原图，响应「比周围暗且比结构元细」的结构
fn black_tophat(lum: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let closed = min_filter(&max_filter(lum, w, h, r), w, h, r);
    closed
        .par_iter()
        .zip(lum.par_iter())
        .map(|(&c, &l)| (c - l).max(0.0))
        .collect()
}

/// 跨通道梯度取最大，得到「颜色/亮度阶跃」强度
///
/// sobel_pair 的核未归一化（幅值是 8 倍），这里除回去，使阈值具有统一的像素量纲。
fn color_edge(planes: &[Vec<f32>], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let mut acc = vec![0.0f32; w * h];
    for p in planes {
        let b = blur::gaussian_blur(p, w, h, sigma);
        let (gx, gy) = blur::sobel_pair(&b, w, h);
        for i in 0..w * h {
            let m = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt() * 0.125;
            if m > acc[i] {
                acc[i] = m;
            }
        }
    }
    acc
}

/// 去掉孤立墨点：邻域内墨点少于 2 个的像素清零
fn despeckle(ink: &mut [f32], w: usize, h: usize, thr: f32) {
    for _ in 0..2 {
        let snapshot = ink.to_vec();
        ink.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                if snapshot[y * w + x] <= thr {
                    continue;
                }
                let mut n = 0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        if snapshot[yy * w + xx] > thr {
                            n += 1;
                        }
                    }
                }
                if n < 2 {
                    row[x] = 0.0;
                }
            }
        });
    }
}

/// 主入口：ink 为 [0,1] 墨量（1=全黑），返回清理后的白度图（1=白）
pub fn denoise(ink: &[f32], planes: &[Vec<f32>], w: usize, h: usize, p: Params) -> Vec<f32> {
    let n = w * h;
    let res = p.res.max(0.1);
    let k = 0.5 + p.strength * 1.0; // 0.5 .. 1.5
    let r = (2.0 * res).round().max(1.0) as usize;
    let grow = r;
    let sigma = 1.2 * res;

    // 亮度通道
    let lum: Vec<f32> = (0..n)
        .map(|i| 0.299 * planes[0][i] + 0.587 * planes[1][i] + 0.114 * planes[2][i])
        .collect();
    let lb = blur::gaussian_blur(&lum, w, h, 0.8 * res);

    // 证据 1：细深色笔画
    let mut th = black_tophat(&lb, w, h, r);
    let ref99 = blur::percentile_abs(&th, 99.5).max(1e-4);
    for v in th.iter_mut() {
        *v = (*v / ref99).clamp(0.0, 1.0);
    }
    let st = max_filter(&th, w, h, grow);
    let mut gate: Vec<f32> = st
        .par_iter()
        .map(|&v| smoothstep(v, 0.10 * k, 0.38 * k))
        .collect();

    // 证据 2：彩色阶跃
    let ce = color_edge(planes, w, h, sigma);
    let ce = max_filter(&ce, w, h, grow);
    for i in 0..n {
        let g = smoothstep(ce[i], 0.06 * k, 0.20 * k);
        if g > gate[i] {
            gate[i] = g;
        }
    }

    // 权重锐化：存活的线保持全黑，被否决的清零，避免整幅变灰
    for g in gate.iter_mut() {
        *g = smoothstep(*g, 0.28, 0.55);
    }

    let mut out: Vec<f32> = ink
        .par_iter()
        .zip(gate.par_iter())
        .map(|(&i, &g)| i * g)
        .collect();
    despeckle(&mut out, w, h, 0.12);

    out.par_iter().map(|&i| (1.0 - i).clamp(0.0, 1.0)).collect()
}
