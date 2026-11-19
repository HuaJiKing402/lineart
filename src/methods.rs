//! 各种线稿提取算法。所有函数输入/输出均为 [0,1] 灰度，1 = 白，0 = 黑。

use crate::blur::*;
use rayon::prelude::*;

/// XDoG 软阈值：u >= eps 时为白，否则用 tanh 平滑压到黑。
/// 平坦区域 DoG=0，因此 eps<=0 时平坦区恒为白；eps 越负只有越强的边缘成线。
fn threshold_tanh(u: &[f32], eps: f32, phi: f32) -> Vec<f32> {
    u.iter()
        .map(|&v| {
            let t = if v >= eps {
                1.0
            } else {
                1.0 + (phi * (v - eps)).tanh()
            };
            t.clamp(0.0, 1.0)
        })
        .collect()
}

fn dog_of(img: &[f32], w: usize, h: usize, sigma: f32, k: f32) -> Vec<f32> {
    let g1 = gaussian_blur(img, w, h, sigma);
    let g2 = gaussian_blur(img, w, h, sigma * k);
    g1.iter().zip(&g2).map(|(a, b)| a - b).collect()
}

/// 用 99 分位把 DoG 响应缩放成大致 [-1,1]，平坦区保持 0
fn signed_normalize(d: &[f32]) -> Vec<f32> {
    let scale = percentile_abs(d, 99.0).max(1e-3);
    d.iter().map(|&v| v / scale).collect()
}

/// XDoG：各向同性高斯差分 + 软阈值，快且线条连贯
pub fn xdog(
    img: &[f32],
    w: usize,
    h: usize,
    sigma: f32,
    k: f32,
    eps: f32,
    phi: f32,
) -> Vec<f32> {
    let dog = dog_of(img, w, h, sigma, k);
    let u = signed_normalize(&dog);
    threshold_tanh(&u, eps, phi)
}

/// 结构张量求边缘切线场（ETF）
fn edge_tangent_field(
    img: &[f32],
    w: usize,
    h: usize,
    sigma_c: f32,
) -> (Vec<f32>, Vec<f32>) {
    let base = gaussian_blur(img, w, h, 1.0);
    let (gx, gy) = sobel_pair(&base, w, h);

    let jxx: Vec<f32> = gx.iter().map(|v| v * v).collect();
    let jxy: Vec<f32> = gx.iter().zip(&gy).map(|(a, b)| a * b).collect();
    let jyy: Vec<f32> = gy.iter().map(|v| v * v).collect();

    let jxx = gaussian_blur(&jxx, w, h, sigma_c);
    let jxy = gaussian_blur(&jxy, w, h, sigma_c);
    let jyy = gaussian_blur(&jyy, w, h, sigma_c);

    let mut tx = vec![0.0f32; w * h];
    let mut ty = vec![0.0f32; w * h];
    for i in 0..w * h {
        // 主特征向量方向即梯度方向
        let theta = 0.5 * (2.0 * jxy[i]).atan2(jxx[i] - jyy[i]);
        let nx = theta.cos();
        let ny = theta.sin();
        // 切线 = 梯度逆时针转 90°
        tx[i] = -ny;
        ty[i] = nx;
    }
    (tx, ty)
}

/// 沿 ETF 做线积分卷积，把 DoG 响应沿边缘方向平均，抑制噪声、连通断线
fn fdog_lic(
    dog: &[f32],
    tx: &[f32],
    ty: &[f32],
    w: usize,
    h: usize,
    sigma_m: f32,
) -> Vec<f32> {
    let steps = ((3.0 * sigma_m).ceil() as usize).max(2);
    let inv_two_s2 = 1.0 / (2.0 * sigma_m * sigma_m);
    let max_x = (w - 1) as f32;
    let max_y = (h - 1) as f32;

    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = dog[y * w + x];
            let mut wsum = 1.0f32;

            for &sign in &[1.0f32, -1.0f32] {
                let mut cx = x as f32;
                let mut cy = y as f32;

                let (mut vx, mut vy) = sample_vec(tx, ty, w, h, cx, cy);
                let l = (vx * vx + vy * vy).sqrt();
                if l < 1e-6 {
                    continue;
                }
                vx = sign * vx / l;
                vy = sign * vy / l;

                let mut arc = 0.0f32;
                for _ in 0..steps {
                    let (mut dx, mut dy) = sample_vec(tx, ty, w, h, cx, cy);
                    let len = (dx * dx + dy * dy).sqrt();
                    if len < 1e-6 {
                        break;
                    }
                    dx /= len;
                    dy /= len;
                    // 与前进方向对齐，避免切线符号翻转导致来回走
                    if dx * vx + dy * vy < 0.0 {
                        dx = -dx;
                        dy = -dy;
                    }
                    vx = dx;
                    vy = dy;

                    cx += dx;
                    cy += dy;
                    if cx < 0.0 || cy < 0.0 || cx > max_x || cy > max_y {
                        break;
                    }
                    arc += 1.0;

                    let v = sample_scalar(dog, w, h, cx, cy);
                    let wt = (-arc * arc * inv_two_s2).exp();
                    acc += v * wt;
                    wsum += wt;
                }
            }
            row[x] = acc / wsum;
        }
    });
    out
}

/// Coherent Line Drawing：ETF 流场 + FDoG，线条最连贯，适合人像/主体
pub fn coherent(
    img: &[f32],
    w: usize,
    h: usize,
    sigma: f32,
    k: f32,
    eps: f32,
    phi: f32,
    sigma_c: f32,
    sigma_m: f32,
) -> Vec<f32> {
    let (tx, ty) = edge_tangent_field(img, w, h, sigma_c);
    let dog = dog_of(img, w, h, sigma, k);
    let fd = fdog_lic(&dog, &tx, &ty, w, h, sigma_m);
    let u = signed_normalize(&fd);
    threshold_tanh(&u, eps, phi)
}

/// Canny：NMS 细化到 1px + 双阈值滞后连接
pub fn canny(
    img: &[f32],
    w: usize,
    h: usize,
    sigma: f32,
    low: f32,
    high: f32,
) -> Vec<f32> {
    let blurred = gaussian_blur(img, w, h, sigma);
    let (gx, gy) = sobel_pair(&blurred, w, h);
    let n = w * h;

    let mut mag = vec![0.0f32; n];
    let mut ang = vec![0.0f32; n];
    for i in 0..n {
        mag[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
        ang[i] = gy[i].atan2(gx[i]);
    }
    let maxm = mag.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    for m in &mut mag {
        *m /= maxm;
    }

    // 非极大值抑制
    let mut thin = vec![0.0f32; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let a = ang[i].to_degrees().rem_euclid(180.0);
            let (n1, n2) = if a < 22.5 || a >= 157.5 {
                (i - 1, i + 1) // 水平
            } else if a < 67.5 {
                (i + w + 1, i - w - 1) // 右下
            } else if a < 112.5 {
                (i - w, i + w) // 垂直
            } else {
                (i + w - 1, i - w + 1) // 左下
            };
            if mag[i] >= mag[n1] && mag[i] >= mag[n2] {
                thin[i] = mag[i];
            }
        }
    }

    let mut strong = vec![false; n];
    let mut weak = vec![false; n];
    for i in 0..n {
        if thin[i] >= high {
            strong[i] = true;
        } else if thin[i] >= low {
            weak[i] = true;
        }
    }

    let mut mask = vec![0u8; n];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        if strong[i] {
            mask[i] = 1;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        let x = (i % w) as isize;
        let y = (i / w) as isize;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx as isize;
                let ny = y + dy as isize;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if weak[j] && mask[j] == 0 {
                    mask[j] = 1;
                    stack.push(j);
                }
            }
        }
    }

    let mut out: Vec<f32> = mask.iter().map(|&m| if m == 1 { 0.0 } else { 1.0 }).collect();
    // 轻微抗锯齿
    out = gaussian_blur(&out, w, h, 0.6);
    out
}

/// 铅笔素描：颜色减淡得到灰阶调子（背景保持白）+ XDoG 线条压暗
pub fn sketch(
    img: &[f32],
    w: usize,
    h: usize,
    sigma: f32,
    k: f32,
    eps: f32,
    phi: f32,
) -> Vec<f32> {
    let lines = xdog(img, w, h, sigma, k, eps, phi);
    let inverted: Vec<f32> = img.iter().map(|&v| 1.0 - v).collect();
    let smooth = gaussian_blur(&inverted, w, h, 4.0);

    lines
        .iter()
        .zip(img.iter())
        .zip(smooth.iter())
        .map(|((&l, &g), &s)| {
            let dodge = (g / (1.0 - s).max(1e-3)).clamp(0.0, 1.0);
            let tone = dodge.powf(2.2);
            tone.min(l)
        })
        .collect()
}
