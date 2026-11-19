//! 基础滤波：高斯模糊、box 滤波、导向滤波（guided filter）、Sobel、双线性采样。

use rayon::prelude::*;

/// 归一化的一维高斯核
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil().max(1.0) as isize;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let mut sum = 0.0f32;
    for x in -radius..=radius {
        let v = (-0.5 * (x as f32 / sigma).powi(2)).exp();
        kernel.push(v);
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// 可分离高斯模糊，边界采用 clamp 延拓，按行并行
pub fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 1e-4 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let kernel = gaussian_kernel(sigma);
    let r = (kernel.len() / 2) as isize;
    let max_x = w as isize - 1;
    let max_y = h as isize - 1;

    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..y * w + w];
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, &kw) in kernel.iter().enumerate() {
                let xx = (x as isize + i as isize - r).clamp(0, max_x) as usize;
                acc += s[xx] * kw;
            }
            row[x] = acc;
        }
    });

    let mut dst = vec![0.0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, &kw) in kernel.iter().enumerate() {
                let yy = (y as isize + i as isize - r).clamp(0, max_y) as usize;
                acc += tmp[yy * w + x] * kw;
            }
            row[x] = acc;
        }
    });
    dst
}

/// 均值（box）滤波，滑动窗口实现，O(1)/像素
pub fn box_blur(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let cnt = (2 * radius + 1) as f32;
    let max_x = w as isize - 1;
    let max_y = h as isize - 1;

    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..y * w + w];
        let mut sum = 0.0f32;
        for k in 0..=(2 * radius) {
            let xx = (k as isize - radius as isize).clamp(0, max_x) as usize;
            sum += s[xx];
        }
        for x in 0..w {
            row[x] = sum / cnt;
            let rem = (x as isize - radius as isize).clamp(0, max_x) as usize;
            let add = (x as isize + radius as isize + 1).clamp(0, max_x) as usize;
            sum += s[add] - s[rem];
        }
    });

    let mut dst = vec![0.0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in 0..=(2 * radius) {
                let yy = (y as isize + k as isize - radius as isize).clamp(0, max_y) as usize;
                sum += tmp[yy * w + x];
            }
            row[x] = sum / cnt;
        }
    });
    dst
}

/// 自导向滤波：保边平滑，用来压掉照片纹理噪声而不糊掉真实边缘。
/// 输入应为 [0,1]。
pub fn guided_filter(src: &[f32], w: usize, h: usize, radius: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    let mean_i = box_blur(src, w, h, radius);
    let sq: Vec<f32> = src.iter().map(|v| v * v).collect();
    let mean_ii = box_blur(&sq, w, h, radius);

    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        let var = (mean_ii[i] - mean_i[i] * mean_i[i]).max(0.0);
        let ai = var / (var + eps);
        a[i] = ai;
        b[i] = mean_i[i] * (1.0 - ai);
    }
    let mean_a = box_blur(&a, w, h, radius);
    let mean_b = box_blur(&b, w, h, radius);

    (0..n)
        .map(|i| (mean_a[i] * src[i] + mean_b[i]).clamp(0.0, 1.0))
        .collect()
}

/// 返回 (gx, gy) Sobel 梯度
pub fn sobel_pair(src: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    if w < 3 || h < 3 {
        return (gx, gy);
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let tl = src[(y - 1) * w + (x - 1)];
            let tc = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + (x + 1)];
            let ml = src[y * w + (x - 1)];
            let mr = src[y * w + (x + 1)];
            let bl = src[(y + 1) * w + (x - 1)];
            let bc = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + (x + 1)];

            gx[y * w + x] = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            gy[y * w + x] = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
        }
    }
    (gx, gy)
}

/// 双线性采样（边界 clamp）
pub fn sample_scalar(a: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let x0c = (x0 as isize).clamp(0, w as isize - 1) as usize;
    let x1c = (x0 as isize + 1).clamp(0, w as isize - 1) as usize;
    let y0c = (y0 as isize).clamp(0, h as isize - 1) as usize;
    let y1c = (y0 as isize + 1).clamp(0, h as isize - 1) as usize;

    let v00 = a[y0c * w + x0c];
    let v10 = a[y0c * w + x1c];
    let v01 = a[y1c * w + x0c];
    let v11 = a[y1c * w + x1c];
    (v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy) + (v01 * (1.0 - fx) + v11 * fx) * fy
}

pub fn sample_vec(a: &[f32], b: &[f32], w: usize, h: usize, x: f32, y: f32) -> (f32, f32) {
    (
        sample_scalar(a, w, h, x, y),
        sample_scalar(b, w, h, x, y),
    )
}

/// 取 |v| 的第 pct 百分位数，用于鲁棒归一化
pub fn percentile_abs(v: &[f32], pct: f32) -> f32 {
    if v.is_empty() {
        return 1.0;
    }
    let mut a: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    a.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((a.len() - 1) as f32) * pct / 100.0).round() as usize;
    a[idx.min(a.len() - 1)]
}
