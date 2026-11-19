//! 后处理：清理碎小线条连通块。
//!
//! 两个手段都在线稿的连通块层面工作，不改变任何保留下来的像素强度，
//! 因此不会把线条变淡或变细：
//!
//! - `hysteresis`：孤立弱响应删除，与强线连通的弱响应保留（Canny / Coherent
//!   Line Drawing 的成熟做法）。用于「弱而碎」的颗粒。
//! - `remove_small_components`：按面积删除，用于「小而成块」的碎点。

/// 迟滞筛选（输入输出均为白度，1 = 白）
///
/// 墨量 `>= strong` 的像素作为种子；墨量 `>= weak` 的连通块，只有包含种子才保留。
/// 弱而孤立的响应是噪声，弱而连成线的是真实结构 —— 这样既去了颗粒又不打断线条。
pub fn hysteresis(img: &mut [f32], w: usize, h: usize, strong: f32, weak: f32) {
    if w == 0 || h == 0 || weak <= 0.0 {
        return;
    }
    let n = w * h;
    let mut visited = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp: Vec<usize> = Vec::new();

    for start in 0..n {
        if visited[start] || 1.0 - img[start] < weak {
            continue;
        }
        stack.clear();
        comp.clear();
        stack.push(start);
        visited[start] = true;
        let mut has_seed = false;

        while let Some(i) = stack.pop() {
            comp.push(i);
            if 1.0 - img[i] >= strong {
                has_seed = true;
            }
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
                    if !visited[j] && 1.0 - img[j] >= weak {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
        }

        if !has_seed {
            for &i in &comp {
                img[i] = 1.0;
            }
        }
    }
}

/// 删除面积小于 min_area 的黑色连通块（8 邻域）
pub fn remove_small_components(img: &mut [f32], w: usize, h: usize, min_area: usize) {
    if min_area <= 1 || w == 0 || h == 0 {
        return;
    }
    let n = w * h;
    let mut visited = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp: Vec<usize> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        if img[start] > 0.5 {
            visited[start] = true;
            continue;
        }

        stack.clear();
        comp.clear();
        stack.push(start);
        visited[start] = true;

        while let Some(i) = stack.pop() {
            comp.push(i);
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
                    if !visited[j] && img[j] <= 0.5 {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
        }

        if comp.len() < min_area {
            for &i in &comp {
                img[i] = 1.0;
            }
        }
    }
}
