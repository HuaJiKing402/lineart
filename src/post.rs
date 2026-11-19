//! 后处理：清理碎小线条连通块。

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
