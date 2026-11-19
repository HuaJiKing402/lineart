//! lineart —— 照片转黑白线稿
//!
//! 算法：
//!   xdog     : XDoG（扩展高斯差分 + tanh 软阈值），快、线条连贯，默认
//!   coherent : Coherent Line Drawing（结构张量 ETF + 沿流线 FDoG），线条最连贯
//!   canny    : Canny（NMS 细化 + 双阈值滞后），细而硬的技术线
//!   sketch   : 铅笔素描（XDoG 线条 + 明暗调子）
//!
//! 预处理：自导向滤波压噪保边，可显著减少照片纹理带来的杂线。

mod blur;
mod methods;
mod post;

use image::{GrayImage, ImageBuffer};
use std::env;
use std::process;
use std::str::FromStr;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Method {
    Xdog,
    Coherent,
    Canny,
    Sketch,
}

struct Config {
    input: String,
    output: String,
    method: Method,
    sigma: f32,
    k: f32,
    epsilon: f32,
    phi: f32,
    denoise: f32,
    canny_high: f32,
    flow_sigma: f32,
    min_area: usize,
    scale: f32,
    invert: bool,
    binarize: bool,
}

fn print_help() {
    print!(
        r#"lineart —— 图片转黑白线稿

用法:
    lineart <输入图片> <输出图片> [选项]

方法 (-m):
    xdog        XDoG 高斯差分 (默认，快且连贯)
    coherent    相干线稿 ETF+FDoG (最连贯，适合人像/主体，较慢)
    canny       Canny 细线 (技术线稿)
    sketch      铅笔素描

预设 (-P):
    photo       照片 (默认)
    anime       动漫/插画 (细节更多、降噪更轻)
    ink         墨线 (更细更锐、适合干净素材)

参数:
    -s, --sigma <f32>        线条尺度，越大线越粗，默认 1.4（按长边相对 1920 自动缩放）
    -k, --k <f32>            DoG 高斯比，默认 1.6
    -e, --epsilon <-0.5..0>  线量阈值，越接近 0 线越多，越负越干净，默认 -0.05
    -p, --phi <f32>          线条锐度，越大越像墨线，默认 15
    -d, --denoise <0..1>     保边降噪强度，默认 0.45
    -t, --threshold <0..1>   Canny 高阈值，默认 0.09
    -f, --flow <f32>         coherent 沿流平滑 sigma，默认 4.0
        --min-area <int>     删除小于 N 像素的杂线，默认 0
        --scale <f32>        处理前缩放，默认 1.0 (大图可用 0.5 提速)
        --binarize           输出纯黑白
        --invert             黑底白线
    -h, --help               显示帮助

示例:
    lineart photo.jpg line.png
    lineart photo.jpg line.png -e -0.06 -s 1.6
    lineart portrait.jpg line.png -m coherent -f 5
    lineart anime.jpg line.png -P anime
    lineart photo.jpg out.png -d 0.6 --min-area 20
"#
    );
}

fn take_value(
    args: &[String],
    i: &mut usize,
    inline: Option<&str>,
    flag: &str,
) -> Result<String, String> {
    if let Some(v) = inline {
        if v.is_empty() {
            return Err(format!("{} 缺少取值", flag));
        }
        return Ok(v.to_string());
    }
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{} 缺少取值", flag))
}

fn num<T: FromStr>(raw: &str, flag: &str) -> Result<T, String> {
    raw.parse::<T>()
        .map_err(|_| format!("{} 的取值无效: {}", flag, raw))
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        process::exit(0);
    }

    // 先扫描预设，作为其余参数的默认值
    let mut preset = String::from("photo");
    let mut j = 0usize;
    while j < args.len() {
        let a = args[j].as_str();
        if a == "-P" || a == "--preset" {
            if let Some(v) = args.get(j + 1) {
                preset = v.clone();
            }
        } else if let Some(v) = a.strip_prefix("--preset=") {
            preset = v.to_string();
        }
        j += 1;
    }
    let (def_sigma, def_eps, def_denoise, def_phi) = match preset.as_str() {
        "photo" => (1.4f32, -0.05f32, 0.45f32, 15.0f32),
        "anime" => (1.2f32, -0.045f32, 0.30f32, 15.0f32),
        "ink" => (1.0f32, -0.03f32, 0.15f32, 25.0f32),
        other => {
            return Err(format!(
                "未知预设: {}（可选 photo / anime / ink）",
                other
            ))
        }
    };

    let mut positional: Vec<String> = Vec::new();
    let mut method: Option<Method> = None;
    let mut sigma: Option<f32> = None;
    let mut k = 1.6f32;
    let mut epsilon = def_eps;
    let mut phi = def_phi;
    let mut denoise = def_denoise;
    let mut canny_high = 0.09f32;
    let mut flow_sigma = 4.0f32;
    let mut min_area = 0usize;
    let mut scale = 1.0f32;
    let mut invert = false;
    let mut binarize = false;

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        let (key, inline) = match arg.split_once('=') {
            Some((a, b)) => (a, Some(b)),
            None => (arg, None),
        };
        match key {
            "-P" | "--preset" => {
                // 已在前面扫描并应用，这里只消费取值
                let _ = take_value(&args, &mut i, inline, key)?;
            }
            "-m" | "--method" => {
                let v = take_value(&args, &mut i, inline, key)?;
                method = Some(match v.to_ascii_lowercase().as_str() {
                    "xdog" => Method::Xdog,
                    "coherent" | "cld" | "fdog" => Method::Coherent,
                    "canny" => Method::Canny,
                    "sketch" | "pencil" => Method::Sketch,
                    other => {
                        return Err(format!(
                            "未知方法: {}（可选 xdog / coherent / canny / sketch）",
                            other
                        ))
                    }
                });
            }
            "-s" | "--sigma" => {
                let v = take_value(&args, &mut i, inline, key)?;
                sigma = Some(num::<f32>(&v, key)?);
            }
            "-k" | "--k" => {
                let v = take_value(&args, &mut i, inline, key)?;
                k = num::<f32>(&v, key)?;
            }
            "-e" | "--epsilon" => {
                let v = take_value(&args, &mut i, inline, key)?;
                epsilon = num::<f32>(&v, key)?;
            }
            "-p" | "--phi" => {
                let v = take_value(&args, &mut i, inline, key)?;
                phi = num::<f32>(&v, key)?;
            }
            "-d" | "--denoise" => {
                let v = take_value(&args, &mut i, inline, key)?;
                denoise = num::<f32>(&v, key)?;
            }
            "-t" | "--threshold" => {
                let v = take_value(&args, &mut i, inline, key)?;
                canny_high = num::<f32>(&v, key)?;
            }
            "-f" | "--flow" => {
                let v = take_value(&args, &mut i, inline, key)?;
                flow_sigma = num::<f32>(&v, key)?;
            }
            "--min-area" => {
                let v = take_value(&args, &mut i, inline, key)?;
                min_area = num::<usize>(&v, key)?;
            }
            "--scale" => {
                let v = take_value(&args, &mut i, inline, key)?;
                scale = num::<f32>(&v, key)?;
            }
            "--invert" => invert = true,
            "--binarize" => binarize = true,
            other if other.starts_with('-') => {
                return Err(format!("未知选项: {}", other));
            }
            _ => positional.push(args[i].clone()),
        }
        i += 1;
    }

    if positional.len() != 2 {
        return Err(format!(
            "需要输入和输出两个路径，当前 {} 个",
            positional.len()
        ));
    }

    let method = method.unwrap_or(Method::Xdog);
    let sigma = sigma.unwrap_or(def_sigma);

    if !sigma.is_finite() || sigma <= 0.0 || sigma > 50.0 {
        return Err("sigma 必须是 (0, 50] 的数值".into());
    }
    if !k.is_finite() || k <= 1.0 || k > 10.0 {
        return Err("k 必须是 (1, 10] 的数值".into());
    }
    if !epsilon.is_finite() || !(-0.5..=0.0).contains(&epsilon) {
        return Err("epsilon 必须在 -0.5 到 0 之间（越接近 0 线条越多）".into());
    }
    if !phi.is_finite() || phi <= 0.0 || phi > 500.0 {
        return Err("phi 必须是 (0, 500] 的数值".into());
    }
    if !denoise.is_finite() || !(0.0..=1.0).contains(&denoise) {
        return Err("denoise 必须在 0 到 1 之间".into());
    }
    if !canny_high.is_finite() || !(0.0..=1.0).contains(&canny_high) {
        return Err("threshold 必须在 0 到 1 之间".into());
    }
    if !flow_sigma.is_finite() || flow_sigma < 0.5 || flow_sigma > 50.0 {
        return Err("flow 必须在 [0.5, 50] 之间".into());
    }
    if !scale.is_finite() || scale <= 0.0 || scale > 4.0 {
        return Err("scale 必须是 (0, 4] 的数值".into());
    }

    Ok(Config {
        input: positional.remove(0),
        output: positional.remove(0),
        method,
        sigma,
        k,
        epsilon,
        phi,
        denoise,
        canny_high,
        flow_sigma,
        min_area,
        scale,
        invert,
        binarize,
    })
}

fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("错误: {}", e);
            eprintln!("使用 --help 查看用法");
            process::exit(1);
        }
    };

    let started = Instant::now();

    let img = match image::open(&cfg.input) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("无法读取图片 '{}': {}", cfg.input, e);
            process::exit(1);
        }
    };

    let mut gray: GrayImage = img.to_luma8();
    if (cfg.scale - 1.0).abs() > 1e-3 {
        let nw = ((gray.width() as f32 * cfg.scale).round() as u32).max(1);
        let nh = ((gray.height() as f32 * cfg.scale).round() as u32).max(1);
        gray = image::imageops::resize(&gray, nw, nh, image::imageops::FilterType::Lanczos3);
    }

    let (width, height) = gray.dimensions();
    let (w, h) = (width as usize, height as usize);
    let mut src: Vec<f32> = gray.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

    // 分辨率归一化：sigma、降噪半径等以像素为单位的量按长边相对 1920 缩放，
    // 使同一套参数在 1080p / 2K / 4K 上都得到一致的视觉线宽与降噪强度。
    let res = (w.max(h) as f32 / 1920.0).clamp(0.3, 4.0);
    let sigma = cfg.sigma * res;

    // 预处理：保边降噪，压制纹理型杂线
    if cfg.denoise > 0.001 {
        let radius = ((1.0 + 7.0 * cfg.denoise) * res).round().max(1.0) as usize;
        let eps = (0.01 + 0.18 * cfg.denoise).powi(2);
        src = blur::guided_filter(&src, w, h, radius, eps);
    }

    let mut out = match cfg.method {
        Method::Xdog => methods::xdog(&src, w, h, sigma, cfg.k, cfg.epsilon, cfg.phi),
        Method::Coherent => methods::coherent(
            &src,
            w,
            h,
            sigma,
            cfg.k,
            cfg.epsilon,
            cfg.phi,
            2.0 * res,
            cfg.flow_sigma * res,
        ),
        Method::Canny => {
            methods::canny(&src, w, h, sigma, cfg.canny_high * 0.4, cfg.canny_high)
        }
        Method::Sketch => methods::sketch(&src, w, h, sigma, cfg.k, cfg.epsilon, cfg.phi),
    };

    if cfg.min_area > 0 {
        let area = ((cfg.min_area as f32) * res * res).round().max(1.0) as usize;
        post::remove_small_components(&mut out, w, h, area);
    }

    let raw: Vec<u8> = out
        .iter()
        .map(|&v| {
            let mut v = v;
            if cfg.binarize {
                v = if v < 0.5 { 0.0 } else { 1.0 };
            }
            if cfg.invert {
                v = 1.0 - v;
            }
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();

    let output: GrayImage = match ImageBuffer::from_raw(width, height, raw) {
        Some(b) => b,
        None => {
            eprintln!("内部错误：输出缓冲区尺寸不匹配");
            process::exit(1);
        }
    };

    if let Err(e) = output.save(&cfg.output) {
        eprintln!("无法保存图片 '{}': {}", cfg.output, e);
        process::exit(1);
    }

    let name = match cfg.method {
        Method::Xdog => "XDoG 线稿",
        Method::Coherent => "相干线稿",
        Method::Canny => "Canny 细线",
        Method::Sketch => "铅笔素描",
    };
    println!(
        "完成 [{}] {}x{} -> {}  ({:.0} ms)",
        name,
        width,
        height,
        cfg.output,
        started.elapsed().as_millis()
    );
}
