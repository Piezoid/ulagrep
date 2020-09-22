use std::time::Duration;

use criterion::*;

use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use triple_accel::levenshtein::{levenshtein_search_simd_with_opts, LEVENSHTEIN_COSTS};
use triple_accel::SearchType;

pub use ula::*;

use std::io::Result;
use std::path::Path;
fn get_url_data<'a, P: AsRef<Path>>(url: &'a str, out_dir: P) -> Result<(Vec<u8>, &'a str)> {
    use std::io::Read;
    use std::io::{Error, ErrorKind};

    let out_dir = out_dir.as_ref();
    let invalid_url_lazy =
        || Error::new(ErrorKind::InvalidInput, format!("Invalid url: {:?}", url));
    let filename = url.rsplit('/').next().ok_or_else(invalid_url_lazy)?;
    let name = filename.split('.').next().ok_or_else(invalid_url_lazy)?;
    let path = out_dir.join(filename);

    if !path.exists() {
        if !out_dir.exists() {
            std::fs::create_dir_all(out_dir)?;
        }

        std::process::Command::new("curl")
            .arg("-L")
            .arg(url)
            .arg("-o")
            .arg(path.as_os_str())
            .status()
            .and_then(|s| {
                if s.success() {
                    Ok(())
                } else {
                    Err(Error::new(
                        ErrorKind::Other,
                        format!("Curl exited with {:?}", s.code()),
                    ))
                }
            })?;
    }

    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    Ok((buf, name))
}

fn bench_corpora(c: &mut Criterion) {
    for url in [
        "https://github.com/smart-tool/smart/raw/master/data/genome/ecoli.txt",
        "https://github.com/smart-tool/smart/raw/master/data/protein/sc.txt",
        "https://github.com/smart-tool/smart/raw/master/data/englishTexts/bible.txt",
    ]
    .iter()
    {
        bench_data(c, url, 1 << 19);
    }
}

fn bench_data(c: &mut Criterion, url: &str, size: usize) {
    let (data, name) = get_url_data(url, "../data").expect("Could not get benchmark data");
    let size = data.len().min(size);
    let data = &data[..size];
    let mut results = Vec::with_capacity(256);

    let mut group = c.benchmark_group(name);
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Bytes(size as u64));
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(50);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for k in 1..=7 {
        for &len in [1, 2, 4, 6, 8, 10, 16, 32, 64, 128, 256, 512].iter() {
            if len < k + 1 {
                continue;
            }
            let arg_str = format!("k{}_m{:03}", k, len);

            let mut pats = Uniform::from(0..(size - len))
                .sample_iter(SmallRng::seed_from_u64(0))
                .map(|i| &data[i..i + len]);

            group.bench_with_input(BenchmarkId::new("ULA_SSE", &arg_str), &k, |b, &k| {
                b.iter(|| {
                    results.clear();
                    ula::search(
                        black_box(k as usize),
                        black_box(pats.next().unwrap()),
                        black_box(&data),
                        black_box(&mut results),
                    );
                    debug_assert!(!results.is_empty());
                    results.len()
                });
            });

            group.bench_with_input(BenchmarkId::new("triple_accel", &arg_str), &k, |b, &k| {
                b.iter(|| {
                    levenshtein_search_simd_with_opts(
                        black_box(pats.next().unwrap()),
                        black_box(&data),
                        black_box(k as u32),
                        black_box(SearchType::All),
                        black_box(LEVENSHTEIN_COSTS),
                        black_box(false),
                    )
                    .last()
                });
            });

            // group.bench_with_input(
            //     BenchmarkId::new("agrep", format!("k{}_m{}", k, len)),
            //     &k,
            //     |b, &k| {
            //         b.iter(|| {
            //             use std::ffi::OsStr;
            //             use std::io::Write;
            //             use std::os::unix::ffi::OsStrExt;
            //             use std::process::{Command, Stdio};
            //             let pat = pats.next().unwrap();
            //             let mut agrep = std::process::Command::new("agrep")
            //                 .stdin(Stdio::piped())
            //                 .stdout(Stdio::piped())
            //                 .arg("--count")
            //                 .arg(format!("--max-errors={}", k))
            //                 .arg(OsStr::from_bytes(pat))
            //                 .spawn()
            //                 .expect("unable to run agrep");
            //             {
            //                 // limited borrow of stdin
            //                 let stdin = agrep.stdin.as_mut().expect("failed to get stdin");
            //                 stdin.write_all(data).expect("failed to write to stdin");
            //             }
            //             let out = agrep.wait_with_output().expect("failed to wait on child");
            //             println!("agrep:{:?} pat:{:?}", out, std::str::from_utf8(pat).unwrap());
            //         });
            //     },
            // );
        }
    }
    group.finish();
}

criterion_group!(benchg, bench_corpora);
criterion_main!(benchg);
