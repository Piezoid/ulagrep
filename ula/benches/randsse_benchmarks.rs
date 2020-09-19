use criterion::*;

use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use triple_accel::levenshtein::{levenshtein_search_simd_with_opts, LEVENSHTEIN_COSTS};
use triple_accel::SearchType;

pub use ula::*;

pub fn make_random_string(seed: u64, size: usize, card: u8) -> Box<[u8]> {
    assert!(card > 0);
    Uniform::from(97u8..97u8.checked_add(card).unwrap())
        .sample_iter(SmallRng::seed_from_u64(seed))
        .take(size)
        .collect()
}

fn bench_random(c: &mut Criterion) {
    const N: usize = 4 << 20;
    let txt = black_box(make_random_string(0, N, 4));
    let pat = black_box(make_random_string(1, 20, 4));

    for k in [7].iter().cloned() {
        let mut g = c.benchmark_group(format!("random_string_k{}", k));
        g.throughput(Throughput::Bytes(N as u64));

        let mut vec = Vec::new();
        g.bench_function("ulasse", |b| {
            b.iter(|| {
                vec.clear();
                ula::search(k, pat.as_ref(), txt.as_ref(), &mut vec);
                vec.len();
            })
        });

        g.finish();
    }
}

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

fn bench_genome(c: &mut Criterion) {
    bench_data(
        c,
        "https://github.com/smart-tool/smart/raw/master/data/genome/ecoli.txt",
        1 << 20,
    )
}

fn bench_data(c: &mut Criterion, url: &str, size: usize) {
    let (data, name) = get_url_data(url, "../data").expect("Could not get benchmark data");
    let size = data.len().min(size);
    let data = &data[..size];

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(size as u64));

    let mut results = Vec::new();

    for k in 1..=7 {
        for &len in [2, 4, 8, 16, 32, 64, 128, 256, 512].iter() {
            if len < k+1 {
                continue;
            }
            group.bench_with_input(
                BenchmarkId::new("ULA_SSE", format!("k{}_m{}", k, len)),
                &k,
                |b, &k| {
                    let mut pats = Uniform::from(0..(size - len))
                        .sample_iter(SmallRng::seed_from_u64(0))
                        .map(|i| &data[i..i + len]);
                    b.iter(|| {
                        results.clear();
                        ula::search(k as usize, pats.next().unwrap(), &data, &mut results);
                        debug_assert!(results.len() > 0);
                        results.len()
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("triple_accel", format!("k{}_m{}", k, len)),
                &k,
                |b, &k| {
                    let mut pats = Uniform::from(0..(size - len))
                        .sample_iter(SmallRng::seed_from_u64(0))
                        .map(|i| &data[i..i + len]);

                    b.iter(|| {
                        levenshtein_search_simd_with_opts(
                            pats.next().unwrap(),
                            &data,
                            k as u32,
                            SearchType::All,
                            LEVENSHTEIN_COSTS,
                            false,
                        )
                        .last()
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benchg, bench_genome);
criterion_main!(benchg);
