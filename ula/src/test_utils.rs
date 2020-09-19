use crate::*;
type BoxStr = Vec<u8>;

fn pad(txt: &[u8], pad: char, l: usize, r: usize) -> BoxStr {
    [
        &vec![pad as u8; l as usize],
        txt,
        &vec![pad as u8; r as usize],
    ]
    .concat()
}

fn forall_mutations<F: FnMut(&[u8], BoxStr, usize, YInt)>(max_dist: usize, mut f: F) {
    let pat = b"ABCDEFGHIJKLABCDEFGHIJKLABCDEFGHIJKL";
    let subs = b"abcdefghijklabcdefghijklabcdefghijkl";

    for m in 1..pat.len() {
        let pat = &pat[..m];

        // dist=0: identity case
        f(pat, pat.as_ref().into(), 0, 0);

        for dist in 1..=(max_dist as usize).min(m - 1) {
            // Position. 0..(m - dist) <= Must end with at least one match
            for pos in 0..(m - dist) {
                for nsubs in 0..=dist {
                    assert!(pos < m - dist);
                    let subs = &subs[pos..pos + nsubs];
                    let delta_abs = dist - nsubs; // Length delta, either from insertions or deletions

                    // substitutions + deletions. No deletion allowed at the beginning,
                    if pos > 0 || delta_abs == 0 {
                        f(
                            pat,
                            [&pat[..pos], subs, &pat[pos + dist..]].concat(),
                            dist as usize,
                            -(delta_abs as isize),
                        );
                    }

                    // substitutions + insertions
                    if delta_abs > 0 // Only when at least one insertion (not duplicate of above)
                    && pos >= delta_abs
                    // Starts with at least delta_abs matches
                    {
                        f(
                            pat,
                            [
                                &pat[..pos],
                                subs,
                                &vec!['%' as u8; delta_abs],
                                &pat[pos + nsubs..],
                            ]
                            .concat(),
                            dist as usize,
                            delta_abs as isize,
                        );
                    }
                }
            }
        }
    }
}

pub fn test_simple_patterns(
    max_k: usize,
    max_padding: usize,
    search: unsafe fn(usize, &[u8], &[u8], &mut super::Matches),
) {
    let mut npat = 0;
    let mut res = Vec::new();

    forall_mutations(max_k, |pat, txt, dist, delta| {
        for lpad in 0..max_padding {
            for rpad in 0..max_padding {
                let txt = &pad(&txt, '$', lpad, rpad);
                //println!("m:{:02} k:{} d:{:+} {}", pat.len(), dist, delta, str::from_utf8(&txt).unwrap());
                debug_only! {
                println!(
                    "pos dist={} delta={} txt={:?} pad={},{}",
                    dist,
                    delta,
                    str::from_utf8(&txt).unwrap(),
                    lpad,
                    rpad,
                );
                }

                unsafe { search(dist as usize, pat, txt, &mut res) };
                assert!(res.len() > 0);
                debug_only!(println!("{:?}", res));
                for occ in res.iter() {
                    assert_eq!(occ.pos as usize, lpad);
                    assert_eq!(occ.dist as usize, dist);
                    assert_eq!(occ.delta as isize, delta);
                    break;
                }
                if res.len() > 1 && res[0].end() != res[1].end() {
                    println!("k:{}, pat:{:?} txt:{:?} {:?}", dist, std::str::from_utf8(pat).unwrap(),std::str::from_utf8(txt).unwrap(), res);
                }
                res.clear();
                if dist > 0 {
                    debug_only! {
                        println!(
                            "neg dist={} delta={} txt={:?}",
                            dist,
                            delta,
                            str::from_utf8(&txt).unwrap()
                        );
                    }
                    unsafe { search(dist as usize - 1, pat, &txt, &mut res) };
                    assert!(res.len() == 0);
                }
                npat += 1;
                debug_only!(println!("---------------"));
            }
        }
    });

    println!("npat:{}", npat);
}