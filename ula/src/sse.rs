use crate::*;
use std::arch::x86_64::*;
use std::intrinsics::cttz_nonzero;
use std::intrinsics::{likely, unlikely};

const LANES: usize = 16;
pub const MAXK: usize = LANES / 2 - 1; // Maximum number of error and position of column x=0 in the vector

fn sse_to_bytes(x: &__m128i) -> &[u8] {
    unsafe { std::slice::from_raw_parts((x as *const _) as *const u8, 16) }
}

fn sse_to_str(x: &__m128i) -> &str {
    std::str::from_utf8(sse_to_bytes(x)).unwrap()
}

#[target_feature(enable = "sse4.2")]
unsafe fn compare_sse(a: __m128i, b: __m128i) -> bool {
    _mm_test_all_ones(_mm_cmpeq_epi8(a, b)) != 0
}

#[inline(always)]
unsafe fn sse_max_index(x: &__m128i) -> (u8, u8) {
    let y = _mm_xor_si128(*x, _mm_set1_epi8(-1)); // Inverse ordering
    let mp_odd = _mm_minpos_epu16(y); // Compares odd u8 as in u16's high byte
    let mp_even = _mm_minpos_epu16(_mm_slli_epi16(y, 8));
    let min_odd = _mm_extract_epi8(mp_odd, 1); // high byte of u16s minimum
    let min_even = _mm_extract_epi8(mp_even, 1);
    let odd = min_odd <= min_even;
    let (mp, min) = if odd {
        (mp_odd, min_odd)
    } else {
        (mp_even, min_even)
    };

    let idx = 2 * _mm_extract_epi8(mp, 2) + odd as i32; // Converts u16 minimum indice to original u8 indices
    (idx as u8, 0xffu8 ^ min as u8)
}

#[target_feature(enable = "sse4.2")]
unsafe fn sse_idx() -> __m128i {
    _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
}

#[target_feature(enable = "sse4.2")]
unsafe fn init_ula2(k: usize) -> __m128i {
    let ula = _mm_subs_epu8(
        _mm_set1_epi8(1 + k as i8),
        _mm_abs_epi8(_mm_sub_epi8(_mm_set1_epi8(14 - k as i8), sse_idx())),
    );
    let p = (&ula as *const _) as *const u8;
    debug_assert_eq!(*p.add(LANES - 2 - k), (1 + k) as u8);
    ula
}

#[target_feature(enable = "sse4.2")]
unsafe fn init_ula_stack(k: usize, center: usize) -> __m128i {
    let mut vec = [0u8; LANES];
    vec[center as usize] = (k + 1) as u8;
    *((&vec as *const _) as *const __m128i)
}

#[target_feature(enable = "sse4.2")]
unsafe fn init_ula(k: usize, center: usize) -> __m128i {
    _mm_and_si128(
        _mm_cmpeq_epi8(sse_idx(), _mm_set1_epi8(center as i8)),
        _mm_set1_epi8((k + 1) as i8),
    )
}

/// Single iteration of NULA.
/// The parameter k is only used to bound the number of deletions.
/// Otherwise the algorithm is offset invariant (the x=0 column can be anywhere).
#[inline(always)]
unsafe fn ula_push_lightk(vec0: __m128i, u_: __m128i, k: usize) -> __m128i {
    let ones = _mm_set1_epi8(0x01);
    let dec_sat = |x| _mm_subs_epu8(x, ones);

    let vec_sub = dec_sat(vec0);
    let vec_ins = _mm_bsrli_si128(vec_sub, 1);

    let trues = _mm_set1_epi8(-1);
    let notu = _mm_xor_si128(u_, trues);

    let vec_ndelid = {
        let mut vacc = vec0;
        let mut vec0_shifted = _mm_bslli_si128(vec_sub, 1);

        for _ in 0..k {
            vacc = _mm_max_epu8(vacc, vec0_shifted);
            vec0_shifted = dec_sat(_mm_bslli_si128(vec0_shifted, 1));
        }
        vacc
    };

    // let vec = _mm_blendv_epi8(_mm_max_epu8(vec_ins, vec_sub), vec_ndelid, u_);
    let vec = _mm_max_epu8(
        _mm_subs_epu8(vec_ndelid, _mm_and_si128(notu, ones)),
        vec_ins,
    );
    vec
}

/// SSE simulation of NULA for a single extenssion with txt right-padded with zeroes
/// txt_vec is preloaded with a prefix of the text at the right offset (the algorithm in offset invariant)
/// txt contains the rest of the text.
#[inline(always)]
unsafe fn simula_slow(
    k: usize,
    pat: &[u8],
    mut ula: __m128i,
    mut txt_vec: __m128i,
    txt: &[u8],
) -> Option<(u8, u8)> {
    for i in 0..pat.len() {
        ula = ula_push_lightk(
            ula,
            _mm_cmpeq_epi8(txt_vec, _mm_set1_epi8(*pat.get_unchecked(i) as i8)),
            (i + 1).min(k) as usize,
        );

        if _mm_test_all_zeros(ula, ula) != 0 {
            return None;
        }

        txt_vec = _mm_bsrli_si128(txt_vec, 1);
        if i < txt.len() {
            txt_vec = _mm_insert_epi8(txt_vec, *txt.get_unchecked(i) as i32, LANES as i32 - 1);
        } // Otherwise shift in zeroes
    }

    Some(sse_max_index(&ula))
}

/// SSE simulation of NULA for a single extenssion with no padding performed, k must be greater than 0
#[inline(always)]
#[allow(unused_assignments)]
unsafe fn simula_fast_fat(
    k: usize,
    pat: &[u8],
    mut ula: __m128i,
    mut txt_win: TxtWin<'_>,
) -> Option<(u8, u8)> {
    debug_assert!(txt_win.it.as_slice().len() >= pat.len());
    debug_assert!(k <= MAXK && k > 0);
    std::intrinsics::assume(k <= MAXK && k > 0);

    let mut pat_ptr = pat.as_ptr();
    let pat_ptr_end = pat_ptr.add(pat.len());

    macro_rules! iter {
        ($k: expr) => {
            debug_assert!(pat_ptr < pat_ptr_end);
            ula = ula_push_lightk(
                ula,
                _mm_cmpeq_epi8(txt_win.buf, _mm_set1_epi8(*pat_ptr as i8)),
                $k,
            );
            std::intrinsics::assume(txt_win.it.clone().next().is_some());
            txt_win.next();
            pat_ptr = pat_ptr.add(1);
        };
    }
    macro_rules! iter_or_loop {
        ($j:expr) => {
            loop {
                iter!($j);
                if k == $j {
                    if _mm_test_all_zeros(ula, ula) != 0 {
                        return None;
                    }
                    if pat_ptr >= pat_ptr_end {
                        return Some(sse_max_index(&ula));
                    }
                    continue;
                }
                break;
            }
            debug_assert!(_mm_test_all_zeros(ula, ula) == 0);
        };
    }

    iter_or_loop!(1);
    iter_or_loop!(2);
    iter_or_loop!(3);
    iter_or_loop!(4);
    iter_or_loop!(5);
    iter_or_loop!(6);
    iter_or_loop!(7);
    std::intrinsics::unreachable()
}

/// Left pads bytes from `data` with `offset` zeros, returns a sse registers and the remaining suffix
#[target_feature(enable = "sse4.2")]
unsafe fn sse_load_offset(data: &[u8], offset: usize) -> __m128i {
    debug_assert!(offset < LANES);
    let n = data.len().min(LANES - offset);
    let mut reg = [0u8; LANES];
    std::ptr::copy_nonoverlapping(data.as_ptr(), reg.as_mut_ptr().add(offset), n);
    _mm_lddqu_si128(reg.as_ptr() as *const __m128i)
}

/// Compares pessimistically the pattern to the prefix of the text
/// The text must be at least as long as the pattern
#[inline(always)]
unsafe fn cmp(pat: &[u8], txt: &[u8]) -> bool {
    debug_assert!(pat.len() <= txt.len());
    for i in 0..pat.len() {
        if *pat.get_unchecked(i) != *txt.get_unchecked(i) {
            return false;
        }
    }
    true
}

use super::{Match, Matches};

struct TxtWin<'a> {
    buf: __m128i,
    it: std::slice::Iter<'a, u8>,
}

impl<'a> TxtWin<'a> {
    #[target_feature(enable = "sse4.2")]
    unsafe fn new(txt: &'a [u8], offset: usize) -> Self {
        debug_assert!(offset <= LANES);
        Self {
            buf: sse_load_offset(txt, offset),
            it: txt[(LANES - offset).min(txt.len())..].iter(),
        }
    }

    #[target_feature(enable = "sse4.2")]
    unsafe fn cmp_as_mask(&self, y: __m128i) -> usize {
        _mm_movemask_epi8(_mm_cmpeq_epi8(self.buf, y)) as usize
    }
}

impl<'a> Iterator for TxtWin<'a> {
    type Item = Self;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let pre_it = self.it.clone();
        let pre_buf = self.buf;
        let shift_buf = unsafe { _mm_bsrli_si128(pre_buf, 1) };
        self.buf = if let Some(c) = self.it.next() {
            unsafe { _mm_insert_epi8(shift_buf, *c as i32, LANES as i32 - 1) }
        } else {
            shift_buf
        };

        Some(Self {
            buf: pre_buf,
            it: pre_it,
        })
    }
}

#[inline(never)]
#[target_feature(enable = "sse4.2")]
pub unsafe fn search(k: usize, pat: &[u8], txt: &[u8], res: &mut Matches) {
    debug_assert!(pat.len() > k);
    debug_assert!(k <= MAXK);
    std::intrinsics::assume(k <= MAXK);
    std::intrinsics::assume(pat.len() > k);

    // MAXK-1: Position of the first character of the pattern in SIMD reg.
    // When the n-th char matches, the center of the NULA window (x=0) will be at position MAXK+n
    let patoffset = MAXK - 1;
    let pat_prefix_vec = sse_load_offset(pat, patoffset);
    //let mut txt_vec = sse_load_offset(txt, patoffset);
    // How many chars are preloaded in txt_vec, The pointer in txt is txt_vec_len chars after the current position.
    let txt_vec_len = (LANES - patoffset) as usize;
    let pat_vec_len = pat.len().min(txt_vec_len);

    // It is somewhat expensive to place a single byte inside a SSE register at a variable location
    // So we precompute on the stack the initial states of the automaton for each offset
    let mut ula0 = [_mm_set1_epi8(0); MAXK + 1];
    {
        let ptr = (&mut ula0 as *mut _) as *mut u8;
        for offset in 0..=k {
            *ptr.add(offset * LANES + MAXK + offset) = (k + 1 - offset) as u8
        }
    };

    // When there is no allowed errors after the first match (offset=k), the strict equality
    // can be checked on the pat_prefix_vec part by comparing the equality bitevector (bveq) to:
    // Note: shifting bits left means going forward in the text.
    let eqkeff0_mask = ((1 << (pat_vec_len - k)) - 1) << k;

    // While txt.len() - pos >= pat.len() - 1, we can bound NULA iterations with the pattern length
    // simula_fast() unrolls the k+1 first iterations, that requires that the pattern length,
    // minus 1chr for the minimal match of 1 char, to be greater than k
    let fast_iters_end = if txt.len() >= pat.len() + txt_vec_len - 1 && pat.len() > k + 1 {
        txt.len() - (pat.len() - 1) - txt_vec_len
    } else {
        0
    };

    let mut it = TxtWin::new(txt, patoffset).enumerate();
    for (pos, win) in (&mut it).take(fast_iters_end) {
        let bveq = win.cmp_as_mask(pat_prefix_vec) as usize >> patoffset;
        let offset = cttz_nonzero(bveq);

        // If one of the k+1 chars of the pattern matches with the current suffixes
        let maybe_match = if bveq != 0 {
            if k > offset {
                let ula = _mm_load_si128((&ula0 as *const __m128i).add(offset));
                let keff = k - offset;
                if let Some((idx, score)) =
                    simula_fast_fat(keff, pat.get_unchecked(offset + 1..), ula, win)
                {
                    Some(Match {
                        pos: pos,
                        delta: idx as i8 - (MAXK + offset) as i8,
                        dist: (k + 1) as u8 - score,
                    })
                } else {
                    None
                }
            } else {
                if bveq == eqkeff0_mask
                    && (pat.len() <= txt_vec_len
                        || cmp(pat.get_unchecked(txt_vec_len..), win.it.as_slice()))
                {
                    // Case offset == k. See note on eqkeff0_mask declaration
                    debug_assert!(cmp(&pat[offset..], &txt[offset + pos..]));
                    Some(Match {
                        pos: pos,
                        delta: 0,
                        dist: k as u8,
                    })
                } else {
                    None
                }
            }
        } else {
            None
        };

        if let Some(m) = maybe_match {
            crate::push_match(res, m);
        }
    }

    let slow_iter_end = txt.len() + k + 1 - pat.len();
    for (pos, win) in it.take(slow_iter_end - fast_iters_end) {
        let bveq = win.cmp_as_mask(pat_prefix_vec) as usize >> patoffset;
        let offset = cttz_nonzero(bveq);

        if likely(bveq != 0 && offset <= k) {
            let keff = k - offset;
            let ula = init_ula(keff, MAXK + offset);
            if let Some((idx, score)) = simula_slow(
                keff,
                pat.get_unchecked(offset + 1..),
                ula,
                win.buf,
                win.it.as_slice(),
            ) {
                res.push(Match {
                    pos: pos,
                    delta: idx as i8 - (MAXK + offset) as i8,
                    dist: (k + 1) as u8 - score,
                });
            }

            debug_only!(println!());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exhaustive_sse() {
        crate::test_utils::test_simple_patterns(MAXK, LANES, search);
    }
}
