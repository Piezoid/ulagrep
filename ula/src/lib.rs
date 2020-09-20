#![feature(stdsimd, core_intrinsics)]


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse;

#[cfg(test)]
mod test_utils;

use std::is_x86_feature_detected;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Match {
    pub pos: usize,
    pub delta: i8,
    pub dist: u8,
}

impl Match {
    pub fn end(&self) -> usize {
        (self.pos as isize + self.delta as isize) as usize
    }
}

type Matches = Vec<Match>;
#[inline(always)]
fn push_match(vec: &mut Matches, m: Match) {
    vec.push(m)
}


pub fn search(k: usize, pat: &[u8], txt: &[u8], res: &mut Matches) {
    assert!(
        pat.len() > k as usize,
        "The pattern must be at least k+1={} chars long",
        k + 1
    );
    if is_x86_feature_detected!("sse4.1") && k <= sse::MAXK {
        unsafe { sse::search(k, pat, txt, res) }
    }
}

