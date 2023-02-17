use aligned_vec::{avec, ABox};

use crate::{
    bit_rev,
    fastdiv::{Div32, Div64},
    roots::find_primitive_root64,
    FwdUpperBound, InvUpperBound,
};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const RECURSION_THRESHOLD: usize = 2048;

mod generic;
mod shoup;

mod less_than_30bit;
mod less_than_31bit;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl crate::Avx2 {
    #[inline(always)]
    fn interleave4_epu32(self, z0z0z0z0z1z1z1z1: [__m256i; 2]) -> [__m256i; 2] {
        let avx = self.avx;
        [
            avx._mm256_permute2f128_si256::<0b00100000>(z0z0z0z0z1z1z1z1[0], z0z0z0z0z1z1z1z1[1]),
            avx._mm256_permute2f128_si256::<0b00110001>(z0z0z0z0z1z1z1z1[0], z0z0z0z0z1z1z1z1[1]),
        ]
    }

    #[inline(always)]
    fn permute4_epu32(self, w: [u32; 2]) -> __m256i {
        let avx = self.avx;
        let w0 = self.sse2._mm_set1_epi32(w[0] as i32);
        let w1 = self.sse2._mm_set1_epi32(w[1] as i32);
        avx._mm256_insertf128_si256::<1>(avx._mm256_castsi128_si256(w0), w1)
    }

    #[inline(always)]
    fn interleave2_epu32(self, z0z0z1z1: [__m256i; 2]) -> [__m256i; 2] {
        [
            self.avx2._mm256_unpacklo_epi64(z0z0z1z1[0], z0z0z1z1[1]),
            self.avx2._mm256_unpackhi_epi64(z0z0z1z1[0], z0z0z1z1[1]),
        ]
    }

    #[inline(always)]
    fn permute2_epu32(self, w: [u32; 4]) -> __m256i {
        let avx = self.avx;
        let w0123 = pulp::cast(w);
        let w0022 = self.sse2._mm_castps_si128(self.sse3._mm_moveldup_ps(w0123));
        let w1133 = self.sse2._mm_castps_si128(self.sse3._mm_movehdup_ps(w0123));
        avx._mm256_insertf128_si256::<1>(avx._mm256_castsi128_si256(w0022), w1133)
    }

    #[inline(always)]
    fn interleave1_epu32(self, z0z0z1z1: [__m256i; 2]) -> [__m256i; 2] {
        let avx = self.avx2;
        let x = [
            // 00 10 01 11 04 14 05 15 08 18 09 19 0c 1c 0d 1d
            avx._mm256_unpacklo_epi32(z0z0z1z1[0], z0z0z1z1[1]),
            // 02 12 03 13 06 16 07 17 0a 1a 0b 1b 0e 1e 0f 1f
            avx._mm256_unpackhi_epi32(z0z0z1z1[0], z0z0z1z1[1]),
        ];
        [
            // 00 10 02 12 04 14 06 16 08 18 0a 1a 0c 1c 0c 1c
            avx._mm256_unpacklo_epi64(x[0], x[1]),
            // 01 11 03 13 05 15 07 17 09 19 0b 1b 0d 1d 0f 1f
            avx._mm256_unpackhi_epi64(x[0], x[1]),
        ]
    }

    #[inline(always)]
    fn permute1_epu32(self, w: [u32; 8]) -> __m256i {
        let avx = self.avx;
        let [w0123, w4567]: [__m128i; 2] = pulp::cast(w);
        let w0415 = self.sse2._mm_unpacklo_epi32(w0123, w4567);
        let w2637 = self.sse2._mm_unpackhi_epi32(w0123, w4567);
        avx._mm256_insertf128_si256::<1>(avx._mm256_castsi128_si256(w0415), w2637)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl crate::Avx512 {
    #[inline(always)]
    fn interleave8_epu32(self, z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1: [__m512i; 2]) -> [__m512i; 2] {
        let avx = self.avx512f;
        let idx_0 = avx._mm512_setr_epi32(
            0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        );
        let idx_1 = avx._mm512_setr_epi32(
            0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
        );
        [
            avx._mm512_permutex2var_epi32(
                z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1[0],
                idx_0,
                z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1[1],
            ),
            avx._mm512_permutex2var_epi32(
                z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1[0],
                idx_1,
                z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1[1],
            ),
        ]
    }

    #[inline(always)]
    fn permute8_epu32(self, w: [u32; 2]) -> __m512i {
        let avx = self.avx512f;
        let w = pulp::cast(w);
        let w01xxxxxxxxxxxxxx = avx._mm512_set1_epi64(w);
        let idx = avx._mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        avx._mm512_permutexvar_epi32(idx, w01xxxxxxxxxxxxxx)
    }

    #[inline(always)]
    fn interleave4_epu32(self, z0z0z0z0z1z1z1z1: [__m512i; 2]) -> [__m512i; 2] {
        let avx = self.avx512f;
        let idx_0 = avx._mm512_setr_epi32(
            0x0, 0x1, 0x2, 0x3, 0x10, 0x11, 0x12, 0x13, 0x8, 0x9, 0xa, 0xb, 0x18, 0x19, 0x1a, 0x1b,
        );
        let idx_1 = avx._mm512_setr_epi32(
            0x4, 0x5, 0x6, 0x7, 0x14, 0x15, 0x16, 0x17, 0xc, 0xd, 0xe, 0xf, 0x1c, 0x1d, 0x1e, 0x1f,
        );
        [
            avx._mm512_permutex2var_epi32(z0z0z0z0z1z1z1z1[0], idx_0, z0z0z0z0z1z1z1z1[1]),
            avx._mm512_permutex2var_epi32(z0z0z0z0z1z1z1z1[0], idx_1, z0z0z0z0z1z1z1z1[1]),
        ]
    }

    #[inline(always)]
    fn permute4_epu32(self, w: [u32; 4]) -> __m512i {
        let avx = self.avx512f;
        let w = pulp::cast(w);
        let w0123xxxxxxxxxxxx = avx._mm512_castsi128_si512(w);
        let idx = avx._mm512_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3);
        avx._mm512_permutexvar_epi32(idx, w0123xxxxxxxxxxxx)
    }

    #[inline(always)]
    fn interleave2_epu32(self, z0z0z1z1: [__m512i; 2]) -> [__m512i; 2] {
        let avx = self.avx512f;
        [
            // 00 01 10 11 04 05 14 15 08 09 18 19 0c 0d 1c 1d
            avx._mm512_unpacklo_epi64(z0z0z1z1[0], z0z0z1z1[1]),
            // 02 03 12 13 06 07 16 17 0a 0b 1a 1b 0e 0f 1e 1f
            avx._mm512_unpackhi_epi64(z0z0z1z1[0], z0z0z1z1[1]),
        ]
    }

    #[inline(always)]
    fn permute2_epu32(self, w: [u32; 8]) -> __m512i {
        let avx = self.avx512f;
        let w = pulp::cast(w);
        let w01234567xxxxxxxx = avx._mm512_castsi256_si512(w);
        let idx = avx._mm512_setr_epi32(0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
        avx._mm512_permutexvar_epi32(idx, w01234567xxxxxxxx)
    }

    #[inline(always)]
    fn interleave1_epu32(self, z0z0z1z1: [__m512i; 2]) -> [__m512i; 2] {
        let avx = self.avx512f;
        let x = [
            // 00 10 01 11 04 14 05 15 08 18 09 19 0c 1c 0d 1d
            avx._mm512_unpacklo_epi32(z0z0z1z1[0], z0z0z1z1[1]),
            // 02 12 03 13 06 16 07 17 0a 1a 0b 1b 0e 1e 0f 1f
            avx._mm512_unpackhi_epi32(z0z0z1z1[0], z0z0z1z1[1]),
        ];
        [
            // 00 10 02 12 04 14 06 16 08 18 0a 1a 0c 1c 0c 1c
            avx._mm512_unpacklo_epi64(x[0], x[1]),
            // 01 11 03 13 05 15 07 17 09 19 0b 1b 0d 1d 0f 1f
            avx._mm512_unpackhi_epi64(x[0], x[1]),
        ]
    }

    #[inline(always)]
    fn permute1_epu32(self, w: [u32; 16]) -> __m512i {
        let avx = self.avx512f;
        let w = pulp::cast(w);
        let idx = avx._mm512_setr_epi32(
            0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb, 0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf,
        );
        avx._mm512_permutexvar_epi32(idx, w)
    }
}

fn init_negacyclic_twiddles(p: u32, n: usize, twid: &mut [u32], inv_twid: &mut [u32]) {
    let div = Div32::new(p);
    let w = find_primitive_root64(Div64::new(p as u64), 2 * n as u64).unwrap() as u32;
    let mut k = 0;
    let mut wk = 1u32;

    let nbits = n.trailing_zeros();
    while k < n {
        let fwd_idx = bit_rev(nbits, k);

        twid[fwd_idx] = wk;

        let inv_idx = bit_rev(nbits, (n - k) % n);
        if k == 0 {
            inv_twid[inv_idx] = wk;
        } else {
            let x = p.wrapping_sub(wk);
            inv_twid[inv_idx] = x;
        }

        wk = Div32::rem_u64(wk as u64 * w as u64, div);
        k += 1;
    }
}

fn init_negacyclic_twiddles_shoup(
    p: u32,
    n: usize,
    twid: &mut [u32],
    twid_shoup: &mut [u32],
    inv_twid: &mut [u32],
    inv_twid_shoup: &mut [u32],
) {
    let div = Div32::new(p);
    let w = find_primitive_root64(Div64::new(p as u64), 2 * n as u64).unwrap() as u32;
    let mut k = 0;
    let mut wk = 1u32;

    let nbits = n.trailing_zeros();
    while k < n {
        let fwd_idx = bit_rev(nbits, k);

        let wk_shoup = Div32::div_u64((wk as u64) << 32, div) as u32;
        twid[fwd_idx] = wk;
        twid_shoup[fwd_idx] = wk_shoup;

        let inv_idx = bit_rev(nbits, (n - k) % n);
        if k == 0 {
            inv_twid[inv_idx] = wk;
            inv_twid_shoup[inv_idx] = wk_shoup;
        } else {
            let x = p.wrapping_sub(wk);
            inv_twid[inv_idx] = x;
            inv_twid_shoup[inv_idx] = Div32::div_u64((x as u64) << 32, div) as u32;
        }

        wk = Div32::rem_u64(wk as u64 * w as u64, div);
        k += 1;
    }
}

#[derive(Debug, Clone)]
pub struct Plan {
    twid: ABox<[u32]>,
    twid_shoup: ABox<[u32]>,
    inv_twid: ABox<[u32]>,
    inv_twid_shoup: ABox<[u32]>,
    p: u32,
    p_div: Div32,
}

impl Plan {
    pub fn try_new(n: usize, p: u32) -> Option<Self> {
        let p_div = Div32::new(p);
        if find_primitive_root64(Div64::new(p as u64), 2 * n as u64).is_none() {
            None
        } else {
            let mut twid = avec![0u32; n].into_boxed_slice();
            let mut inv_twid = avec![0u32; n].into_boxed_slice();
            let (mut twid_shoup, mut inv_twid_shoup) = if p < (1u32 << 31) {
                (
                    avec![0u32; n].into_boxed_slice(),
                    avec![0u32; n].into_boxed_slice(),
                )
            } else {
                (avec![].into_boxed_slice(), avec![].into_boxed_slice())
            };

            if p < (1u32 << 31) {
                init_negacyclic_twiddles_shoup(
                    p,
                    n,
                    &mut twid,
                    &mut twid_shoup,
                    &mut inv_twid,
                    &mut inv_twid_shoup,
                );
            } else {
                init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);
            }

            Some(Self {
                twid,
                twid_shoup,
                inv_twid_shoup,
                inv_twid,
                p,
                p_div,
            })
        }
    }

    #[must_use]
    pub fn fwd(&self, buf: &mut [u32]) -> FwdUpperBound {
        let p = self.p;

        if p < (1u32 << 30) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_30bit::fwd_avx512(simd, p, buf, &self.twid, &self.twid_shoup);
                    return FwdUpperBound::FourP;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_30bit::fwd_avx2(simd, p, buf, &self.twid, &self.twid_shoup);
                    return FwdUpperBound::FourP;
                }
            }
            less_than_30bit::fwd_scalar(p, buf, &self.twid, &self.twid_shoup);
            FwdUpperBound::FourP
        } else if p < (1u32 << 31) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_31bit::fwd_avx512(simd, p, buf, &self.twid, &self.twid_shoup);
                    return FwdUpperBound::TwoP;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_31bit::fwd_avx2(simd, p, buf, &self.twid, &self.twid_shoup);
                    return FwdUpperBound::TwoP;
                }
            }
            less_than_31bit::fwd_scalar(p, buf, &self.twid, &self.twid_shoup);
            FwdUpperBound::TwoP
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                generic::fwd_avx512(simd, buf, p, self.p_div, &self.twid);
                return FwdUpperBound::P;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(simd) = crate::Avx2::try_new() {
                generic::fwd_avx2(simd, buf, p, self.p_div, &self.twid);
                return FwdUpperBound::P;
            }
            generic::fwd_scalar(buf, p, self.p_div, &self.twid);
            FwdUpperBound::P
        }
    }

    #[must_use]
    pub fn inv(&self, buf: &mut [u32]) -> InvUpperBound {
        let p = self.p;

        if p < (1u32 << 30) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_30bit::inv_avx512(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return InvUpperBound::TwoP;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_30bit::inv_avx2(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return InvUpperBound::TwoP;
                }
            }
            less_than_30bit::inv_scalar(p, buf, &self.inv_twid, &self.inv_twid_shoup);
            InvUpperBound::TwoP
        } else if p < (1u32 << 31) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_31bit::inv_avx512(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return InvUpperBound::P;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_31bit::inv_avx2(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return InvUpperBound::P;
                }
            }
            less_than_31bit::inv_scalar(p, buf, &self.inv_twid, &self.inv_twid_shoup);
            InvUpperBound::P
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                generic::inv_avx512(simd, buf, p, self.p_div, &self.inv_twid);
                return InvUpperBound::P;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(simd) = crate::Avx2::try_new() {
                generic::inv_avx2(simd, buf, p, self.p_div, &self.inv_twid);
                return InvUpperBound::P;
            }
            generic::inv_scalar(buf, p, self.p_div, &self.inv_twid);
            InvUpperBound::P
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prime::largest_prime_in_arithmetic_progression64;
    use alloc::vec;
    use rand::random;

    #[inline(always)]
    fn add(p: u32, a: u32, b: u32) -> u32 {
        let neg_b = p - b;
        if a >= neg_b {
            a - neg_b
        } else {
            a + b
        }
    }

    #[inline(always)]
    fn sub(p: u32, a: u32, b: u32) -> u32 {
        let neg_b = p - b;
        if a >= b {
            a - b
        } else {
            a + neg_b
        }
    }

    #[inline(always)]
    fn mul(p: Div32, a: u32, b: u32) -> u32 {
        Div32::rem_u64(a as u64 * b as u64, p)
    }

    extern crate alloc;

    #[test]
    fn test_product() {
        for n in [32, 64, 128, 256, 512, 1024] {
            for p in [
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 29, 1 << 30).unwrap(),
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 30, 1 << 31).unwrap(),
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 31, 1 << 32).unwrap(),
            ] {
                let p = p as u32;
                let plan = Plan::try_new(n, p).unwrap();

                let mut lhs = vec![0u32; n];
                let mut rhs = vec![0u32; n];

                for x in &mut lhs {
                    *x = random();
                    *x %= p;
                }
                for x in &mut rhs {
                    *x = random();
                    *x %= p;
                }

                let lhs = lhs;
                let rhs = rhs;

                let mut full_convolution = vec![0u32; 2 * n];
                let mut negacyclic_convolution = vec![0u32; n];
                for i in 0..n {
                    for j in 0..n {
                        full_convolution[i + j] = add(
                            p,
                            full_convolution[i + j],
                            mul(Div32::new(p), lhs[i], rhs[j]),
                        );
                    }
                }
                for i in 0..n {
                    negacyclic_convolution[i] =
                        sub(p, full_convolution[i], full_convolution[i + n]);
                }

                let mut prod = vec![0u32; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                let bound = match plan.fwd(&mut lhs_fourier) {
                    FwdUpperBound::P => p,
                    FwdUpperBound::TwoP => 2 * p,
                    FwdUpperBound::FourP => 4 * p,
                };
                let _ = plan.fwd(&mut rhs_fourier);

                for x in &lhs_fourier {
                    assert!(*x < bound);
                }
                for x in &rhs_fourier {
                    assert!(*x < bound);
                }

                for i in 0..n {
                    prod[i] = mul(Div32::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                let bound = match plan.inv(&mut prod) {
                    InvUpperBound::P => p,
                    InvUpperBound::TwoP => 2 * p,
                };

                for x in &prod {
                    assert!(*x < bound);
                }

                for i in 0..n {
                    assert_eq!(
                        prod[i] % p,
                        mul(Div32::new(p), negacyclic_convolution[i], n as u32),
                    );
                }
            }
        }
    }
}
