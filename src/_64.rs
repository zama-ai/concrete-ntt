use crate::{bit_rev, fastdiv::Div64, roots::find_primitive_root64};
use aligned_vec::{avec, ABox};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const RECURSION_THRESHOLD: usize = 1024;

mod generic_solinas;
mod shoup;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
mod less_than_50bit;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
mod less_than_51bit;
mod less_than_62bit;
mod less_than_63bit;

pub use generic_solinas::Solinas;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl crate::Avx2 {
    #[inline(always)]
    fn interleave2_epu64(self, z0z0z1z1: [__m256i; 2]) -> [__m256i; 2] {
        let avx = self.avx;
        [
            avx._mm256_permute2f128_si256::<0b0010_0000>(z0z0z1z1[0], z0z0z1z1[1]),
            avx._mm256_permute2f128_si256::<0b0011_0001>(z0z0z1z1[0], z0z0z1z1[1]),
        ]
    }

    #[inline(always)]
    fn permute2_epu64(self, w: [u64; 2]) -> __m256i {
        let avx = self.avx;
        let w00 = self.sse2._mm_set1_epi64x(w[0] as _);
        let w11 = self.sse2._mm_set1_epi64x(w[1] as _);
        avx._mm256_insertf128_si256::<0b1>(avx._mm256_castsi128_si256(w00), w11)
    }

    #[inline(always)]
    fn interleave1_epu64(self, z0z1: [__m256i; 2]) -> [__m256i; 2] {
        let avx = self.avx2;
        [
            avx._mm256_unpacklo_epi64(z0z1[0], z0z1[1]),
            avx._mm256_unpackhi_epi64(z0z1[0], z0z1[1]),
        ]
    }

    #[inline(always)]
    fn permute1_epu64(self, w: [u64; 4]) -> __m256i {
        let avx = self.avx;
        let w0123 = pulp::cast(w);
        let w0101 = avx._mm256_permute2f128_si256::<0b00000000>(w0123, w0123);
        let w2323 = avx._mm256_permute2f128_si256::<0b00110011>(w0123, w0123);
        avx._mm256_castpd_si256(avx._mm256_shuffle_pd::<0b1100>(
            avx._mm256_castsi256_pd(w0101),
            avx._mm256_castsi256_pd(w2323),
        ))
    }

    #[inline(always)]
    pub fn small_mod_epu64(self, modulus: __m256i, x: __m256i) -> __m256i {
        let avx = self.avx2;
        avx._mm256_blendv_epi8(
            avx._mm256_sub_epi64(x, modulus),
            x,
            self._mm256_cmpgt_epu64(modulus, x),
        )
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl crate::Avx512 {
    #[inline(always)]
    fn interleave4_epu64(self, z0z0z0z0z1z1z1z1: [__m512i; 2]) -> [__m512i; 2] {
        let avx = self.avx512f;
        let idx_0 = avx._mm512_setr_epi64(0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xa, 0xb);
        let idx_1 = avx._mm512_setr_epi64(0x4, 0x5, 0x6, 0x7, 0xc, 0xd, 0xe, 0xf);
        [
            avx._mm512_permutex2var_epi64(z0z0z0z0z1z1z1z1[0], idx_0, z0z0z0z0z1z1z1z1[1]),
            avx._mm512_permutex2var_epi64(z0z0z0z0z1z1z1z1[0], idx_1, z0z0z0z0z1z1z1z1[1]),
        ]
    }

    #[inline(always)]
    fn permute4_epu64(self, w: [u64; 2]) -> __m512i {
        let avx = self.avx512f;
        let w = pulp::cast(w);
        let w01xxxxxx = avx._mm512_castsi128_si512(w);
        let idx = avx._mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
        avx._mm512_permutexvar_epi64(idx, w01xxxxxx)
    }

    #[inline(always)]
    fn interleave2_epu64(self, z0z0z1z1: [__m512i; 2]) -> [__m512i; 2] {
        let avx = self.avx512f;
        let idx_0 = avx._mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xc, 0xd);
        let idx_1 = avx._mm512_setr_epi64(0x2, 0x3, 0xa, 0xb, 0x6, 0x7, 0xe, 0xf);
        [
            avx._mm512_permutex2var_epi64(z0z0z1z1[0], idx_0, z0z0z1z1[1]),
            avx._mm512_permutex2var_epi64(z0z0z1z1[0], idx_1, z0z0z1z1[1]),
        ]
    }

    #[inline(always)]
    fn permute2_epu64(self, w: [u64; 4]) -> __m512i {
        let avx = self.avx512f;
        let w = pulp::cast(w);
        let w0123xxxx = avx._mm512_castsi256_si512(w);
        let idx = avx._mm512_setr_epi64(0, 0, 2, 2, 1, 1, 3, 3);
        avx._mm512_permutexvar_epi64(idx, w0123xxxx)
    }

    #[inline(always)]
    fn interleave1_epu64(self, z0z1: [__m512i; 2]) -> [__m512i; 2] {
        let avx = self.avx512f;
        [
            avx._mm512_unpacklo_epi64(z0z1[0], z0z1[1]),
            avx._mm512_unpackhi_epi64(z0z1[0], z0z1[1]),
        ]
    }

    #[inline(always)]
    fn permute1_epu64(self, w: [u64; 8]) -> __m512i {
        let avx = self.avx512f;
        let w = pulp::cast(w);
        let idx = avx._mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
        avx._mm512_permutexvar_epi64(idx, w)
    }

    #[inline(always)]
    pub fn small_mod_epu64(self, modulus: __m512i, x: __m512i) -> __m512i {
        let avx = self.avx512f;
        avx._mm512_mask_blend_epi64(
            avx._mm512_cmpge_epu64_mask(x, modulus),
            x,
            avx._mm512_sub_epi64(x, modulus),
        )
    }
}

fn init_negacyclic_twiddles(p: u64, n: usize, twid: &mut [u64], inv_twid: &mut [u64]) {
    let div = Div64::new(p);
    let w = find_primitive_root64(div, 2 * n as u64).unwrap();
    let mut k = 0;
    let mut wk = 1u64;

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

        wk = Div64::rem_u128(wk as u128 * w as u128, div);
        k += 1;
    }
}

fn init_negacyclic_twiddles_shoup(
    p: u64,
    n: usize,
    max_bits: u32,
    twid: &mut [u64],
    twid_shoup: &mut [u64],
    inv_twid: &mut [u64],
    inv_twid_shoup: &mut [u64],
) {
    let div = Div64::new(p);
    let w = find_primitive_root64(div, 2 * n as u64).unwrap();
    let mut k = 0;
    let mut wk = 1u64;

    let nbits = n.trailing_zeros();
    while k < n {
        let fwd_idx = bit_rev(nbits, k);

        let wk_shoup = Div64::div_u128((wk as u128) << max_bits, div) as u64;
        twid[fwd_idx] = wk;
        twid_shoup[fwd_idx] = wk_shoup;

        let inv_idx = bit_rev(nbits, (n - k) % n);
        if k == 0 {
            inv_twid[inv_idx] = wk;
            inv_twid_shoup[inv_idx] = wk_shoup;
        } else {
            let x = p.wrapping_sub(wk);
            inv_twid[inv_idx] = x;
            inv_twid_shoup[inv_idx] = Div64::div_u128((x as u128) << max_bits, div) as u64;
        }

        wk = Div64::rem_u128(wk as u128 * w as u128, div);
        k += 1;
    }
}

#[derive(Debug, Clone)]
pub struct Plan {
    twid: ABox<[u64]>,
    twid_shoup: ABox<[u64]>,
    inv_twid: ABox<[u64]>,
    inv_twid_shoup: ABox<[u64]>,
    p: u64,
    p_div: Div64,
}

impl Plan {
    pub fn try_new(n: usize, p: u64) -> Option<Self> {
        let p_div = Div64::new(p);
        if find_primitive_root64(p_div, 2 * n as u64).is_none() {
            None
        } else {
            let mut twid = avec![0u64; n].into_boxed_slice();
            let mut inv_twid = avec![0u64; n].into_boxed_slice();
            let (mut twid_shoup, mut inv_twid_shoup) = if p < (1u64 << 63) {
                (
                    avec![0u64; n].into_boxed_slice(),
                    avec![0u64; n].into_boxed_slice(),
                )
            } else {
                (avec![].into_boxed_slice(), avec![].into_boxed_slice())
            };

            if p < (1u64 << 63) {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                #[cfg(feature = "nightly")]
                let max_bits = if p < (1u64 << 51) { 52 } else { 64 };
                #[cfg(not(all(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    feature = "nightly",
                )))]
                let max_bits = 64;
                init_negacyclic_twiddles_shoup(
                    p,
                    n,
                    max_bits,
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

    #[inline]
    pub fn ntt_size(&self) -> usize {
        self.twid.len()
    }

    pub fn fwd(&self, buf: &mut [u64]) {
        assert_eq!(buf.len(), self.ntt_size());
        let p = self.p;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "nightly")]
        if p < (1u64 << 50) {
            if let Some(simd) = crate::Avx512::try_new() {
                less_than_50bit::fwd_avx512(simd, p, buf, &self.twid, &self.twid_shoup);
                return;
            }
        } else if p < (1u64 << 51) {
            if let Some(simd) = crate::Avx512::try_new() {
                less_than_51bit::fwd_avx512(simd, p, buf, &self.twid, &self.twid_shoup);
                return;
            }
        }

        if p < (1u64 << 62) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_62bit::fwd_avx512(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_62bit::fwd_avx2(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
            }
            less_than_62bit::fwd_scalar(p, buf, &self.twid, &self.twid_shoup);
        } else if p < (1u64 << 63) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_63bit::fwd_avx512(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_63bit::fwd_avx2(simd, p, buf, &self.twid, &self.twid_shoup);
                    return;
                }
            }
            less_than_63bit::fwd_scalar(p, buf, &self.twid, &self.twid_shoup);
        } else if p == Solinas::P {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    generic_solinas::fwd_avx512(simd, buf, Solinas, (), &self.twid);
                    return;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    generic_solinas::fwd_avx2(simd, buf, Solinas, (), &self.twid);
                    return;
                }
            }
            generic_solinas::fwd_scalar(buf, Solinas, (), &self.twid);
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                let crate::u256 { x0, x1, x2, x3 } = self.p_div.double_reciprocal;
                let p_div = (p, x0, x1, x2, x3);
                generic_solinas::fwd_avx512(simd, buf, p, p_div, &self.twid);
                return;
            }
            generic_solinas::fwd_scalar(buf, p, self.p_div, &self.twid);
        }
    }

    pub fn inv(&self, buf: &mut [u64]) {
        assert_eq!(buf.len(), self.ntt_size());
        let p = self.p;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "nightly")]
        if p < (1u64 << 50) {
            if let Some(simd) = crate::Avx512::try_new() {
                less_than_50bit::inv_avx512(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                return;
            }
        } else if p < (1u64 << 51) {
            if let Some(simd) = crate::Avx512::try_new() {
                less_than_51bit::inv_avx512(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                return;
            }
        }

        if p < (1u64 << 62) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_62bit::inv_avx512(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_62bit::inv_avx2(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
            }
            less_than_62bit::inv_scalar(p, buf, &self.inv_twid, &self.inv_twid_shoup);
        } else if p < (1u64 << 63) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    less_than_63bit::inv_avx512(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    less_than_63bit::inv_avx2(simd, p, buf, &self.inv_twid, &self.inv_twid_shoup);
                    return;
                }
            }
            less_than_63bit::inv_scalar(p, buf, &self.inv_twid, &self.inv_twid_shoup);
        } else if p == Solinas::P {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = crate::Avx512::try_new() {
                    generic_solinas::inv_avx512(simd, buf, Solinas, (), &self.inv_twid);
                    return;
                }
                if let Some(simd) = crate::Avx2::try_new() {
                    generic_solinas::inv_avx2(simd, buf, Solinas, (), &self.inv_twid);
                    return;
                }
            }
            generic_solinas::inv_scalar(buf, Solinas, (), &self.inv_twid);
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                let crate::u256 { x0, x1, x2, x3 } = self.p_div.double_reciprocal;
                let p_div = (p, x0, x1, x2, x3);
                generic_solinas::inv_avx512(simd, buf, p, p_div, &self.inv_twid);
                return;
            }
            generic_solinas::inv_scalar(buf, p, self.p_div, &self.inv_twid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fastdiv::Div64, prime::largest_prime_in_arithmetic_progression64,
        _64::generic_solinas::PrimeModulus,
    };
    use alloc::vec;
    use rand::random;

    extern crate alloc;

    #[test]
    fn test_product() {
        for n in [16, 32, 64, 128, 256, 512, 1024] {
            for p in [
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 49, 1 << 50).unwrap(),
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 50, 1 << 51).unwrap(),
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 61, 1 << 62).unwrap(),
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63).unwrap(),
                Solinas::P,
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 63, u64::MAX).unwrap(),
            ] {
                dbg!(p);
                let plan = Plan::try_new(n, p).unwrap();

                let mut lhs = vec![0u64; n];
                let mut rhs = vec![0u64; n];

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

                let mut full_convolution = vec![0u64; 2 * n];
                let mut negacyclic_convolution = vec![0u64; n];
                for i in 0..n {
                    for j in 0..n {
                        full_convolution[i + j] = PrimeModulus::add(
                            p,
                            full_convolution[i + j],
                            <u64 as PrimeModulus>::mul(Div64::new(p), lhs[i], rhs[j]),
                        );
                    }
                }
                for i in 0..n {
                    negacyclic_convolution[i] =
                        PrimeModulus::sub(p, full_convolution[i], full_convolution[i + n]);
                }

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                plan.fwd(&mut lhs_fourier);
                plan.fwd(&mut rhs_fourier);

                for x in &lhs_fourier {
                    assert!(*x < p);
                }
                for x in &rhs_fourier {
                    assert!(*x < p);
                }

                for i in 0..n {
                    prod[i] =
                        <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                plan.inv(&mut prod);

                for x in &prod {
                    assert!(*x < p);
                }

                for i in 0..n {
                    assert_eq!(
                        prod[i],
                        <u64 as PrimeModulus>::mul(
                            Div64::new(p),
                            negacyclic_convolution[i],
                            n as u64
                        ),
                    );
                }
            }
        }
    }
}
