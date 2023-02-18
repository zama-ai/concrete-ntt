#![cfg_attr(feature = "nightly", feature(avx512_target_feature, stdsimd))]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::too_many_arguments, clippy::let_unit_value)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::fmt::Debug;
use pulp::simd_type;
pub use u256_impl::u256;

pub mod fastdiv;
pub mod prime;
mod roots;
mod u256_impl;

pub mod _128;
pub mod _32;
pub mod _64;

#[inline]
fn bit_rev(nbits: u32, i: usize) -> usize {
    i.reverse_bits() >> (usize::BITS - nbits)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
simd_type! {
    pub struct Avx2 {
        pub sse: "sse",
        pub sse2: "sse2",
        pub sse3: "sse3",
        pub avx: "avx",
        pub avx2: "avx2",
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
simd_type! {
    pub struct Avx512 {
        pub sse: "sse",
        pub sse2: "sse2",
        pub avx: "avx",
        pub avx2: "avx2",
        pub avx512f: "avx512f",
        pub avx512ifma: "avx512ifma",
        pub avx512dq: "avx512dq",
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Avx2 {
    #[inline(always)]
    pub fn _mm256_mul_u64_u64_epu64(self, x: __m256i, y: __m256i) -> (__m256i, __m256i) {
        let avx = self.avx;
        let avx2 = self.avx2;
        let lo_mask = avx._mm256_set1_epi64x(0x00000000FFFFFFFFu64 as _);
        let x_hi = avx2._mm256_shuffle_epi32::<0b10110001>(x);
        let y_hi = avx2._mm256_shuffle_epi32::<0b10110001>(y);

        let z_lo_lo = avx2._mm256_mul_epu32(x, y);
        let z_lo_hi = avx2._mm256_mul_epu32(x, y_hi);
        let z_hi_lo = avx2._mm256_mul_epu32(x_hi, y);
        let z_hi_hi = avx2._mm256_mul_epu32(x_hi, y_hi);

        let z_lo_lo_shift = avx2._mm256_srli_epi64::<32>(z_lo_lo);

        let sum_tmp = avx2._mm256_add_epi64(z_lo_hi, z_lo_lo_shift);
        let sum_lo = avx2._mm256_and_si256(sum_tmp, lo_mask);
        let sum_mid = avx2._mm256_srli_epi64::<32>(sum_tmp);

        let sum_mid2 = avx2._mm256_add_epi64(z_hi_lo, sum_lo);
        let sum_mid2_hi = avx2._mm256_srli_epi64::<32>(sum_mid2);
        let sum_hi = avx2._mm256_add_epi64(z_hi_hi, sum_mid);

        let prod_hi = avx2._mm256_add_epi64(sum_hi, sum_mid2_hi);
        let prod_lo = avx2._mm256_add_epi64(
            avx2._mm256_slli_epi64::<32>(avx2._mm256_add_epi64(z_lo_hi, z_hi_lo)),
            z_lo_lo,
        );

        (prod_lo, prod_hi)
    }

    #[inline(always)]
    pub fn _mm256_cmpgt_epu64(self, x: __m256i, y: __m256i) -> __m256i {
        let c = self.avx._mm256_set1_epi64x(0x8000000000000000u64 as _);
        self.avx2._mm256_cmpgt_epi64(
            self.avx2._mm256_xor_si256(x, c),
            self.avx2._mm256_xor_si256(y, c),
        )
    }

    #[inline(always)]
    pub fn _mm256_mul_u32_u32_epu32(self, a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        let avx2 = self.avx2;

        // a0b0_lo a0b0_hi a2b2_lo a2b2_hi
        let ab_evens = avx2._mm256_mul_epu32(a, b);
        // a1b1_lo a1b1_hi a3b3_lo a3b3_hi
        let ab_odds = avx2._mm256_mul_epu32(
            avx2._mm256_srli_epi64::<32>(a),
            avx2._mm256_srli_epi64::<32>(b),
        );

        let ab_lo = avx2._mm256_blend_epi32::<0b10101010>(
            // a0b0_lo xxxxxxx a2b2_lo xxxxxxx
            ab_evens,
            // xxxxxxx a1b1_lo xxxxxxx a3b3_lo
            avx2._mm256_slli_epi64::<32>(ab_odds),
        );
        let ab_hi = avx2._mm256_blend_epi32::<0b10101010>(
            // a0b0_hi xxxxxxx a2b2_hi xxxxxxx
            avx2._mm256_srli_epi64::<32>(ab_evens),
            // xxxxxxx a1b1_hi xxxxxxx a3b3_hi
            ab_odds,
        );

        (ab_lo, ab_hi)
    }

    #[inline(always)]
    pub fn _mm256_cmpgt_epu32(self, x: __m256i, y: __m256i) -> __m256i {
        let c = self.avx._mm256_set1_epi32(0x80000000u32 as _);
        self.avx2._mm256_cmpgt_epi32(
            self.avx2._mm256_xor_si256(x, c),
            self.avx2._mm256_xor_si256(y, c),
        )
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl Avx512 {
    #[inline(always)]
    pub fn _mm512_mul_u64_u64_epu64(self, x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        let avx = self.avx512f;
        let lo_mask = avx._mm512_set1_epi64(0x00000000FFFFFFFFu64 as _);
        let x_hi = avx._mm512_shuffle_epi32::<0b10110001>(x);
        let y_hi = avx._mm512_shuffle_epi32::<0b10110001>(y);

        let z_lo_lo = avx._mm512_mul_epu32(x, y);
        let z_lo_hi = avx._mm512_mul_epu32(x, y_hi);
        let z_hi_lo = avx._mm512_mul_epu32(x_hi, y);
        let z_hi_hi = avx._mm512_mul_epu32(x_hi, y_hi);

        let z_lo_lo_shift = avx._mm512_srli_epi64::<32>(z_lo_lo);

        let sum_tmp = avx._mm512_add_epi64(z_lo_hi, z_lo_lo_shift);
        let sum_lo = avx._mm512_and_si512(sum_tmp, lo_mask);
        let sum_mid = avx._mm512_srli_epi64::<32>(sum_tmp);

        let sum_mid2 = avx._mm512_add_epi64(z_hi_lo, sum_lo);
        let sum_mid2_hi = avx._mm512_srli_epi64::<32>(sum_mid2);
        let sum_hi = avx._mm512_add_epi64(z_hi_hi, sum_mid);

        let prod_hi = avx._mm512_add_epi64(sum_hi, sum_mid2_hi);
        let prod_lo = avx._mm512_add_epi64(
            avx._mm512_slli_epi64::<32>(avx._mm512_add_epi64(z_lo_hi, z_hi_lo)),
            z_lo_lo,
        );

        (prod_lo, prod_hi)
    }

    #[inline(always)]
    pub fn _mm512_mul_u32_u32_epu32(self, a: __m512i, b: __m512i) -> (__m512i, __m512i) {
        let avx2 = self.avx512f;

        // a0b0_lo a0b0_hi a2b2_lo a2b2_hi
        let ab_evens = avx2._mm512_mul_epu32(a, b);
        // a1b1_lo a1b1_hi a3b3_lo a3b3_hi
        let ab_odds = avx2._mm512_mul_epu32(
            avx2._mm512_srli_epi64::<32>(a),
            avx2._mm512_srli_epi64::<32>(b),
        );

        let ab_lo = avx2._mm512_mask_blend_epi32(
            0b1010101010101010,
            // a0b0_lo xxxxxxx a2b2_lo xxxxxxx
            ab_evens,
            // xxxxxxx a1b1_lo xxxxxxx a3b3_lo
            avx2._mm512_slli_epi64::<32>(ab_odds),
        );
        let ab_hi = avx2._mm512_mask_blend_epi32(
            0b1010101010101010,
            // a0b0_hi xxxxxxx a2b2_hi xxxxxxx
            avx2._mm512_srli_epi64::<32>(ab_evens),
            // xxxxxxx a1b1_hi xxxxxxx a3b3_hi
            ab_odds,
        );

        (ab_lo, ab_hi)
    }

    #[inline(always)]
    pub fn _mm512_movm_epi64(self, k: __mmask8) -> __m512i {
        let avx = self.avx512f;
        let zeros = avx._mm512_setzero_si512();
        let ones = avx._mm512_set1_epi64(-1);
        avx._mm512_mask_blend_epi64(k, zeros, ones)
    }

    #[inline(always)]
    pub fn _mm512_movm_epi32(self, k: __mmask16) -> __m512i {
        let avx = self.avx512f;
        let zeros = avx._mm512_setzero_si512();
        let ones = avx._mm512_set1_epi32(-1);
        avx._mm512_mask_blend_epi32(k, zeros, ones)
    }
}

#[cfg(test)]
mod tests {
    use crate::prime::largest_prime_in_arithmetic_progression64;
    use rand::random;

    #[test]
    fn test_barrett() {
        let q = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63).unwrap();

        let big_q: u32 = q.ilog2() + 1;
        let big_l: u32 = big_q + 63;
        let k: u64 = ((1u128 << big_l) / q as u128).try_into().unwrap();

        for _ in 0..10000 {
            let a = random::<u64>() % q;
            let b = random::<u64>() % q;

            let d = a as u128 * b as u128;
            // Q < 63
            // d < 2^(2Q)
            // (d >> (Q-1)) < 2^(Q+1)         -> c1 fits in u64
            let c1 = (d >> (big_q - 1)) as u64;
            // c2 < 2^(Q+65)
            let c3 = ((c1 as u128 * k as u128) >> 64) as u64;
            let c = (d as u64).wrapping_sub(q.wrapping_mul(c3));
            let c = if c >= q { c - q } else { c };
            assert_eq!(c as u128, d % q as u128);
        }
    }
}
