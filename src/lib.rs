#![cfg_attr(feature = "nightly", feature(avx512_target_feature, stdsimd))]
#![cfg_attr(not(feature = "std"), no_std)]

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

pub enum FwdUpperBound {
    P,
    TwoP,
    FourP,
}

pub enum InvUpperBound {
    P,
    TwoP,
}

#[inline]
fn bit_rev(nbits: u32, i: usize) -> usize {
    i.reverse_bits() >> (usize::BITS - nbits)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
simd_type! {
    pub struct Avx2 {
        pub sse: "sse",
        pub sse2: "sse2",
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
    pub fn _mm512_movm_epi64(self, k: __mmask8) -> __m512i {
        let avx = self.avx512f;
        let zeros = avx._mm512_setzero_si512();
        let ones = avx._mm512_set1_epi64(-1);
        avx._mm512_mask_blend_epi64(k, zeros, ones)
    }
}
