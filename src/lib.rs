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

pub mod native128;
pub mod native32;
pub mod native64;

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
    pub fn _mm256_mullo_u64_u32_epu64(self, x: __m256i, y: __m256i) -> __m256i {
        let avx2 = self.avx2;
        let x_hi = avx2._mm256_shuffle_epi32::<0b10110001>(x);
        let z_lo_lo = avx2._mm256_mul_epu32(x, y);
        let z_hi_lo = avx2._mm256_mul_epu32(x_hi, y);
        avx2._mm256_add_epi64(avx2._mm256_slli_epi64::<32>(z_hi_lo), z_lo_lo)
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

pub mod primes32 {
    use crate::{
        fastdiv::{Div32, Div64},
        prime::exp_mod32,
    };

    pub const P0: u32 = 0b00111111010110100000000000000001;
    pub const P1: u32 = 0b00111111010111010000000000000001;
    pub const P2: u32 = 0b00111111011101100000000000000001;
    pub const P3: u32 = 0b00111111100000100000000000000001;
    pub const P4: u32 = 0b00111111101011000000000000000001;
    pub const P5: u32 = 0b00111111101011110000000000000001;
    pub const P6: u32 = 0b00111111101100010000000000000001;
    pub const P7: u32 = 0b00111111101110110000000000000001;
    pub const P8: u32 = 0b00111111110111100000000000000001;
    pub const P9: u32 = 0b00111111111111000000000000000001;

    pub const P0_MAGIC: u64 = 9317778228489988551;
    pub const P1_MAGIC: u64 = 4658027473943558643;
    pub const P2_MAGIC: u64 = 1162714878353869247;
    pub const P3_MAGIC: u64 = 4647426722536610861;
    pub const P4_MAGIC: u64 = 9270903515973367219;
    pub const P5_MAGIC: u64 = 2317299382174935855;
    pub const P6_MAGIC: u64 = 9268060552616330319;
    pub const P7_MAGIC: u64 = 2315594963384859737;
    pub const P8_MAGIC: u64 = 9242552129100825291;
    pub const P9_MAGIC: u64 = 576601523622774689;

    pub const P0_MAGIC_SHIFT: u32 = 29;
    pub const P1_MAGIC_SHIFT: u32 = 28;
    pub const P2_MAGIC_SHIFT: u32 = 26;
    pub const P3_MAGIC_SHIFT: u32 = 28;
    pub const P4_MAGIC_SHIFT: u32 = 29;
    pub const P5_MAGIC_SHIFT: u32 = 27;
    pub const P6_MAGIC_SHIFT: u32 = 29;
    pub const P7_MAGIC_SHIFT: u32 = 27;
    pub const P8_MAGIC_SHIFT: u32 = 29;
    pub const P9_MAGIC_SHIFT: u32 = 25;

    const fn mul_mod(modulus: u32, a: u32, b: u32) -> u32 {
        let wide = a as u64 * b as u64;
        (wide % modulus as u64) as u32
    }

    const fn inv_mod(modulus: u32, x: u32) -> u32 {
        exp_mod32(Div32::new(modulus), x, modulus - 2)
    }

    const fn shoup(modulus: u32, w: u32) -> u32 {
        (((w as u64) << 32) / modulus as u64) as u32
    }

    const fn mul_mod64(modulus: u64, a: u64, b: u64) -> u64 {
        let wide = a as u128 * b as u128;
        (wide % modulus as u128) as u64
    }

    const fn exp_mod64(modulus: u64, base: u64, pow: u64) -> u64 {
        crate::prime::exp_mod64(Div64::new(modulus), base, pow)
    }

    const fn shoup64(modulus: u64, w: u64) -> u64 {
        (((w as u128) << 64) / modulus as u128) as u64
    }

    pub const P0_INV_MOD_P1: u32 = inv_mod(P1, P0);
    pub const P0_INV_MOD_P1_SHOUP: u32 = shoup(P1, P0_INV_MOD_P1);
    pub const P01_INV_MOD_P2: u32 = inv_mod(P2, mul_mod(P2, P0, P1));
    pub const P01_INV_MOD_P2_SHOUP: u32 = shoup(P2, P01_INV_MOD_P2);
    pub const P012_INV_MOD_P3: u32 = inv_mod(P3, mul_mod(P3, mul_mod(P3, P0, P1), P2));
    pub const P012_INV_MOD_P3_SHOUP: u32 = shoup(P3, P012_INV_MOD_P3);
    pub const P0123_INV_MOD_P4: u32 =
        inv_mod(P4, mul_mod(P4, mul_mod(P4, mul_mod(P4, P0, P1), P2), P3));
    pub const P0123_INV_MOD_P4_SHOUP: u32 = shoup(P4, P0123_INV_MOD_P4);

    pub const P0_MOD_P2_SHOUP: u32 = shoup(P2, P0);
    pub const P0_MOD_P3_SHOUP: u32 = shoup(P3, P0);
    pub const P1_MOD_P3_SHOUP: u32 = shoup(P3, P1);
    pub const P0_MOD_P4_SHOUP: u32 = shoup(P4, P0);
    pub const P1_MOD_P4_SHOUP: u32 = shoup(P4, P1);
    pub const P2_MOD_P4_SHOUP: u32 = shoup(P4, P2);

    pub const P1_INV_MOD_P2: u32 = inv_mod(P2, P1);
    pub const P1_INV_MOD_P2_SHOUP: u32 = shoup(P2, P1_INV_MOD_P2);
    pub const P3_INV_MOD_P4: u32 = inv_mod(P4, P3);
    pub const P3_INV_MOD_P4_SHOUP: u32 = shoup(P4, P3_INV_MOD_P4);
    pub const P12: u64 = P1 as u64 * P2 as u64;
    pub const P34: u64 = P3 as u64 * P4 as u64;
    pub const P0_INV_MOD_P12: u64 =
        exp_mod64(P12, P0 as u64, (P1 as u64 - 1) * (P2 as u64 - 1) - 1);
    pub const P0_INV_MOD_P12_SHOUP: u64 = shoup64(P12, P0_INV_MOD_P12);
    pub const P0_MOD_P34_SHOUP: u64 = shoup64(P34, P0 as u64);
    pub const P012_INV_MOD_P34: u64 = exp_mod64(
        P34,
        mul_mod64(P34, P0 as u64, P12),
        (P3 as u64 - 1) * (P4 as u64 - 1) - 1,
    );
    pub const P012_INV_MOD_P34_SHOUP: u64 = shoup64(P34, P012_INV_MOD_P34);

    pub const P2_INV_MOD_P3: u32 = inv_mod(P3, P2);
    pub const P2_INV_MOD_P3_SHOUP: u32 = shoup(P3, P2_INV_MOD_P3);
    pub const P4_INV_MOD_P5: u32 = inv_mod(P5, P4);
    pub const P4_INV_MOD_P5_SHOUP: u32 = shoup(P5, P4_INV_MOD_P5);
    pub const P6_INV_MOD_P7: u32 = inv_mod(P7, P6);
    pub const P6_INV_MOD_P7_SHOUP: u32 = shoup(P7, P6_INV_MOD_P7);
    pub const P8_INV_MOD_P9: u32 = inv_mod(P9, P8);
    pub const P8_INV_MOD_P9_SHOUP: u32 = shoup(P9, P8_INV_MOD_P9);

    pub const P01: u64 = P0 as u64 * P1 as u64;
    pub const P23: u64 = P2 as u64 * P3 as u64;
    pub const P45: u64 = P4 as u64 * P5 as u64;
    pub const P67: u64 = P6 as u64 * P7 as u64;
    pub const P89: u64 = P8 as u64 * P9 as u64;

    pub const P01_MOD_P45_SHOUP: u64 = shoup64(P45, P01);
    pub const P01_MOD_P67_SHOUP: u64 = shoup64(P67, P01);
    pub const P01_MOD_P89_SHOUP: u64 = shoup64(P89, P01);

    pub const P23_MOD_P67_SHOUP: u64 = shoup64(P67, P23);
    pub const P23_MOD_P89_SHOUP: u64 = shoup64(P89, P23);

    pub const P45_MOD_P89_SHOUP: u64 = shoup64(P89, P45);

    pub const P01_INV_MOD_P23: u64 = exp_mod64(P23, P01, (P2 as u64 - 1) * (P3 as u64 - 1) - 1);
    pub const P01_INV_MOD_P23_SHOUP: u64 = shoup64(P23, P01_INV_MOD_P23);
    pub const P0123_INV_MOD_P45: u64 = exp_mod64(
        P45,
        mul_mod64(P45, P01, P23),
        (P4 as u64 - 1) * (P5 as u64 - 1) - 1,
    );
    pub const P0123_INV_MOD_P45_SHOUP: u64 = shoup64(P45, P0123_INV_MOD_P45);
    pub const P012345_INV_MOD_P67: u64 = exp_mod64(
        P67,
        mul_mod64(P67, mul_mod64(P67, P01, P23), P45),
        (P6 as u64 - 1) * (P7 as u64 - 1) - 1,
    );
    pub const P012345_INV_MOD_P67_SHOUP: u64 = shoup64(P67, P012345_INV_MOD_P67);
    pub const P01234567_INV_MOD_P89: u64 = exp_mod64(
        P89,
        mul_mod64(P89, mul_mod64(P89, mul_mod64(P89, P01, P23), P45), P67),
        (P8 as u64 - 1) * (P9 as u64 - 1) - 1,
    );
    pub const P01234567_INV_MOD_P89_SHOUP: u64 = shoup64(P89, P01234567_INV_MOD_P89);

    pub const P0123: u128 = u128::wrapping_mul(P01 as u128, P23 as u128);
    pub const P012345: u128 = u128::wrapping_mul(P0123, P45 as u128);
    pub const P01234567: u128 = u128::wrapping_mul(P012345, P67 as u128);
    pub const P0123456789: u128 = u128::wrapping_mul(P01234567, P89 as u128);
}

pub mod primes52 {
    use crate::fastdiv::Div64;

    pub const P0: u64 = 0b0011111111111111111111111110011101110000000000000001;
    pub const P1: u64 = 0b0011111111111111111111111110101110010000000000000001;
    pub const P2: u64 = 0b0011111111111111111111111110110010000000000000000001;
    pub const P3: u64 = 0b0011111111111111111111111111100010110000000000000001;
    pub const P4: u64 = 0b0011111111111111111111111111101110000000000000000001;
    pub const P5: u64 = 0b0011111111111111111111111111110001110000000000000001;

    pub const P0_MAGIC: u64 = 9223372247845040859;
    pub const P1_MAGIC: u64 = 4611686106205779591;
    pub const P2_MAGIC: u64 = 4611686102179247601;
    pub const P3_MAGIC: u64 = 2305843024917166187;
    pub const P4_MAGIC: u64 = 4611686037754736721;
    pub const P5_MAGIC: u64 = 4611686033728204851;

    pub const P0_MAGIC_SHIFT: u32 = 49;
    pub const P1_MAGIC_SHIFT: u32 = 48;
    pub const P2_MAGIC_SHIFT: u32 = 48;
    pub const P3_MAGIC_SHIFT: u32 = 47;
    pub const P4_MAGIC_SHIFT: u32 = 48;
    pub const P5_MAGIC_SHIFT: u32 = 48;

    const fn mul_mod(modulus: u64, a: u64, b: u64) -> u64 {
        let wide = a as u128 * b as u128;
        (wide % modulus as u128) as u64
    }

    const fn inv_mod(modulus: u64, x: u64) -> u64 {
        crate::prime::exp_mod64(Div64::new(modulus), x, modulus - 2)
    }

    const fn shoup(modulus: u64, w: u64) -> u64 {
        (((w as u128) << 52) / modulus as u128) as u64
    }

    pub const P0_INV_MOD_P1: u64 = inv_mod(P1, P0);
    pub const P0_INV_MOD_P1_SHOUP: u64 = shoup(P1, P0_INV_MOD_P1);

    pub const P01_INV_MOD_P2: u64 = inv_mod(P2, mul_mod(P2, P0, P1));
    pub const P01_INV_MOD_P2_SHOUP: u64 = shoup(P2, P01_INV_MOD_P2);
    pub const P012_INV_MOD_P3: u64 = inv_mod(P3, mul_mod(P3, mul_mod(P3, P0, P1), P2));
    pub const P012_INV_MOD_P3_SHOUP: u64 = shoup(P3, P012_INV_MOD_P3);
    pub const P0123_INV_MOD_P4: u64 =
        inv_mod(P4, mul_mod(P4, mul_mod(P4, mul_mod(P4, P0, P1), P2), P3));
    pub const P0123_INV_MOD_P4_SHOUP: u64 = shoup(P4, P0123_INV_MOD_P4);

    pub const P0_MOD_P2_SHOUP: u64 = shoup(P2, P0);
    pub const P0_MOD_P3_SHOUP: u64 = shoup(P3, P0);
    pub const P1_MOD_P3_SHOUP: u64 = shoup(P3, P1);
    pub const P0_MOD_P4_SHOUP: u64 = shoup(P4, P0);
    pub const P1_MOD_P4_SHOUP: u64 = shoup(P4, P1);
    pub const P2_MOD_P4_SHOUP: u64 = shoup(P4, P2);
}

macro_rules! izip {
    (@ __closure @ ($a:expr)) => { |a| (a,) };
    (@ __closure @ ($a:expr, $b:expr)) => { |(a, b)| (a, b) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr)) => { |((a, b), c)| (a, b, c) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr)) => { |(((a, b), c), d)| (a, b, c, d) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr)) => { |((((a, b), c), d), e)| (a, b, c, d, e) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr)) => { |(((((a, b), c), d), e), f)| (a, b, c, d, e, f) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr)) => { |((((((a, b), c), d), e), f), g)| (a, b, c, d, e, f, g) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr)) => { |(((((((a, b), c), d), e), f), g), h)| (a, b, c, d, e, f, g, h) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr)) => { |((((((((a, b), c), d), e), f), g), h), i)| (a, b, c, d, e, f, g, h, i) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr)) => { |(((((((((a, b), c), d), e), f), g), h), i), j)| (a, b, c, d, e, f, g, h, i, j) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr)) => { |((((((((((a, b), c), d), e), f), g), h), i), j), k)| (a, b, c, d, e, f, g, h, i, j, k) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr)) => { |(((((((((((a, b), c), d), e), f), g), h), i), j), k), l)| (a, b, c, d, e, f, g, h, i, j, k, l) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr)) => { |((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m)| (a, b, c, d, e, f, g, h, i, j, k, l, m) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr, $n:expr)) => { |(((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m), n)| (a, b, c, d, e, f, g, h, i, j, k, l, m, n) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr, $n:expr, $o:expr)) => { |((((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m), n), o)| (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) };

    ( $first:expr $(,)?) => {
        {
            ::core::iter::IntoIterator::into_iter($first)
        }
    };
    ( $first:expr, $($rest:expr),+ $(,)?) => {
        {
            ::core::iter::IntoIterator::into_iter($first)
                $(.zip($rest))*
                .map(crate::izip!(@ __closure @ ($first, $($rest),*)))
        }
    };
}
pub(crate) use izip;

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

    #[test]
    fn generate_primes() {
        let mut p = 1u64 << 30;
        for _ in 0..100 {
            p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 0, p - 1).unwrap();
            println!("{p:#034b}");
        }

        let mut p = 1u64 << 50;
        for _ in 0..100 {
            p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 0, p - 1).unwrap();
            println!("{p:#054b}");
        }
    }
}
