#![cfg_attr(feature = "nightly", feature(avx512_target_feature, stdsimd))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use aligned_vec::{avec, ABox};
use core::fmt::Debug;
use core::iter::zip;
use fastdiv::Div64;
use pulp::{as_arrays, as_arrays_mut, cast, simd_type};
use roots::find_primitive_root64;
pub use u256_impl::u256;

pub mod fastdiv;
pub mod prime;
mod roots;
mod u256_impl;

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
impl Avx2 {
    #[inline(always)]
    pub fn _mm256_mul_epu64(self, x: __m256i, y: __m256i) -> (__m256i, __m256i) {
        let Self {
            avx,
            avx2,
            sse: _,
            sse2: _,
        } = self;
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
simd_type! {
    pub struct Avx512 {
        pub sse: "sse",
        pub sse2: "sse2",
        pub avx: "avx",
        pub avx2: "avx2",
        pub avx512f: "avx512f",
        pub avx512dq: "avx512dq",
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl Avx512 {
    #[inline(always)]
    pub fn _mm512_mul_epu64(self, x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        let Self {
            avx: _,
            avx2: _,
            sse: _,
            sse2: _,
            avx512f: avx,
            avx512dq: _,
        } = self;
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

pub trait PrimeModulus: Debug + Copy {
    type Div: Debug + Copy;

    fn add(self, a: u64, b: u64) -> u64;
    fn sub(self, a: u64, b: u64) -> u64;
    fn mul(p: Self::Div, a: u64, b: u64) -> u64;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub trait PrimeModulusAvx2: Debug + Copy {
    type Div: Debug + Copy;

    fn add(self, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4];
    fn sub(self, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4];
    fn mul(p: Self::Div, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4];
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub trait PrimeModulusAvx512: Debug + Copy {
    type Div: Debug + Copy;

    fn add(self, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8];
    fn sub(self, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8];
    fn mul(p: Self::Div, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8];
}

#[derive(Copy, Clone, Debug)]
pub struct Solinas;

impl Solinas {
    pub const P: u64 = ((1u128 << 64) - (1u128 << 32) + 1u128) as u64;
}

impl PrimeModulus for u64 {
    type Div = Div64;

    #[inline(always)]
    fn add(self, a: u64, b: u64) -> u64 {
        let p = self;
        // a + b >= p
        // implies
        // a >= p - b

        let neg_b = p - b;
        if a >= neg_b {
            a - neg_b
        } else {
            a + b
        }
    }

    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        let p = self;
        let neg_b = p - b;
        if a >= b {
            a - b
        } else {
            a + neg_b
        }
    }

    #[inline(always)]
    fn mul(p: Self::Div, a: u64, b: u64) -> u64 {
        Div64::rem_u128(a as u128 * b as u128, p)
    }
}

impl PrimeModulus for Solinas {
    type Div = ();

    #[inline(always)]
    fn add(self, a: u64, b: u64) -> u64 {
        let p = Self::P;
        let neg_b = p - b;
        if a >= neg_b {
            a - neg_b
        } else {
            a + b
        }
    }

    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        let p = Self::P;
        let neg_b = p - b;
        if a >= b {
            a - b
        } else {
            a + neg_b
        }
    }

    #[inline(always)]
    fn mul(p: Self::Div, a: u64, b: u64) -> u64 {
        let _ = p;
        let p = Self::P;

        let wide = a as u128 * b as u128;

        // https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/
        let lo = wide as u64;
        let __hi = (wide >> 64) as u64;
        let mid = __hi & 0x00000000FFFFFFFF;
        let hi = (__hi & 0xFFFFFFFF00000000) >> 32;

        let mut low2 = lo.wrapping_sub(hi);
        if hi > lo {
            low2 = low2.wrapping_add(p);
        }

        let mut product = mid << 32;
        product -= mid;

        let mut result = low2.wrapping_add(product);
        if (result < product) || (result >= p) {
            result = result.wrapping_sub(p);
        }
        result
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl PrimeModulusAvx2 for u64 {
    type Div = (u64, u64, u64, u64, u64);

    #[inline(always)]
    fn add(self, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let a = cast(a);
        let b = cast(b);
        let p = simd.avx._mm256_set1_epi64x(self as _);
        let neg_b = simd.avx2._mm256_sub_epi64(p, b);
        let not_a_ge_neg_b = simd._mm256_cmpgt_epu64(neg_b, a);
        cast(simd.avx2._mm256_blendv_epi8(
            simd.avx2._mm256_sub_epi64(a, neg_b),
            simd.avx2._mm256_add_epi64(a, b),
            not_a_ge_neg_b,
        ))
    }

    #[inline(always)]
    fn sub(self, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let a = cast(a);
        let b = cast(b);
        let p = simd.avx._mm256_set1_epi64x(self as _);
        let neg_b = simd.avx2._mm256_sub_epi64(p, b);
        let not_a_ge_b = simd._mm256_cmpgt_epu64(b, a);
        cast(simd.avx2._mm256_blendv_epi8(
            simd.avx2._mm256_sub_epi64(a, b),
            simd.avx2._mm256_add_epi64(a, neg_b),
            not_a_ge_b,
        ))
    }

    #[inline(always)]
    fn mul(p: Self::Div, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        #[inline(always)]
        fn mul_with_carry(simd: Avx2, l: __m256i, r: __m256i, c: __m256i) -> (__m256i, __m256i) {
            let (lo, hi) = simd._mm256_mul_epu64(l, r);
            let lo_plus_c = simd.avx2._mm256_add_epi64(lo, c);
            let overflow = simd._mm256_cmpgt_epu64(lo, lo_plus_c);
            (lo_plus_c, simd.avx2._mm256_sub_epi64(hi, overflow))
        }

        #[inline(always)]
        fn mul_u256_u64(
            simd: Avx2,
            lhs0: __m256i,
            lhs1: __m256i,
            lhs2: __m256i,
            lhs3: __m256i,
            rhs: __m256i,
        ) -> (__m256i, __m256i, __m256i, __m256i, __m256i) {
            let (x0, carry) = simd._mm256_mul_epu64(lhs0, rhs);
            let (x1, carry) = mul_with_carry(simd, lhs1, rhs, carry);
            let (x2, carry) = mul_with_carry(simd, lhs2, rhs, carry);
            let (x3, carry) = mul_with_carry(simd, lhs3, rhs, carry);
            (x0, x1, x2, x3, carry)
        }

        #[inline(always)]
        fn wrapping_mul_u256_u128(
            simd: Avx2,
            lhs0: __m256i,
            lhs1: __m256i,
            lhs2: __m256i,
            lhs3: __m256i,
            rhs0: __m256i,
            rhs1: __m256i,
        ) -> (__m256i, __m256i, __m256i, __m256i) {
            let avx2 = simd.avx2;

            let (x0, x1, x2, x3, _) = mul_u256_u64(simd, lhs0, lhs1, lhs2, lhs3, rhs0);
            let (y0, y1, y2, _, _) = mul_u256_u64(simd, lhs0, lhs1, lhs2, lhs3, rhs1);

            let z0 = x0;

            let z1 = avx2._mm256_add_epi64(x1, y0);
            let carry = simd._mm256_cmpgt_epu64(x1, z1);

            let z2 = avx2._mm256_add_epi64(x2, y1);
            let o0 = simd._mm256_cmpgt_epu64(x2, z2);
            let o1 = avx2._mm256_cmpeq_epi64(z2, carry);
            let z2 = avx2._mm256_sub_epi64(z2, carry);
            let carry = avx2._mm256_or_si256(o0, o1);

            let z3 = avx2._mm256_add_epi64(x3, y2);
            let z3 = avx2._mm256_sub_epi64(z3, carry);

            (z0, z1, z2, z3)
        }

        let (p, p_div0, p_div1, p_div2, p_div3) = p;

        let avx = simd.avx;
        let a = cast(a);
        let b = cast(b);
        let p = avx._mm256_set1_epi64x(p as _);
        let p_div0 = avx._mm256_set1_epi64x(p_div0 as _);
        let p_div1 = avx._mm256_set1_epi64x(p_div1 as _);
        let p_div2 = avx._mm256_set1_epi64x(p_div2 as _);
        let p_div3 = avx._mm256_set1_epi64x(p_div3 as _);

        let (lo, hi) = simd._mm256_mul_epu64(a, b);
        let (low_bits0, low_bits1, low_bits2, low_bits3) =
            wrapping_mul_u256_u128(simd, p_div0, p_div1, p_div2, p_div3, lo, hi);

        cast(mul_u256_u64(simd, low_bits0, low_bits1, low_bits2, low_bits3, p).4)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl PrimeModulusAvx2 for Solinas {
    type Div = ();

    #[inline(always)]
    fn add(self, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let a = cast(a);
        let b = cast(b);
        let p = simd.avx._mm256_set1_epi64x(Self::P as _);
        let neg_b = simd.avx2._mm256_sub_epi64(p, b);
        let not_a_ge_neg_b = simd._mm256_cmpgt_epu64(neg_b, a);
        cast(simd.avx2._mm256_blendv_epi8(
            simd.avx2._mm256_sub_epi64(a, neg_b),
            simd.avx2._mm256_add_epi64(a, b),
            not_a_ge_neg_b,
        ))
    }

    #[inline(always)]
    fn sub(self, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let a = cast(a);
        let b = cast(b);
        let p = simd.avx._mm256_set1_epi64x(Self::P as _);
        let neg_b = simd.avx2._mm256_sub_epi64(p, b);
        let not_a_ge_b = simd._mm256_cmpgt_epu64(b, a);
        cast(simd.avx2._mm256_blendv_epi8(
            simd.avx2._mm256_sub_epi64(a, b),
            simd.avx2._mm256_add_epi64(a, neg_b),
            not_a_ge_b,
        ))
    }

    #[inline(always)]
    fn mul(p: Self::Div, simd: Avx2, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let _ = p;

        let avx = simd.avx;
        let avx2 = simd.avx2;
        let a = cast(a);
        let b = cast(b);
        let p = avx._mm256_set1_epi64x(Self::P as _);

        // https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/
        let (lo, __hi) = simd._mm256_mul_epu64(a, b);
        let mid = avx2._mm256_and_si256(__hi, avx._mm256_set1_epi64x(0x00000000FFFFFFFF));
        let hi = avx2._mm256_and_si256(__hi, avx._mm256_set1_epi64x(0xFFFFFFFF00000000u64 as i64));
        let hi = avx2._mm256_srli_epi64::<32>(hi);

        let low2 = avx2._mm256_sub_epi64(lo, hi);
        let low2 = avx2._mm256_blendv_epi8(
            low2,
            avx2._mm256_add_epi64(low2, p),
            simd._mm256_cmpgt_epu64(hi, lo),
        );

        let product = avx2._mm256_slli_epi64::<32>(mid);
        let product = avx2._mm256_sub_epi64(product, mid);

        let result = avx2._mm256_add_epi64(low2, product);

        // (result < product) || (result >= p)
        // (result < product) || !(p > result)
        // !(!(result < product) && (p > result))
        let product_gt_result = simd._mm256_cmpgt_epu64(product, result);
        let p_gt_result = simd._mm256_cmpgt_epu64(p, result);
        let not_cond = avx2._mm256_andnot_si256(product_gt_result, p_gt_result);

        let result = avx2._mm256_blendv_epi8(avx2._mm256_sub_epi64(result, p), result, not_cond);

        cast(result)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl PrimeModulusAvx512 for u64 {
    type Div = (u64, u64, u64, u64, u64);

    #[inline(always)]
    fn add(self, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        let avx = simd.avx512f;
        let a = cast(a);
        let b = cast(b);
        let p = avx._mm512_set1_epi64(self as _);
        let neg_b = avx._mm512_sub_epi64(p, b);
        let a_ge_neg_b = avx._mm512_cmpge_epu64_mask(neg_b, a);
        cast(avx._mm512_mask_blend_epi64(
            a_ge_neg_b,
            avx._mm512_sub_epi64(a, neg_b),
            avx._mm512_add_epi64(a, b),
        ))
    }

    #[inline(always)]
    fn sub(self, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        let avx = simd.avx512f;
        let a = cast(a);
        let b = cast(b);
        let p = avx._mm512_set1_epi64(self as _);
        let neg_b = avx._mm512_sub_epi64(p, b);
        let a_ge_b = avx._mm512_cmpge_epu64_mask(b, a);
        cast(avx._mm512_mask_blend_epi64(
            a_ge_b,
            avx._mm512_sub_epi64(a, b),
            avx._mm512_add_epi64(a, neg_b),
        ))
    }

    #[inline(always)]
    fn mul(p: Self::Div, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        #[inline(always)]
        fn mul_with_carry(simd: Avx512, l: __m512i, r: __m512i, c: __m512i) -> (__m512i, __m512i) {
            let avx = simd.avx512f;
            let (lo, hi) = simd._mm512_mul_epu64(l, r);
            let lo_plus_c = avx._mm512_add_epi64(lo, c);
            let overflow = avx._mm512_cmpgt_epu64_mask(lo, lo_plus_c);

            (
                lo_plus_c,
                avx._mm512_sub_epi64(hi, simd._mm512_movm_epi64(overflow)),
            )
        }

        #[inline(always)]
        fn mul_u256_u64(
            simd: Avx512,
            lhs0: __m512i,
            lhs1: __m512i,
            lhs2: __m512i,
            lhs3: __m512i,
            rhs: __m512i,
        ) -> (__m512i, __m512i, __m512i, __m512i, __m512i) {
            let (x0, carry) = simd._mm512_mul_epu64(lhs0, rhs);
            let (x1, carry) = mul_with_carry(simd, lhs1, rhs, carry);
            let (x2, carry) = mul_with_carry(simd, lhs2, rhs, carry);
            let (x3, carry) = mul_with_carry(simd, lhs3, rhs, carry);
            (x0, x1, x2, x3, carry)
        }

        #[inline(always)]
        fn wrapping_mul_u256_u128(
            simd: Avx512,
            lhs0: __m512i,
            lhs1: __m512i,
            lhs2: __m512i,
            lhs3: __m512i,
            rhs0: __m512i,
            rhs1: __m512i,
        ) -> (__m512i, __m512i, __m512i, __m512i) {
            let avx = simd.avx512f;

            let (x0, x1, x2, x3, _) = mul_u256_u64(simd, lhs0, lhs1, lhs2, lhs3, rhs0);
            let (y0, y1, y2, _, _) = mul_u256_u64(simd, lhs0, lhs1, lhs2, lhs3, rhs1);

            let z0 = x0;

            let z1 = avx._mm512_add_epi64(x1, y0);
            let carry = simd._mm512_movm_epi64(avx._mm512_cmpgt_epu64_mask(x1, z1));

            let z2 = avx._mm512_add_epi64(x2, y1);
            let o0 = avx._mm512_cmpgt_epu64_mask(x2, z2);
            let o1 = avx._mm512_cmpeq_epi64_mask(z2, carry);
            let z2 = avx._mm512_sub_epi64(z2, carry);
            let carry = simd._mm512_movm_epi64(o0 | o1);

            let z3 = avx._mm512_add_epi64(x3, y2);
            let z3 = avx._mm512_sub_epi64(z3, carry);

            (z0, z1, z2, z3)
        }

        let (p, p_div0, p_div1, p_div2, p_div3) = p;

        let avx = simd.avx512f;
        let a = cast(a);
        let b = cast(b);
        let p = avx._mm512_set1_epi64(p as _);
        let p_div0 = avx._mm512_set1_epi64(p_div0 as _);
        let p_div1 = avx._mm512_set1_epi64(p_div1 as _);
        let p_div2 = avx._mm512_set1_epi64(p_div2 as _);
        let p_div3 = avx._mm512_set1_epi64(p_div3 as _);

        let (lo, hi) = simd._mm512_mul_epu64(a, b);
        let (low_bits0, low_bits1, low_bits2, low_bits3) =
            wrapping_mul_u256_u128(simd, p_div0, p_div1, p_div2, p_div3, lo, hi);

        cast(mul_u256_u64(simd, low_bits0, low_bits1, low_bits2, low_bits3, p).4)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl PrimeModulusAvx512 for Solinas {
    type Div = ();

    #[inline(always)]
    fn add(self, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        PrimeModulusAvx512::add(Self::P, simd, a, b)
    }

    #[inline(always)]
    fn sub(self, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        PrimeModulusAvx512::sub(Self::P, simd, a, b)
    }

    #[inline(always)]
    fn mul(p: Self::Div, simd: Avx512, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        let _ = p;

        let avx = simd.avx512f;
        let a = cast(a);
        let b = cast(b);
        let p = avx._mm512_set1_epi64(Self::P as _);

        // https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/
        let (lo, __hi) = simd._mm512_mul_epu64(a, b);
        let mid = avx._mm512_and_si512(__hi, avx._mm512_set1_epi64(0x00000000FFFFFFFF));
        let hi = avx._mm512_and_si512(__hi, avx._mm512_set1_epi64(0xFFFFFFFF00000000u64 as i64));
        let hi = avx._mm512_srli_epi64::<32>(hi);

        let low2 = avx._mm512_sub_epi64(lo, hi);
        let low2 = avx._mm512_mask_blend_epi64(
            avx._mm512_cmpgt_epu64_mask(hi, lo),
            low2,
            avx._mm512_add_epi64(low2, p),
        );

        let product = avx._mm512_slli_epi64::<32>(mid);
        let product = avx._mm512_sub_epi64(product, mid);

        let result = avx._mm512_add_epi64(low2, product);

        // (result < product) || (result >= p)
        // (result < product) || !(p > result)
        // !(!(result < product) && (p > result))
        let product_gt_result = avx._mm512_cmpgt_epu64_mask(product, result);
        let p_gt_result = avx._mm512_cmpgt_epu64_mask(p, result);
        let not_cond = !product_gt_result & p_gt_result;

        let result = avx._mm512_mask_blend_epi64(not_cond, avx._mm512_sub_epi64(result, p), result);

        cast(result)
    }
}

#[inline(always)]
pub fn fwd_breadth_first_scalar<P: PrimeModulus>(
    data: &mut [u64],
    p: P,
    p_div: P::Div,
    twid: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = n / 2;
    let mut m = 1;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    while m < n {
        let w = &twid[w_idx..];

        for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
            let (z0, z1) = data.split_at_mut(t);

            for (z0, z1) in zip(z0, z1) {
                let z1w = P::mul(p_div, *z1, w1);

                (*z0, *z1) = (p.add(*z0, z1w), p.sub(*z0, z1w));
            }
        }

        t /= 2;
        m *= 2;
        w_idx *= 2;
    }
}

#[inline(always)]
pub fn inv_breadth_first_scalar<P: PrimeModulus>(
    data: &mut [u64],
    p: P,
    p_div: P::Div,
    inv_twid: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = 1;
    let mut m = n;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    while m > 1 {
        m /= 2;
        w_idx /= 2;

        let w = &inv_twid[w_idx..];

        for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
            let (z0, z1) = data.split_at_mut(t);

            for (z0, z1) in zip(z0, z1) {
                (*z0, *z1) = (p.add(*z0, *z1), P::mul(p_div, p.sub(*z0, *z1), w1));
            }
        }

        t *= 2;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn fwd_breadth_first_avx2<P: PrimeModulusAvx2>(
    simd: Avx2,
    data: &mut [u64],
    p: P,
    p_div: P::Div,
    twid: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = n / 2;
    let mut m = 1;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    simd.vectorize(
        #[inline(always)]
        || {
            while m < n / 4 {
                let w = &twid[w_idx..];

                for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<4, _>(z0).0;
                    let z1 = as_arrays_mut::<4, _>(z1).0;
                    let w1 = cast(simd.avx._mm256_set1_epi64x(w1 as _));

                    for (z0, z1) in zip(z0, z1) {
                        let z1w = P::mul(p_div, simd, *z1, w1);
                        (*z0, *z1) = (p.add(simd, *z0, z1w), p.sub(simd, *z0, z1w));
                    }
                }

                t /= 2;
                m *= 2;
                w_idx *= 2;
            }

            // m = n / 4
            // t = 2
            {
                let interleave = {
                    #[inline(always)]
                    |z0z0z1z1: [[u64; 4]; 2]| -> [[u64; 4]; 2] {
                        [
                            cast(simd.avx._mm256_permute2f128_si256::<0b00100000>(
                                cast(z0z0z1z1[0]),
                                cast(z0z0z1z1[1]),
                            )),
                            cast(simd.avx._mm256_permute2f128_si256::<0b00110001>(
                                cast(z0z0z1z1[0]),
                                cast(z0z0z1z1[1]),
                            )),
                        ]
                    }
                };

                let permute = {
                    #[inline(always)]
                    |w: [u64; 2]| -> [u64; 4] {
                        let w00 = simd.sse2._mm_set1_epi64x(w[0] as _);
                        let w11 = simd.sse2._mm_set1_epi64x(w[1] as _);

                        let w0011 = simd.avx._mm256_insertf128_si256::<0b1>(
                            simd.avx._mm256_castsi128_si256(w00),
                            w11,
                        );

                        cast(w0011)
                    }
                };

                let w = as_arrays::<2, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<4, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, w1) in zip(data, w) {
                    let w1 = permute(*w1);
                    let [mut z0, mut z1] = interleave(*z0z0z1z1);
                    let z1w = P::mul(p_div, simd, z1, w1);
                    (z0, z1) = (p.add(simd, z0, z1w), p.sub(simd, z0, z1w));
                    *z0z0z1z1 = interleave([z0, z1]);
                }

                w_idx *= 2;
            }

            // m = n / 2
            // t = 1
            {
                let interleave = {
                    #[inline(always)]
                    |z0z1: [[u64; 4]; 2]| -> [[u64; 4]; 2] {
                        [
                            cast(
                                simd.avx2
                                    ._mm256_unpacklo_epi64(cast(z0z1[0]), cast(z0z1[1])),
                            ),
                            cast(
                                simd.avx2
                                    ._mm256_unpackhi_epi64(cast(z0z1[0]), cast(z0z1[1])),
                            ),
                        ]
                    }
                };

                let permute = {
                    #[inline(always)]
                    |w: [u64; 4]| -> [u64; 4] {
                        let avx = simd.avx;
                        let w0123 = cast(w);
                        let w0101 = avx._mm256_permute2f128_si256::<0b00000000>(w0123, w0123);
                        let w2323 = avx._mm256_permute2f128_si256::<0b00110011>(w0123, w0123);
                        let w0213 = avx._mm256_shuffle_pd::<0b1100>(cast(w0101), cast(w2323));
                        cast(w0213)
                    }
                };

                let w = as_arrays::<4, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<4, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, w1) in zip(data, w) {
                    let w1 = permute(*w1);
                    let [mut z0, mut z1] = interleave(*z0z0z1z1);
                    let z1w = P::mul(p_div, simd, z1, w1);
                    (z0, z1) = (p.add(simd, z0, z1w), p.sub(simd, z0, z1w));
                    *z0z0z1z1 = interleave([z0, z1]);
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn inv_breadth_first_avx2<P: PrimeModulusAvx2>(
    simd: Avx2,
    data: &mut [u64],
    p: P,
    p_div: P::Div,
    inv_twid: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = 1;
    let mut m = n;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    // m = n / 2
    // t = 1
    {
        m /= 2;
        w_idx /= 2;

        let interleave = {
            #[inline(always)]
            |z0z1: [[u64; 4]; 2]| -> [[u64; 4]; 2] {
                [
                    cast(
                        simd.avx2
                            ._mm256_unpacklo_epi64(cast(z0z1[0]), cast(z0z1[1])),
                    ),
                    cast(
                        simd.avx2
                            ._mm256_unpackhi_epi64(cast(z0z1[0]), cast(z0z1[1])),
                    ),
                ]
            }
        };

        let permute = {
            #[inline(always)]
            |w: [u64; 4]| -> [u64; 4] {
                let avx = simd.avx;
                let w0123 = cast(w);
                let w0101 = avx._mm256_permute2f128_si256::<0b00000000>(w0123, w0123);
                let w2323 = avx._mm256_permute2f128_si256::<0b00110011>(w0123, w0123);
                let w0213 = avx._mm256_shuffle_pd::<0b1100>(cast(w0101), cast(w2323));
                cast(w0213)
            }
        };

        let w = as_arrays::<4, _>(&inv_twid[w_idx..]).0;
        let data = as_arrays_mut::<4, _>(data).0;
        let data = as_arrays_mut::<2, _>(data).0;

        for (z0z1, w1) in zip(data, w) {
            let w1 = permute(*w1);
            let [mut z0, mut z1] = interleave(*z0z1);
            (z0, z1) = (
                p.add(simd, z0, z1),
                P::mul(p_div, simd, p.sub(simd, z0, z1), w1),
            );
            *z0z1 = interleave([z0, z1]);
        }

        t *= 2;
    }

    // m = n / 4
    // t = 2
    {
        m /= 2;
        w_idx /= 2;

        let interleave = {
            #[inline(always)]
            |z0z0z1z1: [[u64; 4]; 2]| -> [[u64; 4]; 2] {
                [
                    cast(simd.avx._mm256_permute2f128_si256::<0b00100000>(
                        cast(z0z0z1z1[0]),
                        cast(z0z0z1z1[1]),
                    )),
                    cast(simd.avx._mm256_permute2f128_si256::<0b00110001>(
                        cast(z0z0z1z1[0]),
                        cast(z0z0z1z1[1]),
                    )),
                ]
            }
        };

        let permute = {
            #[inline(always)]
            |w: [u64; 2]| -> [u64; 4] {
                let w00 = simd.sse2._mm_set1_epi64x(w[0] as _);
                let w11 = simd.sse2._mm_set1_epi64x(w[1] as _);

                let w0011 = simd
                    .avx
                    ._mm256_insertf128_si256::<0b1>(simd.avx._mm256_castsi128_si256(w00), w11);

                cast(w0011)
            }
        };

        let w = as_arrays::<2, _>(&inv_twid[w_idx..]).0;
        let data = as_arrays_mut::<4, _>(data).0;
        let data = as_arrays_mut::<2, _>(data).0;

        for (z0z0z1z1, w1) in zip(data, w) {
            let w1 = permute(*w1);
            let [mut z0, mut z1] = interleave(*z0z0z1z1);
            (z0, z1) = (
                p.add(simd, z0, z1),
                P::mul(p_div, simd, p.sub(simd, z0, z1), w1),
            );
            *z0z0z1z1 = interleave([z0, z1]);
        }

        t *= 2;
    }

    while m > 1 {
        m /= 2;
        w_idx /= 2;

        let w = &inv_twid[w_idx..];

        for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
            let (z0, z1) = data.split_at_mut(t);
            let z0 = as_arrays_mut::<4, _>(z0).0;
            let z1 = as_arrays_mut::<4, _>(z1).0;
            let w1 = cast(simd.avx._mm256_set1_epi64x(w1 as _));

            for (z0, z1) in zip(z0, z1) {
                (*z0, *z1) = (
                    p.add(simd, *z0, *z1),
                    P::mul(p_div, simd, p.sub(simd, *z0, *z1), w1),
                );
            }
        }

        t *= 2;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub fn fwd_breadth_first_avx512<P: PrimeModulusAvx512>(
    simd: Avx512,
    data: &mut [u64],
    p: P,
    p_div: P::Div,
    twid: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = n / 2;
    let mut m = 1;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    simd.vectorize(
        #[inline(always)]
        || {
            while m < n / 8 {
                let w = &twid[w_idx..];

                for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<8, _>(z0).0;
                    let z1 = as_arrays_mut::<8, _>(z1).0;
                    let w1 = cast(simd.avx512f._mm512_set1_epi64(w1 as _));

                    for (z0, z1) in zip(z0, z1) {
                        let z1w = P::mul(p_div, simd, *z1, w1);
                        (*z0, *z1) = (p.add(simd, *z0, z1w), p.sub(simd, *z0, z1w));
                    }
                }

                t /= 2;
                m *= 2;
                w_idx *= 2;
            }

            // m = n / 8
            // t = 4
            {
                // 0 1 -> 0 0 0 0 1 1 1 1
                let permute = {
                    #[inline(always)]
                    |w: [u64; 2]| -> [u64; 8] {
                        let avx512f = simd.avx512f;
                        let w = cast(w);
                        let w01xxxxxx = avx512f._mm512_castsi128_si512(w);
                        let idx = avx512f._mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);

                        cast(avx512f._mm512_permutexvar_epi64(idx, w01xxxxxx))
                    }
                };

                // 0 1 2 3 4 5 6 7 | 8 9 a b c d e f -> 0 1 2 3 8 9 a b | 4 5 6 7 c d e f
                let interleave = {
                    #[inline(always)]
                    |z0z0z1z1: [[u64; 8]; 2]| -> [[u64; 8]; 2] {
                        let avx512f = simd.avx512f;
                        let idx_0 =
                            avx512f._mm512_setr_epi64(0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xa, 0xb);
                        let idx_1 =
                            avx512f._mm512_setr_epi64(0x4, 0x5, 0x6, 0x7, 0xc, 0xd, 0xe, 0xf);
                        [
                            cast(avx512f._mm512_permutex2var_epi64(
                                cast(z0z0z1z1[0]),
                                idx_0,
                                cast(z0z0z1z1[1]),
                            )),
                            cast(avx512f._mm512_permutex2var_epi64(
                                cast(z0z0z1z1[0]),
                                idx_1,
                                cast(z0z0z1z1[1]),
                            )),
                        ]
                    }
                };

                let w = as_arrays::<2, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, w1) in zip(data, w) {
                    let w1 = permute(*w1);
                    let [mut z0, mut z1] = interleave(*z0z0z0z0z1z1z1z1);
                    let z1w = P::mul(p_div, simd, z1, w1);
                    (z0, z1) = (p.add(simd, z0, z1w), p.sub(simd, z0, z1w));
                    *z0z0z0z0z1z1z1z1 = interleave([z0, z1]);
                }

                w_idx *= 2;
            }

            // m = n / 4
            // t = 2
            {
                // 0 1 2 3 -> 0 0 2 2 1 1 3 3
                let permute = {
                    #[inline(always)]
                    |w: [u64; 4]| -> [u64; 8] {
                        let avx512f = simd.avx512f;
                        let w = cast(w);
                        let w0123xxxx = avx512f._mm512_castsi256_si512(w);
                        let idx = avx512f._mm512_setr_epi64(0, 0, 2, 2, 1, 1, 3, 3);

                        cast(avx512f._mm512_permutexvar_epi64(idx, w0123xxxx))
                    }
                };

                // 0 1 2 3 4 5 6 7 | 8 9 a b c d e f -> 0 1 8 9 4 5 c d | 2 3 a b 6 7 e f
                let interleave = {
                    #[inline(always)]
                    |z0z0z1z1: [[u64; 8]; 2]| -> [[u64; 8]; 2] {
                        let avx512f = simd.avx512f;
                        let idx_0 =
                            avx512f._mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xc, 0xd);
                        let idx_1 =
                            avx512f._mm512_setr_epi64(0x2, 0x3, 0xa, 0xb, 0x6, 0x7, 0xe, 0xf);
                        [
                            cast(avx512f._mm512_permutex2var_epi64(
                                cast(z0z0z1z1[0]),
                                idx_0,
                                cast(z0z0z1z1[1]),
                            )),
                            cast(avx512f._mm512_permutex2var_epi64(
                                cast(z0z0z1z1[0]),
                                idx_1,
                                cast(z0z0z1z1[1]),
                            )),
                        ]
                    }
                };

                let w = as_arrays::<4, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, w1) in zip(data, w) {
                    let w1 = permute(*w1);
                    let [mut z0, mut z1] = interleave(*z0z0z1z1);
                    let z1w = P::mul(p_div, simd, z1, w1);
                    (z0, z1) = (p.add(simd, z0, z1w), p.sub(simd, z0, z1w));
                    *z0z0z1z1 = interleave([z0, z1]);
                }

                w_idx *= 2;
            }

            // m = n / 2
            // t = 1
            {
                // 0 1 2 3 4 5 6 7 -> 0 4 1 5 2 6 3 7
                let permute = {
                    #[inline(always)]
                    |w: [u64; 8]| -> [u64; 8] {
                        let avx512f = simd.avx512f;
                        let w = cast(w);
                        let idx = avx512f._mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
                        cast(avx512f._mm512_permutexvar_epi64(idx, w))
                    }
                };

                // 0 1 2 3 4 5 6 7 | 8 9 a b c d e f -> 0 8 2 a 4 c 6 e | 1 9 3 b 5 d 7 f
                //
                // is its own inverse since:
                // 0 8 2 a 4 c 6 e | 1 9 3 b 5 d 7 f -> 0 1 2 3 4 5 6 7 | 8 9 a b c d e f
                let interleave = {
                    #[inline(always)]
                    |z0z1: [[u64; 8]; 2]| -> [[u64; 8]; 2] {
                        let avx512f = simd.avx512f;
                        [
                            cast(avx512f._mm512_unpacklo_epi64(cast(z0z1[0]), cast(z0z1[1]))),
                            cast(avx512f._mm512_unpackhi_epi64(cast(z0z1[0]), cast(z0z1[1]))),
                        ]
                    }
                };

                let w = as_arrays::<8, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, w1) in zip(data, w) {
                    let w1 = permute(*w1);
                    let [mut z0, mut z1] = interleave(*z0z1);
                    let z1w = P::mul(p_div, simd, z1, w1);
                    (z0, z1) = (p.add(simd, z0, z1w), p.sub(simd, z0, z1w));
                    *z0z1 = interleave([z0, z1]);
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub fn inv_breadth_first_avx512<P: PrimeModulusAvx512>(
    simd: Avx512,
    data: &mut [u64],
    p: P,
    p_div: P::Div,
    inv_twid: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = 1;
    let mut m = n;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    // m = n / 2
    // t = 1
    {
        m /= 2;
        w_idx /= 2;

        let interleave = {
            #[inline(always)]
            |z0z1: [[u64; 8]; 2]| -> [[u64; 8]; 2] {
                let avx512f = simd.avx512f;
                [
                    cast(avx512f._mm512_unpacklo_epi64(cast(z0z1[0]), cast(z0z1[1]))),
                    cast(avx512f._mm512_unpackhi_epi64(cast(z0z1[0]), cast(z0z1[1]))),
                ]
            }
        };

        let permute = {
            #[inline(always)]
            |w: [u64; 8]| -> [u64; 8] {
                let avx512f = simd.avx512f;
                let w = cast(w);
                let idx = avx512f._mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
                cast(avx512f._mm512_permutexvar_epi64(idx, w))
            }
        };

        let w = as_arrays::<8, _>(&inv_twid[w_idx..]).0;
        let data = as_arrays_mut::<8, _>(data).0;
        let data = as_arrays_mut::<2, _>(data).0;

        for (z0z1, w1) in zip(data, w) {
            let w1 = permute(*w1);
            let [mut z0, mut z1] = interleave(*z0z1);
            (z0, z1) = (
                p.add(simd, z0, z1),
                P::mul(p_div, simd, p.sub(simd, z0, z1), w1),
            );
            *z0z1 = interleave([z0, z1]);
        }

        t *= 2;
    }

    // m = n / 4
    // t = 2
    {
        m /= 2;
        w_idx /= 2;

        // 0 1 2 3 -> 0 0 2 2 1 1 3 3
        let permute = {
            #[inline(always)]
            |w: [u64; 4]| -> [u64; 8] {
                let avx512f = simd.avx512f;
                let w = cast(w);
                let w0123xxxx = avx512f._mm512_castsi256_si512(w);
                let idx = avx512f._mm512_setr_epi64(0, 0, 2, 2, 1, 1, 3, 3);

                cast(avx512f._mm512_permutexvar_epi64(idx, w0123xxxx))
            }
        };

        // 0 1 2 3 4 5 6 7 | 8 9 a b c d e f -> 0 1 8 9 4 5 c d | 2 3 a b 6 7 e f
        let interleave = {
            #[inline(always)]
            |z0z0z1z1: [[u64; 8]; 2]| -> [[u64; 8]; 2] {
                let avx512f = simd.avx512f;
                let idx_0 = avx512f._mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xc, 0xd);
                let idx_1 = avx512f._mm512_setr_epi64(0x2, 0x3, 0xa, 0xb, 0x6, 0x7, 0xe, 0xf);
                [
                    cast(avx512f._mm512_permutex2var_epi64(
                        cast(z0z0z1z1[0]),
                        idx_0,
                        cast(z0z0z1z1[1]),
                    )),
                    cast(avx512f._mm512_permutex2var_epi64(
                        cast(z0z0z1z1[0]),
                        idx_1,
                        cast(z0z0z1z1[1]),
                    )),
                ]
            }
        };

        let w = as_arrays::<4, _>(&inv_twid[w_idx..]).0;
        let data = as_arrays_mut::<8, _>(data).0;
        let data = as_arrays_mut::<2, _>(data).0;

        for (z0z0z1z1, w1) in zip(data, w) {
            let w1 = permute(*w1);
            let [mut z0, mut z1] = interleave(*z0z0z1z1);
            (z0, z1) = (
                p.add(simd, z0, z1),
                P::mul(p_div, simd, p.sub(simd, z0, z1), w1),
            );
            *z0z0z1z1 = interleave([z0, z1]);
        }

        t *= 2;
    }

    // m = n / 8
    // t = 4
    {
        m /= 2;
        w_idx /= 2;

        // 0 1 -> 0 0 0 0 1 1 1 1
        let permute = {
            #[inline(always)]
            |w: [u64; 2]| -> [u64; 8] {
                let avx512f = simd.avx512f;
                let w = cast(w);
                let w01xxxxxx = avx512f._mm512_castsi128_si512(w);
                let idx = avx512f._mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);

                cast(avx512f._mm512_permutexvar_epi64(idx, w01xxxxxx))
            }
        };

        // 0 1 2 3 4 5 6 7 | 8 9 a b c d e f -> 0 1 2 3 8 9 a b | 4 5 6 7 c d e f
        let interleave = {
            #[inline(always)]
            |z0z0z1z1: [[u64; 8]; 2]| -> [[u64; 8]; 2] {
                let avx512f = simd.avx512f;
                let idx_0 = avx512f._mm512_setr_epi64(0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xa, 0xb);
                let idx_1 = avx512f._mm512_setr_epi64(0x4, 0x5, 0x6, 0x7, 0xc, 0xd, 0xe, 0xf);
                [
                    cast(avx512f._mm512_permutex2var_epi64(
                        cast(z0z0z1z1[0]),
                        idx_0,
                        cast(z0z0z1z1[1]),
                    )),
                    cast(avx512f._mm512_permutex2var_epi64(
                        cast(z0z0z1z1[0]),
                        idx_1,
                        cast(z0z0z1z1[1]),
                    )),
                ]
            }
        };

        let w = as_arrays::<2, _>(&inv_twid[w_idx..]).0;
        let data = as_arrays_mut::<8, _>(data).0;
        let data = as_arrays_mut::<2, _>(data).0;

        for (z0z0z1z1, w1) in zip(data, w) {
            let w1 = permute(*w1);
            let [mut z0, mut z1] = interleave(*z0z0z1z1);
            (z0, z1) = (
                p.add(simd, z0, z1),
                P::mul(p_div, simd, p.sub(simd, z0, z1), w1),
            );
            *z0z0z1z1 = interleave([z0, z1]);
        }

        t *= 2;
    }

    while m > 1 {
        m /= 2;
        w_idx /= 2;

        let w = &inv_twid[w_idx..];

        for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
            let (z0, z1) = data.split_at_mut(t);
            let z0 = as_arrays_mut::<8, _>(z0).0;
            let z1 = as_arrays_mut::<8, _>(z1).0;
            let w1 = cast(simd.avx512f._mm512_set1_epi64(w1 as _));

            for (z0, z1) in zip(z0, z1) {
                (*z0, *z1) = (
                    p.add(simd, *z0, *z1),
                    P::mul(p_div, simd, p.sub(simd, *z0, *z1), w1),
                );
            }
        }

        t *= 2;
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

#[derive(Debug, Clone)]
pub struct Plan {
    twid: ABox<[u64]>,
    inv_twid: ABox<[u64]>,
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

            init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

            Some(Self {
                twid,
                inv_twid,
                p,
                p_div,
            })
        }
    }

    pub fn fwd(&self, buf: &mut [u64]) {
        let p = self.p;
        if p == Solinas::P {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = Avx512::try_new() {
                    return fwd_breadth_first_avx512(simd, buf, Solinas, (), &self.twid, 0, 0);
                }
                if let Some(simd) = Avx2::try_new() {
                    return fwd_breadth_first_avx2(simd, buf, Solinas, (), &self.twid, 0, 0);
                }
            }
            fwd_breadth_first_scalar(buf, Solinas, (), &self.twid, 0, 0);
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "nightly")]
            if let Some(simd) = Avx512::try_new() {
                let u256 { x0, x1, x2, x3 } = self.p_div.double_reciprocal;
                let p_div = (p, x0, x1, x2, x3);
                return fwd_breadth_first_avx512(simd, buf, p, p_div, &self.twid, 0, 0);
            }
            fwd_breadth_first_scalar(buf, p, self.p_div, &self.twid, 0, 0);
        }
    }

    pub fn inv(&self, buf: &mut [u64]) {
        let p = self.p;
        if p == Solinas::P {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if let Some(simd) = Avx512::try_new() {
                    return inv_breadth_first_avx512(simd, buf, Solinas, (), &self.inv_twid, 0, 0);
                }
                if let Some(simd) = Avx2::try_new() {
                    return inv_breadth_first_avx2(simd, buf, Solinas, (), &self.inv_twid, 0, 0);
                }
            }
            inv_breadth_first_scalar(buf, Solinas, (), &self.inv_twid, 0, 0);
        } else {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[cfg(feature = "nightly")]
            if let Some(simd) = Avx512::try_new() {
                let u256 { x0, x1, x2, x3 } = self.p_div.double_reciprocal;
                let p_div = (p, x0, x1, x2, x3);
                return inv_breadth_first_avx512(simd, buf, p, p_div, &self.inv_twid, 0, 0);
            }
            inv_breadth_first_scalar(buf, p, self.p_div, &self.inv_twid, 0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use rand::random;

    extern crate alloc;

    #[test]
    fn test_product() {
        for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let p = Solinas::P;

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

            let mut twid = vec![0u64; n];
            let mut inv_twid = vec![0u64; n];
            init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

            let mut prod = vec![0u64; n];
            let mut lhs_fourier = lhs.clone();
            let mut rhs_fourier = rhs.clone();

            fwd_breadth_first_scalar(&mut lhs_fourier, p, Div64::new(p), &twid, 0, 0);
            fwd_breadth_first_scalar(&mut rhs_fourier, p, Div64::new(p), &twid, 0, 0);

            for i in 0..n {
                prod[i] = <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
            }

            inv_breadth_first_scalar(&mut prod, p, Div64::new(p), &inv_twid, 0, 0);
            let result = prod;

            for i in 0..n {
                assert_eq!(
                    result[i],
                    <u64 as PrimeModulus>::mul(Div64::new(p), negacyclic_convolution[i], n as u64),
                );
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_product_avx2() {
        if let Some(simd) = Avx2::try_new() {
            for n in [8, 16, 32, 64, 128, 256, 512, 1024] {
                let p = Solinas::P;

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

                let mut twid = vec![0u64; n];
                let mut inv_twid = vec![0u64; n];
                init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                let u256 { x0, x1, x2, x3 } = Div64::new(p).double_reciprocal;
                fwd_breadth_first_avx2(simd, &mut lhs_fourier, p, (p, x0, x1, x2, x3), &twid, 0, 0);
                fwd_breadth_first_avx2(simd, &mut rhs_fourier, p, (p, x0, x1, x2, x3), &twid, 0, 0);

                for i in 0..n {
                    prod[i] =
                        <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_breadth_first_avx2(simd, &mut prod, p, (p, x0, x1, x2, x3), &inv_twid, 0, 0);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i],
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly")]
    #[test]
    fn test_product_avx512() {
        if let Some(simd) = Avx512::try_new() {
            for n in [16, 32, 64, 128, 256, 512, 1024] {
                let p = Solinas::P;

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

                let mut twid = vec![0u64; n];
                let mut inv_twid = vec![0u64; n];
                init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                let u256 { x0, x1, x2, x3 } = Div64::new(p).double_reciprocal;
                fwd_breadth_first_avx512(
                    simd,
                    &mut lhs_fourier,
                    p,
                    (p, x0, x1, x2, x3),
                    &twid,
                    0,
                    0,
                );
                fwd_breadth_first_avx512(
                    simd,
                    &mut rhs_fourier,
                    p,
                    (p, x0, x1, x2, x3),
                    &twid,
                    0,
                    0,
                );

                for i in 0..n {
                    prod[i] =
                        <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_breadth_first_avx512(simd, &mut prod, p, (p, x0, x1, x2, x3), &inv_twid, 0, 0);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i],
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

    #[test]
    fn test_product_solinas() {
        for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let p = Solinas::P;

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

            let mut twid = vec![0u64; n];
            let mut inv_twid = vec![0u64; n];
            init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

            let mut prod = vec![0u64; n];
            let mut lhs_fourier = lhs.clone();
            let mut rhs_fourier = rhs.clone();

            fwd_breadth_first_scalar(&mut lhs_fourier, Solinas, (), &twid, 0, 0);
            fwd_breadth_first_scalar(&mut rhs_fourier, Solinas, (), &twid, 0, 0);

            for i in 0..n {
                prod[i] = <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
            }

            inv_breadth_first_scalar(&mut prod, Solinas, (), &inv_twid, 0, 0);
            let result = prod;

            for i in 0..n {
                assert_eq!(
                    result[i],
                    <u64 as PrimeModulus>::mul(Div64::new(p), negacyclic_convolution[i], n as u64),
                );
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_product_solinas_avx2() {
        if let Some(simd) = Avx2::try_new() {
            for n in [8, 16, 32, 64, 128, 256, 512, 1024] {
                let p = Solinas::P;

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

                let mut twid = vec![0u64; n];
                let mut inv_twid = vec![0u64; n];
                init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_breadth_first_avx2(simd, &mut lhs_fourier, Solinas, (), &twid, 0, 0);
                fwd_breadth_first_avx2(simd, &mut rhs_fourier, Solinas, (), &twid, 0, 0);

                for i in 0..n {
                    prod[i] =
                        <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_breadth_first_avx2(simd, &mut prod, Solinas, (), &inv_twid, 0, 0);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i],
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly")]
    #[test]
    fn test_product_solinas_avx512() {
        if let Some(simd) = Avx512::try_new() {
            for n in [16, 32, 64, 128, 256, 512, 1024] {
                let p = Solinas::P;

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

                let mut twid = vec![0u64; n];
                let mut inv_twid = vec![0u64; n];
                init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_breadth_first_avx512(simd, &mut lhs_fourier, Solinas, (), &twid, 0, 0);
                fwd_breadth_first_avx512(simd, &mut rhs_fourier, Solinas, (), &twid, 0, 0);

                for i in 0..n {
                    prod[i] =
                        <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_breadth_first_avx512(simd, &mut prod, Solinas, (), &inv_twid, 0, 0);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i],
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
