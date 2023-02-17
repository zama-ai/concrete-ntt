use super::RECURSION_THRESHOLD;
use crate::fastdiv::Div32;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
use crate::Avx512;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::iter::zip;
use pulp::{as_arrays_mut, cast};

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn add_avx2(simd: Avx2, p: __m256i, a: __m256i, b: __m256i) -> __m256i {
    let neg_b = simd.avx2._mm256_sub_epi32(p, b);
    let not_a_ge_neg_b = simd._mm256_cmpgt_epu32(neg_b, a);
    simd.avx2._mm256_blendv_epi8(
        simd.avx2._mm256_sub_epi32(a, neg_b),
        simd.avx2._mm256_add_epi32(a, b),
        not_a_ge_neg_b,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn sub_avx2(simd: Avx2, p: __m256i, a: __m256i, b: __m256i) -> __m256i {
    let neg_b = simd.avx2._mm256_sub_epi32(p, b);
    let not_a_ge_b = simd._mm256_cmpgt_epu32(b, a);
    simd.avx2._mm256_blendv_epi8(
        simd.avx2._mm256_sub_epi32(a, b),
        simd.avx2._mm256_add_epi32(a, neg_b),
        not_a_ge_b,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn mul_avx2(
    simd: Avx2,
    p: __m256i,
    p_div0: __m256i,
    p_div1: __m256i,
    p_div2: __m256i,
    p_div3: __m256i,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    #[inline(always)]
    fn mul_with_carry(simd: Avx2, l: __m256i, r: __m256i, c: __m256i) -> (__m256i, __m256i) {
        let (lo, hi) = simd._mm256_mul_u32_u32_epu32(l, r);
        let lo_plus_c = simd.avx2._mm256_add_epi32(lo, c);
        let overflow = simd._mm256_cmpgt_epu32(lo, lo_plus_c);
        (lo_plus_c, simd.avx2._mm256_sub_epi32(hi, overflow))
    }
    #[inline(always)]
    fn mul_u128_u32(
        simd: Avx2,
        lhs0: __m256i,
        lhs1: __m256i,
        lhs2: __m256i,
        lhs3: __m256i,
        rhs: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i, __m256i) {
        let (x0, carry) = simd._mm256_mul_u32_u32_epu32(lhs0, rhs);
        let (x1, carry) = mul_with_carry(simd, lhs1, rhs, carry);
        let (x2, carry) = mul_with_carry(simd, lhs2, rhs, carry);
        let (x3, carry) = mul_with_carry(simd, lhs3, rhs, carry);
        (x0, x1, x2, x3, carry)
    }

    #[inline(always)]
    fn wrapping_mul_u128_u64(
        simd: Avx2,
        lhs0: __m256i,
        lhs1: __m256i,
        lhs2: __m256i,
        lhs3: __m256i,
        rhs0: __m256i,
        rhs1: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        let avx2 = simd.avx2;

        let (x0, x1, x2, x3, _) = mul_u128_u32(simd, lhs0, lhs1, lhs2, lhs3, rhs0);
        let (y0, y1, y2, _, _) = mul_u128_u32(simd, lhs0, lhs1, lhs2, lhs3, rhs1);

        let z0 = x0;

        let z1 = avx2._mm256_add_epi32(x1, y0);
        let carry = simd._mm256_cmpgt_epu32(x1, z1);

        let z2 = avx2._mm256_add_epi32(x2, y1);
        let o0 = simd._mm256_cmpgt_epu32(x2, z2);
        let o1 = avx2._mm256_cmpeq_epi32(z2, carry);
        let z2 = avx2._mm256_sub_epi32(z2, carry);
        let carry = avx2._mm256_or_si256(o0, o1);

        let z3 = avx2._mm256_add_epi32(x3, y2);
        let z3 = avx2._mm256_sub_epi32(z3, carry);

        (z0, z1, z2, z3)
    }

    let (lo, hi) = simd._mm256_mul_u32_u32_epu32(a, b);
    let (low_bits0, low_bits1, low_bits2, low_bits3) =
        wrapping_mul_u128_u64(simd, p_div0, p_div1, p_div2, p_div3, lo, hi);

    mul_u128_u32(simd, low_bits0, low_bits1, low_bits2, low_bits3, p).4
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn add_avx512(simd: Avx512, p: __m512i, a: __m512i, b: __m512i) -> __m512i {
    let avx = simd.avx512f;
    let neg_b = avx._mm512_sub_epi32(p, b);
    let a_ge_neg_b = avx._mm512_cmpge_epu32_mask(a, neg_b);
    avx._mm512_mask_blend_epi32(
        a_ge_neg_b,
        avx._mm512_add_epi32(a, b),
        avx._mm512_sub_epi32(a, neg_b),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn sub_avx512(simd: Avx512, p: __m512i, a: __m512i, b: __m512i) -> __m512i {
    let avx = simd.avx512f;
    let neg_b = avx._mm512_sub_epi32(p, b);
    let a_ge_b = avx._mm512_cmpge_epu32_mask(a, b);
    avx._mm512_mask_blend_epi32(
        a_ge_b,
        avx._mm512_add_epi32(a, neg_b),
        avx._mm512_sub_epi32(a, b),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn mul_avx512(
    simd: Avx512,
    p: __m512i,
    p_div0: __m512i,
    p_div1: __m512i,
    p_div2: __m512i,
    p_div3: __m512i,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    #[inline(always)]
    fn mul_with_carry(simd: Avx512, l: __m512i, r: __m512i, c: __m512i) -> (__m512i, __m512i) {
        let avx = simd.avx512f;
        let (lo, hi) = simd._mm512_mul_u32_u32_epu32(l, r);
        let lo_plus_c = avx._mm512_add_epi32(lo, c);
        let overflow = simd._mm512_movm_epi32(avx._mm512_cmpgt_epu32_mask(lo, lo_plus_c));
        (lo_plus_c, avx._mm512_sub_epi32(hi, overflow))
    }
    #[inline(always)]
    fn mul_u128_u32(
        simd: Avx512,
        lhs0: __m512i,
        lhs1: __m512i,
        lhs2: __m512i,
        lhs3: __m512i,
        rhs: __m512i,
    ) -> (__m512i, __m512i, __m512i, __m512i, __m512i) {
        let (x0, carry) = simd._mm512_mul_u32_u32_epu32(lhs0, rhs);
        let (x1, carry) = mul_with_carry(simd, lhs1, rhs, carry);
        let (x2, carry) = mul_with_carry(simd, lhs2, rhs, carry);
        let (x3, carry) = mul_with_carry(simd, lhs3, rhs, carry);
        (x0, x1, x2, x3, carry)
    }

    #[inline(always)]
    fn wrapping_mul_u128_u64(
        simd: Avx512,
        lhs0: __m512i,
        lhs1: __m512i,
        lhs2: __m512i,
        lhs3: __m512i,
        rhs0: __m512i,
        rhs1: __m512i,
    ) -> (__m512i, __m512i, __m512i, __m512i) {
        let avx = simd.avx512f;

        let (x0, x1, x2, x3, _) = mul_u128_u32(simd, lhs0, lhs1, lhs2, lhs3, rhs0);
        let (y0, y1, y2, _, _) = mul_u128_u32(simd, lhs0, lhs1, lhs2, lhs3, rhs1);

        let z0 = x0;

        let z1 = avx._mm512_add_epi32(x1, y0);
        let carry = simd._mm512_movm_epi32(avx._mm512_cmpgt_epu32_mask(x1, z1));

        let z2 = avx._mm512_add_epi32(x2, y1);
        let o0 = avx._mm512_cmpgt_epu32_mask(x2, z2);
        let o1 = avx._mm512_cmpeq_epi32_mask(z2, carry);
        let z2 = avx._mm512_sub_epi32(z2, carry);
        let carry = simd._mm512_movm_epi32(o0 | o1);

        let z3 = avx._mm512_add_epi32(x3, y2);
        let z3 = avx._mm512_sub_epi32(z3, carry);

        (z0, z1, z2, z3)
    }

    let (lo, hi) = simd._mm512_mul_u32_u32_epu32(a, b);
    let (low_bits0, low_bits1, low_bits2, low_bits3) =
        wrapping_mul_u128_u64(simd, p_div0, p_div1, p_div2, p_div3, lo, hi);

    mul_u128_u32(simd, low_bits0, low_bits1, low_bits2, low_bits3, p).4
}

pub fn fwd_breadth_first_scalar(
    data: &mut [u32],
    p: u32,
    p_div: Div32,
    twid: &[u32],
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
                let z1w = mul(p_div, *z1, w1);

                (*z0, *z1) = (add(p, *z0, z1w), sub(p, *z0, z1w));
            }
        }

        t /= 2;
        m *= 2;
        w_idx *= 2;
    }
}

pub fn fwd_depth_first_scalar(
    data: &mut [u32],
    p: u32,
    p_div: Div32,
    twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    if n <= RECURSION_THRESHOLD {
        fwd_breadth_first_scalar(data, p, p_div, twid, recursion_depth, recursion_half);
    } else {
        let t = n / 2;
        let m = 1;
        let w_idx = (m << recursion_depth) + m * recursion_half;

        let w = &twid[w_idx..];

        for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
            let (z0, z1) = data.split_at_mut(t);

            for (z0, z1) in zip(z0, z1) {
                let z1w = mul(p_div, *z1, w1);

                (*z0, *z1) = (add(p, *z0, z1w), sub(p, *z0, z1w));
            }
        }

        let (data0, data1) = data.split_at_mut(n / 2);
        fwd_depth_first_scalar(
            data0,
            p,
            p_div,
            twid,
            recursion_depth + 1,
            recursion_half * 2,
        );
        fwd_depth_first_scalar(
            data1,
            p,
            p_div,
            twid,
            recursion_depth + 1,
            recursion_half * 2 + 1,
        );
    }
}

pub fn inv_breadth_first_scalar(
    data: &mut [u32],
    p: u32,
    p_div: Div32,
    inv_twid: &[u32],
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
                (*z0, *z1) = (add(p, *z0, *z1), mul(p_div, sub(p, *z0, *z1), w1));
            }
        }

        t *= 2;
    }
}

pub fn inv_depth_first_scalar(
    data: &mut [u32],
    p: u32,
    p_div: Div32,
    inv_twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= RECURSION_THRESHOLD {
        inv_breadth_first_scalar(data, p, p_div, inv_twid, recursion_depth, recursion_half);
    } else {
        let (data0, data1) = data.split_at_mut(n / 2);
        inv_depth_first_scalar(
            data0,
            p,
            p_div,
            inv_twid,
            recursion_depth + 1,
            recursion_half * 2,
        );
        inv_depth_first_scalar(
            data1,
            p,
            p_div,
            inv_twid,
            recursion_depth + 1,
            recursion_half * 2 + 1,
        );

        let t = n / 2;
        let m = 1;
        let w_idx = (m << recursion_depth) + m * recursion_half;

        let w = &inv_twid[w_idx..];

        for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
            let (z0, z1) = data.split_at_mut(t);

            for (z0, z1) in zip(z0, z1) {
                (*z0, *z1) = (add(p, *z0, *z1), mul(p_div, sub(p, *z0, *z1), w1));
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_breadth_first_avx2(
    simd: Avx2,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    use pulp::as_arrays;

    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            let mut t = n / 2;
            let mut m = 1;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;
            let p = simd.avx._mm256_set1_epi32(p as i32);
            let p_div0 = simd.avx._mm256_set1_epi32(p_div.0 as i32);
            let p_div1 = simd.avx._mm256_set1_epi32(p_div.1 as i32);
            let p_div2 = simd.avx._mm256_set1_epi32(p_div.2 as i32);
            let p_div3 = simd.avx._mm256_set1_epi32(p_div.3 as i32);

            while m < n / 8 {
                let w = &twid[w_idx..];

                for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<8, _>(z0).0;
                    let z1 = as_arrays_mut::<8, _>(z1).0;
                    let w1 = simd.avx._mm256_set1_epi32(w1 as _);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        let z1w = mul_avx2(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                        (z0, z1) = (add_avx2(simd, p, z0, z1w), sub_avx2(simd, p, z0, z1w));
                        *__z0 = cast(z0);
                        *__z1 = cast(z1);
                    }
                }

                t /= 2;
                m *= 2;
                w_idx *= 2;
            }

            // m = n / 8
            // t = 4
            {
                let w = as_arrays::<2, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute4_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    let z1w = mul_avx2(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                    (z0, z1) = (add_avx2(simd, p, z0, z1w), sub_avx2(simd, p, z0, z1w));
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 4
            // t = 2
            {
                let w = as_arrays::<4, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute2_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    let z1w = mul_avx2(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                    (z0, z1) = (add_avx2(simd, p, z0, z1w), sub_avx2(simd, p, z0, z1w));
                    *z0z0z1z1 = cast(simd.interleave2_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 2
            // t = 1
            {
                let w = as_arrays::<8, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, w1) in zip(data, w) {
                    let w1 = simd.permute1_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    let z1w = mul_avx2(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                    (z0, z1) = (add_avx2(simd, p, z0, z1w), sub_avx2(simd, p, z0, z1w));
                    *z0z1 = cast(simd.interleave1_epu32([z0, z1]));
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_depth_first_avx2(
    simd: Avx2,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                fwd_breadth_first_avx2(simd, data, p, p_div, twid, recursion_depth, recursion_half);
            } else {
                {
                    let t = n / 2;
                    let m = 1;
                    let w_idx = (m << recursion_depth) + m * recursion_half;
                    let w = &twid[w_idx..];
                    let p = simd.avx._mm256_set1_epi32(p as i32);
                    let p_div0 = simd.avx._mm256_set1_epi32(p_div.0 as i32);
                    let p_div1 = simd.avx._mm256_set1_epi32(p_div.1 as i32);
                    let p_div2 = simd.avx._mm256_set1_epi32(p_div.2 as i32);
                    let p_div3 = simd.avx._mm256_set1_epi32(p_div.3 as i32);

                    for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<8, _>(z0).0;
                        let z1 = as_arrays_mut::<8, _>(z1).0;
                        let w1 = simd.avx._mm256_set1_epi32(w1 as _);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            let z1w = mul_avx2(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                            (z0, z1) = (add_avx2(simd, p, z0, z1w), sub_avx2(simd, p, z0, z1w));
                            *__z0 = cast(z0);
                            *__z1 = cast(z1);
                        }
                    }
                }

                let (data0, data1) = data.split_at_mut(n / 2);
                fwd_depth_first_avx2(
                    simd,
                    data0,
                    p,
                    p_div,
                    twid,
                    recursion_depth + 1,
                    recursion_half * 2,
                );
                fwd_depth_first_avx2(
                    simd,
                    data1,
                    p,
                    p_div,
                    twid,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                );
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn inv_breadth_first_avx2(
    simd: Avx2,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    inv_twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    use pulp::as_arrays;

    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            let mut t = 1;
            let mut m = n;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;
            let p = simd.avx._mm256_set1_epi32(p as i32);
            let p_div0 = simd.avx._mm256_set1_epi32(p_div.0 as i32);
            let p_div1 = simd.avx._mm256_set1_epi32(p_div.1 as i32);
            let p_div2 = simd.avx._mm256_set1_epi32(p_div.2 as i32);
            let p_div3 = simd.avx._mm256_set1_epi32(p_div.3 as i32);

            // m = n / 2
            // t = 1
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<8, _>(&inv_twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, w1) in zip(data, w) {
                    let w1 = simd.permute1_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    (z0, z1) = (
                        add_avx2(simd, p, z0, z1),
                        mul_avx2(
                            simd,
                            p,
                            p_div0,
                            p_div1,
                            p_div2,
                            p_div3,
                            sub_avx2(simd, p, z0, z1),
                            w1,
                        ),
                    );
                    *z0z1 = cast(simd.interleave1_epu32([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 4
            // t = 2
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<4, _>(&inv_twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute2_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    (z0, z1) = (
                        add_avx2(simd, p, z0, z1),
                        mul_avx2(
                            simd,
                            p,
                            p_div0,
                            p_div1,
                            p_div2,
                            p_div3,
                            sub_avx2(simd, p, z0, z1),
                            w1,
                        ),
                    );
                    *z0z0z1z1 = cast(simd.interleave2_epu32([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 8
            // t = 4
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<2, _>(&inv_twid[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute4_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = (
                        add_avx2(simd, p, z0, z1),
                        mul_avx2(
                            simd,
                            p,
                            p_div0,
                            p_div1,
                            p_div2,
                            p_div3,
                            sub_avx2(simd, p, z0, z1),
                            w1,
                        ),
                    );
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu32([z0, z1]));
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
                    let w1 = simd.avx._mm256_set1_epi32(w1 as _);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        (z0, z1) = (
                            add_avx2(simd, p, z0, z1),
                            mul_avx2(
                                simd,
                                p,
                                p_div0,
                                p_div1,
                                p_div2,
                                p_div3,
                                sub_avx2(simd, p, z0, z1),
                                w1,
                            ),
                        );
                        *__z0 = cast(z0);
                        *__z1 = cast(z1);
                    }
                }

                t *= 2;
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn inv_depth_first_avx2(
    simd: Avx2,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    inv_twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                inv_breadth_first_avx2(
                    simd,
                    data,
                    p,
                    p_div,
                    inv_twid,
                    recursion_depth,
                    recursion_half,
                );
            } else {
                let (data0, data1) = data.split_at_mut(n / 2);
                inv_depth_first_avx2(
                    simd,
                    data0,
                    p,
                    p_div,
                    inv_twid,
                    recursion_depth + 1,
                    recursion_half * 2,
                );
                inv_depth_first_avx2(
                    simd,
                    data1,
                    p,
                    p_div,
                    inv_twid,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                );

                {
                    let t = n / 2;
                    let m = 1;
                    let w_idx = (m << recursion_depth) + m * recursion_half;
                    let w = &inv_twid[w_idx..];
                    let p = simd.avx._mm256_set1_epi32(p as i32);
                    let p_div0 = simd.avx._mm256_set1_epi32(p_div.0 as i32);
                    let p_div1 = simd.avx._mm256_set1_epi32(p_div.1 as i32);
                    let p_div2 = simd.avx._mm256_set1_epi32(p_div.2 as i32);
                    let p_div3 = simd.avx._mm256_set1_epi32(p_div.3 as i32);

                    for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<8, _>(z0).0;
                        let z1 = as_arrays_mut::<8, _>(z1).0;
                        let w1 = simd.avx._mm256_set1_epi32(w1 as _);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            (z0, z1) = (
                                add_avx2(simd, p, z0, z1),
                                mul_avx2(
                                    simd,
                                    p,
                                    p_div0,
                                    p_div1,
                                    p_div2,
                                    p_div3,
                                    sub_avx2(simd, p, z0, z1),
                                    w1,
                                ),
                            );
                            *__z0 = cast(z0);
                            *__z1 = cast(z1);
                        }
                    }
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn fwd_breadth_first_avx512(
    simd: Avx512,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    use pulp::as_arrays;

    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            let mut t = n / 2;
            let mut m = 1;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;
            let p = simd.avx512f._mm512_set1_epi32(p as i32);
            let p_div0 = simd.avx512f._mm512_set1_epi32(p_div.0 as i32);
            let p_div1 = simd.avx512f._mm512_set1_epi32(p_div.1 as i32);
            let p_div2 = simd.avx512f._mm512_set1_epi32(p_div.2 as i32);
            let p_div3 = simd.avx512f._mm512_set1_epi32(p_div.3 as i32);

            while m < n / 16 {
                let w = &twid[w_idx..];

                for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<16, _>(z0).0;
                    let z1 = as_arrays_mut::<16, _>(z1).0;
                    let w1 = simd.avx512f._mm512_set1_epi32(w1 as _);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        let z1w = mul_avx512(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                        (z0, z1) = (add_avx512(simd, p, z0, z1w), sub_avx512(simd, p, z0, z1w));
                        *__z0 = cast(z0);
                        *__z1 = cast(z1);
                    }
                }

                t /= 2;
                m *= 2;
                w_idx *= 2;
            }

            // m = n / 16
            // t = 8
            {
                let w = as_arrays::<2, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute8_epu32(*w1);
                    let [mut z0, mut z1] =
                        simd.interleave8_epu32(cast(*z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1));
                    let z1w = mul_avx512(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                    (z0, z1) = (add_avx512(simd, p, z0, z1w), sub_avx512(simd, p, z0, z1w));
                    *z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1 = cast(simd.interleave8_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 8
            // t = 4
            {
                let w = as_arrays::<4, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute4_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    let z1w = mul_avx512(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                    (z0, z1) = (add_avx512(simd, p, z0, z1w), sub_avx512(simd, p, z0, z1w));
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 4
            // t = 2
            {
                let w = as_arrays::<8, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute2_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    let z1w = mul_avx512(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                    (z0, z1) = (add_avx512(simd, p, z0, z1w), sub_avx512(simd, p, z0, z1w));
                    *z0z0z1z1 = cast(simd.interleave2_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 2
            // t = 1
            {
                let w = as_arrays::<16, _>(&twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, w1) in zip(data, w) {
                    let w1 = simd.permute1_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    let z1w = mul_avx512(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                    (z0, z1) = (add_avx512(simd, p, z0, z1w), sub_avx512(simd, p, z0, z1w));
                    *z0z1 = cast(simd.interleave1_epu32([z0, z1]));
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn fwd_depth_first_avx512(
    simd: Avx512,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                fwd_breadth_first_avx512(
                    simd,
                    data,
                    p,
                    p_div,
                    twid,
                    recursion_depth,
                    recursion_half,
                );
            } else {
                {
                    let t = n / 2;
                    let m = 1;
                    let w_idx = (m << recursion_depth) + m * recursion_half;
                    let w = &twid[w_idx..];
                    let p = simd.avx512f._mm512_set1_epi32(p as i32);
                    let p_div0 = simd.avx512f._mm512_set1_epi32(p_div.0 as i32);
                    let p_div1 = simd.avx512f._mm512_set1_epi32(p_div.1 as i32);
                    let p_div2 = simd.avx512f._mm512_set1_epi32(p_div.2 as i32);
                    let p_div3 = simd.avx512f._mm512_set1_epi32(p_div.3 as i32);

                    for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<16, _>(z0).0;
                        let z1 = as_arrays_mut::<16, _>(z1).0;
                        let w1 = simd.avx512f._mm512_set1_epi32(w1 as _);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            let z1w = mul_avx512(simd, p, p_div0, p_div1, p_div2, p_div3, z1, w1);
                            (z0, z1) = (add_avx512(simd, p, z0, z1w), sub_avx512(simd, p, z0, z1w));
                            *__z0 = cast(z0);
                            *__z1 = cast(z1);
                        }
                    }
                }

                let (data0, data1) = data.split_at_mut(n / 2);
                fwd_depth_first_avx512(
                    simd,
                    data0,
                    p,
                    p_div,
                    twid,
                    recursion_depth + 1,
                    recursion_half * 2,
                );
                fwd_depth_first_avx512(
                    simd,
                    data1,
                    p,
                    p_div,
                    twid,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                );
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn inv_breadth_first_avx512(
    simd: Avx512,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    inv_twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    use pulp::as_arrays;

    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            let mut t = 1;
            let mut m = n;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;
            let p = simd.avx512f._mm512_set1_epi32(p as i32);
            let p_div0 = simd.avx512f._mm512_set1_epi32(p_div.0 as i32);
            let p_div1 = simd.avx512f._mm512_set1_epi32(p_div.1 as i32);
            let p_div2 = simd.avx512f._mm512_set1_epi32(p_div.2 as i32);
            let p_div3 = simd.avx512f._mm512_set1_epi32(p_div.3 as i32);

            // m = n / 2
            // t = 1
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<16, _>(&inv_twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, w1) in zip(data, w) {
                    let w1 = simd.permute1_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    (z0, z1) = (
                        add_avx512(simd, p, z0, z1),
                        mul_avx512(
                            simd,
                            p,
                            p_div0,
                            p_div1,
                            p_div2,
                            p_div3,
                            sub_avx512(simd, p, z0, z1),
                            w1,
                        ),
                    );
                    *z0z1 = cast(simd.interleave1_epu32([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 4
            // t = 2
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<8, _>(&inv_twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute2_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    (z0, z1) = (
                        add_avx512(simd, p, z0, z1),
                        mul_avx512(
                            simd,
                            p,
                            p_div0,
                            p_div1,
                            p_div2,
                            p_div3,
                            sub_avx512(simd, p, z0, z1),
                            w1,
                        ),
                    );
                    *z0z0z1z1 = cast(simd.interleave2_epu32([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 8
            // t = 4
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<4, _>(&inv_twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute4_epu32(*w1);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = (
                        add_avx512(simd, p, z0, z1),
                        mul_avx512(
                            simd,
                            p,
                            p_div0,
                            p_div1,
                            p_div2,
                            p_div3,
                            sub_avx512(simd, p, z0, z1),
                            w1,
                        ),
                    );
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu32([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 16
            // t = 8
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<2, _>(&inv_twid[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1, w1) in zip(data, w) {
                    let w1 = simd.permute8_epu32(*w1);
                    let [mut z0, mut z1] =
                        simd.interleave8_epu32(cast(*z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1));
                    (z0, z1) = (
                        add_avx512(simd, p, z0, z1),
                        mul_avx512(
                            simd,
                            p,
                            p_div0,
                            p_div1,
                            p_div2,
                            p_div3,
                            sub_avx512(simd, p, z0, z1),
                            w1,
                        ),
                    );
                    *z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1 = cast(simd.interleave8_epu32([z0, z1]));
                }

                t *= 2;
            }

            while m > 1 {
                m /= 2;
                w_idx /= 2;

                let w = &inv_twid[w_idx..];

                for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<16, _>(z0).0;
                    let z1 = as_arrays_mut::<16, _>(z1).0;
                    let w1 = simd.avx512f._mm512_set1_epi32(w1 as _);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        (z0, z1) = (
                            add_avx512(simd, p, z0, z1),
                            mul_avx512(
                                simd,
                                p,
                                p_div0,
                                p_div1,
                                p_div2,
                                p_div3,
                                sub_avx512(simd, p, z0, z1),
                                w1,
                            ),
                        );
                        *__z0 = cast(z0);
                        *__z1 = cast(z1);
                    }
                }

                t *= 2;
            }
        },
    );
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn inv_depth_first_avx512(
    simd: Avx512,
    data: &mut [u32],
    p: u32,
    p_div: (u32, u32, u32, u32),
    inv_twid: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                inv_breadth_first_avx512(
                    simd,
                    data,
                    p,
                    p_div,
                    inv_twid,
                    recursion_depth,
                    recursion_half,
                );
            } else {
                let (data0, data1) = data.split_at_mut(n / 2);
                inv_depth_first_avx512(
                    simd,
                    data0,
                    p,
                    p_div,
                    inv_twid,
                    recursion_depth + 1,
                    recursion_half * 2,
                );
                inv_depth_first_avx512(
                    simd,
                    data1,
                    p,
                    p_div,
                    inv_twid,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                );

                {
                    let t = n / 2;
                    let m = 1;
                    let w_idx = (m << recursion_depth) + m * recursion_half;
                    let w = &inv_twid[w_idx..];
                    let p = simd.avx512f._mm512_set1_epi32(p as i32);
                    let p_div0 = simd.avx512f._mm512_set1_epi32(p_div.0 as i32);
                    let p_div1 = simd.avx512f._mm512_set1_epi32(p_div.1 as i32);
                    let p_div2 = simd.avx512f._mm512_set1_epi32(p_div.2 as i32);
                    let p_div3 = simd.avx512f._mm512_set1_epi32(p_div.3 as i32);

                    for (data, &w1) in zip(data.chunks_exact_mut(2 * t), w) {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<16, _>(z0).0;
                        let z1 = as_arrays_mut::<16, _>(z1).0;
                        let w1 = simd.avx512f._mm512_set1_epi32(w1 as _);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            (z0, z1) = (
                                add_avx512(simd, p, z0, z1),
                                mul_avx512(
                                    simd,
                                    p,
                                    p_div0,
                                    p_div1,
                                    p_div2,
                                    p_div3,
                                    sub_avx512(simd, p, z0, z1),
                                    w1,
                                ),
                            );
                            *__z0 = cast(z0);
                            *__z1 = cast(z1);
                        }
                    }
                }
            }
        },
    );
}

pub fn fwd_scalar(data: &mut [u32], p: u32, p_div: Div32, twid: &[u32]) {
    fwd_depth_first_scalar(data, p, p_div, twid, 0, 0);
}
pub fn inv_scalar(data: &mut [u32], p: u32, p_div: Div32, inv_twid: &[u32]) {
    inv_depth_first_scalar(data, p, p_div, inv_twid, 0, 0);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_avx2(simd: Avx2, data: &mut [u32], p: u32, p_div: Div32, twid: &[u32]) {
    let p_div = p_div.double_reciprocal;
    let p_div = (
        p_div as u32,
        (p_div >> 32) as u32,
        (p_div >> 64) as u32,
        (p_div >> 96) as u32,
    );
    fwd_depth_first_avx2(simd, data, p, p_div, twid, 0, 0);
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn inv_avx2(simd: Avx2, data: &mut [u32], p: u32, p_div: Div32, inv_twid: &[u32]) {
    let p_div = p_div.double_reciprocal;
    let p_div = (
        p_div as u32,
        (p_div >> 32) as u32,
        (p_div >> 64) as u32,
        (p_div >> 96) as u32,
    );
    inv_depth_first_avx2(simd, data, p, p_div, inv_twid, 0, 0);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn fwd_avx512(simd: Avx512, data: &mut [u32], p: u32, p_div: Div32, twid: &[u32]) {
    let p_div = p_div.double_reciprocal;
    let p_div = (
        p_div as u32,
        (p_div >> 32) as u32,
        (p_div >> 64) as u32,
        (p_div >> 96) as u32,
    );
    fwd_depth_first_avx512(simd, data, p, p_div, twid, 0, 0);
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn inv_avx512(simd: Avx512, data: &mut [u32], p: u32, p_div: Div32, inv_twid: &[u32]) {
    let p_div = p_div.double_reciprocal;
    let p_div = (
        p_div as u32,
        (p_div >> 32) as u32,
        (p_div >> 64) as u32,
        (p_div >> 96) as u32,
    );
    inv_depth_first_avx512(simd, data, p, p_div, inv_twid, 0, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prime::largest_prime_in_arithmetic_progression64, _32::init_negacyclic_twiddles};
    use rand::random;

    #[test]
    fn test_product() {
        for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 31, 1 << 32).unwrap()
                as u32;

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
                negacyclic_convolution[i] = sub(p, full_convolution[i], full_convolution[i + n]);
            }

            let mut twid = vec![0u32; n];
            let mut inv_twid = vec![0u32; n];
            init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

            let mut prod = vec![0u32; n];
            let mut lhs_fourier = lhs.clone();
            let mut rhs_fourier = rhs.clone();

            fwd_scalar(&mut lhs_fourier, p, Div32::new(p), &twid);
            fwd_scalar(&mut rhs_fourier, p, Div32::new(p), &twid);

            for i in 0..n {
                prod[i] = mul(Div32::new(p), lhs_fourier[i], rhs_fourier[i]);
            }

            inv_scalar(&mut prod, p, Div32::new(p), &inv_twid);
            let result = prod;

            for i in 0..n {
                assert_eq!(
                    result[i],
                    mul(Div32::new(p), negacyclic_convolution[i], n as u32),
                );
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_product_avx2() {
        if let Some(simd) = Avx2::try_new() {
            for n in [32, 64, 128, 256, 512, 1024] {
                let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 31, 1 << 32)
                    .unwrap() as u32;

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

                let mut twid = vec![0u32; n];
                let mut inv_twid = vec![0u32; n];
                init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

                let mut prod = vec![0u32; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_avx2(simd, &mut lhs_fourier, p, Div32::new(p), &twid);
                fwd_avx2(simd, &mut rhs_fourier, p, Div32::new(p), &twid);

                for i in 0..n {
                    prod[i] = mul(Div32::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_avx2(simd, &mut prod, p, Div32::new(p), &inv_twid);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i],
                        mul(Div32::new(p), negacyclic_convolution[i], n as u32),
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
            for n in [32, 64, 128, 256, 512, 1024] {
                let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 31, 1 << 32)
                    .unwrap() as u32;

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

                let mut twid = vec![0u32; n];
                let mut inv_twid = vec![0u32; n];
                init_negacyclic_twiddles(p, n, &mut twid, &mut inv_twid);

                let mut prod = vec![0u32; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_avx512(simd, &mut lhs_fourier, p, Div32::new(p), &twid);
                fwd_avx512(simd, &mut rhs_fourier, p, Div32::new(p), &twid);

                for i in 0..n {
                    prod[i] = mul(Div32::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_avx512(simd, &mut prod, p, Div32::new(p), &inv_twid);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i],
                        mul(Div32::new(p), negacyclic_convolution[i], n as u32),
                    );
                }
            }
        }
    }
}
