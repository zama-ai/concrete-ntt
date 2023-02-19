#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub(crate) use crate::native32::mul_mod32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use crate::native32::mul_mod32_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub(crate) use crate::native32::{mul_mod32_avx512, mul_mod52_avx512};

pub struct Plan32(
    crate::_32::Plan,
    crate::_32::Plan,
    crate::_32::Plan,
    crate::_32::Plan,
    crate::_32::Plan,
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub struct Plan52(
    crate::_64::Plan,
    crate::_64::Plan,
    crate::_64::Plan,
    crate::Avx512,
);

#[inline(always)]
pub(crate) fn mul_mod64(p_neg: u64, a: u64, b: u64, b_shoup: u64) -> u64 {
    let q = ((a as u128 * b_shoup as u128) >> 64) as u64;
    let r = a.wrapping_mul(b).wrapping_add(p_neg.wrapping_mul(q));
    r.min(r.wrapping_add(p_neg))
}

#[inline(always)]
#[allow(dead_code)]
fn reconstruct_32bit_01234(mod_p0: u32, mod_p1: u32, mod_p2: u32, mod_p3: u32, mod_p4: u32) -> u64 {
    use crate::primes32::*;

    let v0 = mod_p0;
    let v1 = mul_mod32(P1, P0_INV_MOD_P1, 2 * P1 + mod_p1 - v0);
    let v2 = mul_mod32(
        P2,
        P01_INV_MOD_P2,
        2 * P2 + mod_p2 - (v0 + mul_mod32(P2, P0, v1)),
    );
    let v3 = mul_mod32(
        P3,
        P012_INV_MOD_P3,
        2 * P3 + mod_p3 - (v0 + mul_mod32(P3, P0, v1 + mul_mod32(P3, P1, v2))),
    );
    let v4 = mul_mod32(
        P4,
        P0123_INV_MOD_P4,
        2 * P4 + mod_p4
            - (v0 + mul_mod32(P4, P0, v1 + mul_mod32(P4, P1, v2 + mul_mod32(P4, P2, v3)))),
    );

    let sign = v4 > (P4 / 2);

    const _0: u64 = P0 as u64;
    const _01: u64 = _0.wrapping_mul(P1 as u64);
    const _012: u64 = _01.wrapping_mul(P2 as u64);
    const _0123: u64 = _012.wrapping_mul(P3 as u64);
    const _01234: u64 = _0123.wrapping_mul(P4 as u64);

    let pos = (v0 as u64)
        .wrapping_add((v1 as u64).wrapping_mul(_0))
        .wrapping_add((v2 as u64).wrapping_mul(_01))
        .wrapping_add((v3 as u64).wrapping_mul(_012))
        .wrapping_add((v4 as u64).wrapping_mul(_0123));

    let neg = pos.wrapping_sub(_01234);

    if sign {
        neg
    } else {
        pos
    }
}

#[inline(always)]
fn reconstruct_32bit_01234_v2(
    mod_p0: u32,
    mod_p1: u32,
    mod_p2: u32,
    mod_p3: u32,
    mod_p4: u32,
) -> u64 {
    use crate::primes32::*;

    let mod_p12 = {
        let v1 = mod_p1;
        let v2 = mul_mod32(P2, P1_INV_MOD_P2, 2 * P2 + mod_p2 - v1);
        v1 as u64 + (v2 as u64 * P1 as u64)
    };
    let mod_p34 = {
        let v3 = mod_p3;
        let v4 = mul_mod32(P4, P3_INV_MOD_P4, 2 * P4 + mod_p4 - v3);
        v3 as u64 + (v4 as u64 * P3 as u64)
    };

    let v0 = mod_p0 as u64;
    let v12 = mul_mod64(
        P12.wrapping_neg(),
        2 * P12 + mod_p12 - v0,
        P0_INV_MOD_P12,
        P0_INV_MOD_P12_SHOUP,
    );
    let v34 = mul_mod64(
        P34.wrapping_neg(),
        2 * P34 + mod_p34 - (v0 + mul_mod64(P34.wrapping_neg(), v12, P0 as u64, P0_MOD_P34_SHOUP)),
        P012_INV_MOD_P34,
        P012_INV_MOD_P34_SHOUP,
    );

    let sign = v34 > (P34 / 2);

    const _0: u64 = P0 as u64;
    const _012: u64 = _0.wrapping_mul(P12);
    const _01234: u64 = _012.wrapping_mul(P34);

    let pos = v0
        .wrapping_add(v12.wrapping_mul(_0))
        .wrapping_add(v34.wrapping_mul(_012));
    let neg = pos.wrapping_sub(_01234);

    if sign {
        neg
    } else {
        pos
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn mul_mod32_v2_avx2(
    simd: crate::Avx2,
    p: __m256i,
    a: __m256i,
    b: __m256i,
    b_shoup: __m256i,
) -> __m256i {
    let shoup_q = simd
        .avx2
        ._mm256_srli_epi64::<32>(simd.avx2._mm256_mul_epu32(a, b_shoup));
    let t = simd.avx2._mm256_and_si256(
        simd.avx._mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0),
        simd.avx2._mm256_sub_epi32(
            simd.avx2._mm256_mul_epu32(a, b),
            simd.avx2._mm256_mul_epu32(shoup_q, p),
        ),
    );
    simd.small_mod_epu32(p, t)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub(crate) fn mul_mod32_v2_avx512(
    simd: crate::Avx512,
    p: __m512i,
    a: __m512i,
    b: __m512i,
    b_shoup: __m512i,
) -> __m512i {
    let avx = simd.avx512f;
    let shoup_q = avx._mm512_srli_epi64::<32>(avx._mm512_mul_epu32(a, b_shoup));
    let t = avx._mm512_and_si512(
        avx._mm512_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0),
        avx._mm512_sub_epi32(avx._mm512_mul_epu32(a, b), avx._mm512_mul_epu32(shoup_q, p)),
    );
    simd.small_mod_epu32(p, t)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn mul_mod64_avx2(
    simd: crate::Avx2,
    p: __m256i,
    a: __m256i,
    b: __m256i,
    b_shoup: __m256i,
) -> __m256i {
    let q = simd._mm256_mul_u64_u64_epu64(a, b_shoup).1;
    let r = simd.avx2._mm256_sub_epi64(
        simd._mm256_mul_u64_u64_epu64(a, b).0,
        simd._mm256_mul_u64_u64_epu64(p, q).0,
    );
    simd.small_mod_epu64(p, r)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub(crate) fn mul_mod64_avx512(
    simd: crate::Avx512,
    p: __m512i,
    a: __m512i,
    b: __m512i,
    b_shoup: __m512i,
) -> __m512i {
    let avx = simd.avx512f;
    let q = simd._mm512_mul_u64_u64_epu64(a, b_shoup).1;
    let r = avx._mm512_sub_epi64(avx._mm512_mullox_epi64(a, b), avx._mm512_mullox_epi64(p, q));
    simd.small_mod_epu64(p, r)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn reconstruct_32bit_01234_v2_avx2(
    simd: crate::Avx2,
    mod_p0: __m128i,
    mod_p1: __m128i,
    mod_p2: __m128i,
    mod_p3: __m128i,
    mod_p4: __m128i,
) -> __m256i {
    use crate::primes32::*;

    let avx = simd.avx2;

    let p0 = simd.avx._mm256_set1_epi64x(P0 as i64);
    let p1 = simd.avx._mm256_set1_epi64x(P1 as i64);
    let p2 = simd.avx._mm256_set1_epi64x(P2 as i64);
    let p3 = simd.avx._mm256_set1_epi64x(P3 as i64);
    let p4 = simd.avx._mm256_set1_epi64x(P4 as i64);
    let p12 = simd.avx._mm256_set1_epi64x(P12 as i64);
    let p34 = simd.avx._mm256_set1_epi64x(P34 as i64);
    let p012 = simd
        .avx
        ._mm256_set1_epi64x((P0 as u64).wrapping_mul(P12) as i64);
    let p01234 = simd
        .avx
        ._mm256_set1_epi64x((P0 as u64).wrapping_mul(P12).wrapping_mul(P34) as i64);

    let two_p2 = simd.avx._mm256_set1_epi64x((2 * P2) as i64);
    let two_p4 = simd.avx._mm256_set1_epi64x((2 * P4) as i64);
    let two_p12 = simd.avx._mm256_set1_epi64x((2 * P12) as i64);
    let two_p34 = simd.avx._mm256_set1_epi64x((2 * P34) as i64);
    let half_p34 = simd.avx._mm256_set1_epi64x((P34 / 2) as i64);

    let p0_inv_mod_p12 = simd.avx._mm256_set1_epi64x(P0_INV_MOD_P12 as i64);
    let p0_inv_mod_p12_shoup = simd.avx._mm256_set1_epi64x(P0_INV_MOD_P12_SHOUP as i64);
    let p1_inv_mod_p2 = simd.avx._mm256_set1_epi64x(P1_INV_MOD_P2 as i64);
    let p1_inv_mod_p2_shoup = simd.avx._mm256_set1_epi64x(P1_INV_MOD_P2_SHOUP as i64);
    let p3_inv_mod_p4 = simd.avx._mm256_set1_epi64x(P3_INV_MOD_P4 as i64);
    let p3_inv_mod_p4_shoup = simd.avx._mm256_set1_epi64x(P3_INV_MOD_P4_SHOUP as i64);

    let p012_inv_mod_p34 = simd.avx._mm256_set1_epi64x(P012_INV_MOD_P34 as i64);
    let p012_inv_mod_p34_shoup = simd.avx._mm256_set1_epi64x(P012_INV_MOD_P34_SHOUP as i64);
    let p0_mod_p34_shoup = simd.avx._mm256_set1_epi64x(P0_MOD_P34_SHOUP as i64);

    let mod_p0 = avx._mm256_cvtepu32_epi64(mod_p0);
    let mod_p1 = avx._mm256_cvtepu32_epi64(mod_p1);
    let mod_p2 = avx._mm256_cvtepu32_epi64(mod_p2);
    let mod_p3 = avx._mm256_cvtepu32_epi64(mod_p3);
    let mod_p4 = avx._mm256_cvtepu32_epi64(mod_p4);

    let mod_p12 = {
        let v1 = mod_p1;
        let v2 = mul_mod32_v2_avx2(
            simd,
            p2,
            avx._mm256_sub_epi32(avx._mm256_add_epi32(two_p2, mod_p2), v1),
            p1_inv_mod_p2,
            p1_inv_mod_p2_shoup,
        );
        avx._mm256_add_epi64(v1, simd._mm256_mullo_u64_u32_epu64(v2, p1))
    };
    let mod_p34 = {
        let v3 = mod_p3;
        let v4 = mul_mod32_v2_avx2(
            simd,
            p4,
            avx._mm256_sub_epi32(avx._mm256_add_epi32(two_p4, mod_p4), v3),
            p3_inv_mod_p4,
            p3_inv_mod_p4_shoup,
        );
        avx._mm256_add_epi64(v3, simd._mm256_mullo_u64_u32_epu64(v4, p3))
    };

    let v0 = mod_p0;
    let v12 = mul_mod64_avx2(
        simd,
        p12,
        avx._mm256_sub_epi64(avx._mm256_add_epi64(two_p12, mod_p12), v0),
        p0_inv_mod_p12,
        p0_inv_mod_p12_shoup,
    );
    let v34 = mul_mod64_avx2(
        simd,
        p34,
        avx._mm256_sub_epi64(
            avx._mm256_add_epi64(two_p34, mod_p34),
            avx._mm256_add_epi64(v0, mul_mod64_avx2(simd, p34, v12, p0, p0_mod_p34_shoup)),
        ),
        p012_inv_mod_p34,
        p012_inv_mod_p34_shoup,
    );

    let sign = simd._mm256_cmpgt_epu64(v34, half_p34);
    let pos = v0;
    let pos = avx._mm256_add_epi64(pos, simd._mm256_mullo_u64_u32_epu64(v12, p0));
    let pos = avx._mm256_add_epi64(pos, simd._mm256_mul_u64_u64_epu64(v34, p012).0);
    let neg = avx._mm256_sub_epi64(pos, p01234);
    avx._mm256_blendv_epi8(pos, neg, sign)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(dead_code)]
#[inline(always)]
fn reconstruct_32bit_01234_avx2(
    simd: crate::Avx2,
    mod_p0: __m256i,
    mod_p1: __m256i,
    mod_p2: __m256i,
    mod_p3: __m256i,
    mod_p4: __m256i,
) -> [__m256i; 2] {
    use crate::primes32::*;

    let avx = simd.avx2;

    let p0 = simd.avx._mm256_set1_epi32(P0 as i32);
    let p1 = simd.avx._mm256_set1_epi32(P1 as i32);
    let p2 = simd.avx._mm256_set1_epi32(P2 as i32);
    let p3 = simd.avx._mm256_set1_epi32(P3 as i32);
    let p4 = simd.avx._mm256_set1_epi32(P4 as i32);
    let two_p1 = simd.avx._mm256_set1_epi32((2 * P1) as i32);
    let two_p2 = simd.avx._mm256_set1_epi32((2 * P2) as i32);
    let two_p3 = simd.avx._mm256_set1_epi32((2 * P3) as i32);
    let two_p4 = simd.avx._mm256_set1_epi32((2 * P4) as i32);
    let half_p4 = simd.avx._mm256_set1_epi32((P4 / 2) as i32);

    let p0_inv_mod_p1 = simd.avx._mm256_set1_epi32(P0_INV_MOD_P1 as i32);
    let p0_inv_mod_p1_shoup = simd.avx._mm256_set1_epi32(P0_INV_MOD_P1_SHOUP as i32);
    let p0_mod_p2_shoup = simd.avx._mm256_set1_epi32(P0_MOD_P2_SHOUP as i32);
    let p0_mod_p3_shoup = simd.avx._mm256_set1_epi32(P0_MOD_P3_SHOUP as i32);
    let p1_mod_p3_shoup = simd.avx._mm256_set1_epi32(P1_MOD_P3_SHOUP as i32);
    let p0_mod_p4_shoup = simd.avx._mm256_set1_epi32(P0_MOD_P4_SHOUP as i32);
    let p1_mod_p4_shoup = simd.avx._mm256_set1_epi32(P1_MOD_P4_SHOUP as i32);
    let p2_mod_p4_shoup = simd.avx._mm256_set1_epi32(P2_MOD_P4_SHOUP as i32);

    let p01_inv_mod_p2 = simd.avx._mm256_set1_epi32(P01_INV_MOD_P2 as i32);
    let p01_inv_mod_p2_shoup = simd.avx._mm256_set1_epi32(P01_INV_MOD_P2_SHOUP as i32);
    let p012_inv_mod_p3 = simd.avx._mm256_set1_epi32(P012_INV_MOD_P3 as i32);
    let p012_inv_mod_p3_shoup = simd.avx._mm256_set1_epi32(P012_INV_MOD_P3_SHOUP as i32);
    let p0123_inv_mod_p4 = simd.avx._mm256_set1_epi32(P0123_INV_MOD_P4 as i32);
    let p0123_inv_mod_p4_shoup = simd.avx._mm256_set1_epi32(P0123_INV_MOD_P4_SHOUP as i32);

    let p01 = simd
        .avx
        ._mm256_set1_epi64x((P0 as u64).wrapping_mul(P1 as u64) as i64);
    let p012 = simd
        .avx
        ._mm256_set1_epi64x((P0 as u64).wrapping_mul(P1 as u64).wrapping_mul(P2 as u64) as i64);
    let p0123 = simd.avx._mm256_set1_epi64x(
        (P0 as u64)
            .wrapping_mul(P1 as u64)
            .wrapping_mul(P2 as u64)
            .wrapping_mul(P3 as u64) as i64,
    );
    let p01234 = simd.avx._mm256_set1_epi64x(
        (P0 as u64)
            .wrapping_mul(P1 as u64)
            .wrapping_mul(P2 as u64)
            .wrapping_mul(P3 as u64)
            .wrapping_mul(P4 as u64) as i64,
    );

    let v0 = mod_p0;
    let v1 = mul_mod32_avx2(
        simd,
        p1,
        avx._mm256_sub_epi32(avx._mm256_add_epi32(two_p1, mod_p1), v0),
        p0_inv_mod_p1,
        p0_inv_mod_p1_shoup,
    );
    let v2 = mul_mod32_avx2(
        simd,
        p2,
        avx._mm256_sub_epi32(
            avx._mm256_add_epi32(two_p2, mod_p2),
            avx._mm256_add_epi32(v0, mul_mod32_avx2(simd, p2, v1, p0, p0_mod_p2_shoup)),
        ),
        p01_inv_mod_p2,
        p01_inv_mod_p2_shoup,
    );
    let v3 = mul_mod32_avx2(
        simd,
        p3,
        avx._mm256_sub_epi32(
            avx._mm256_add_epi32(two_p3, mod_p3),
            avx._mm256_add_epi32(
                v0,
                mul_mod32_avx2(
                    simd,
                    p3,
                    avx._mm256_add_epi32(v1, mul_mod32_avx2(simd, p3, v2, p1, p1_mod_p3_shoup)),
                    p0,
                    p0_mod_p3_shoup,
                ),
            ),
        ),
        p012_inv_mod_p3,
        p012_inv_mod_p3_shoup,
    );
    let v4 = mul_mod32_avx2(
        simd,
        p4,
        avx._mm256_sub_epi32(
            avx._mm256_add_epi32(two_p4, mod_p4),
            avx._mm256_add_epi32(
                v0,
                mul_mod32_avx2(
                    simd,
                    p4,
                    avx._mm256_add_epi32(
                        v1,
                        mul_mod32_avx2(
                            simd,
                            p4,
                            avx._mm256_add_epi32(
                                v2,
                                mul_mod32_avx2(simd, p4, v3, p2, p2_mod_p4_shoup),
                            ),
                            p1,
                            p1_mod_p4_shoup,
                        ),
                    ),
                    p0,
                    p0_mod_p4_shoup,
                ),
            ),
        ),
        p0123_inv_mod_p4,
        p0123_inv_mod_p4_shoup,
    );

    let sign = simd._mm256_cmpgt_epu32(v4, half_p4);
    let sign0 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(sign));
    let sign1 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(sign));

    let v00 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v0));
    let v01 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v0));
    let v10 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v1));
    let v11 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v1));
    let v20 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v2));
    let v21 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v2));
    let v30 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v3));
    let v31 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v3));
    let v40 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v4));
    let v41 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v4));

    let pos0 = v00;
    let pos0 = avx._mm256_add_epi64(pos0, avx._mm256_mul_epi32(p0, v10));
    let pos0 = avx._mm256_add_epi64(pos0, simd._mm256_mullo_u64_u32_epu64(p01, v20));
    let pos0 = avx._mm256_add_epi64(pos0, simd._mm256_mullo_u64_u32_epu64(p012, v30));
    let pos0 = avx._mm256_add_epi64(pos0, simd._mm256_mullo_u64_u32_epu64(p0123, v40));

    let pos1 = v01;
    let pos1 = avx._mm256_add_epi64(pos1, avx._mm256_mul_epi32(p0, v11));
    let pos1 = avx._mm256_add_epi64(pos1, simd._mm256_mullo_u64_u32_epu64(p01, v21));
    let pos1 = avx._mm256_add_epi64(pos1, simd._mm256_mullo_u64_u32_epu64(p012, v31));
    let pos1 = avx._mm256_add_epi64(pos1, simd._mm256_mullo_u64_u32_epu64(p0123, v41));

    let neg0 = avx._mm256_sub_epi64(pos0, p01234);
    let neg1 = avx._mm256_sub_epi64(pos1, p01234);

    [
        avx._mm256_blendv_epi8(pos0, neg0, sign0),
        avx._mm256_blendv_epi8(pos1, neg1, sign1),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[allow(dead_code)]
#[inline(always)]
fn reconstruct_32bit_01234_avx512(
    simd: crate::Avx512,
    mod_p0: __m512i,
    mod_p1: __m512i,
    mod_p2: __m512i,
    mod_p3: __m512i,
    mod_p4: __m512i,
) -> [__m512i; 2] {
    use crate::primes32::*;

    let avx = simd.avx512f;

    let p0 = avx._mm512_set1_epi32(P0 as i32);
    let p1 = avx._mm512_set1_epi32(P1 as i32);
    let p2 = avx._mm512_set1_epi32(P2 as i32);
    let p3 = avx._mm512_set1_epi32(P3 as i32);
    let p4 = avx._mm512_set1_epi32(P4 as i32);
    let two_p1 = avx._mm512_set1_epi32((2 * P1) as i32);
    let two_p2 = avx._mm512_set1_epi32((2 * P2) as i32);
    let two_p3 = avx._mm512_set1_epi32((2 * P3) as i32);
    let two_p4 = avx._mm512_set1_epi32((2 * P4) as i32);
    let half_p4 = avx._mm512_set1_epi32((P4 / 2) as i32);

    let p0_inv_mod_p1 = avx._mm512_set1_epi32(P0_INV_MOD_P1 as i32);
    let p0_inv_mod_p1_shoup = avx._mm512_set1_epi32(P0_INV_MOD_P1_SHOUP as i32);
    let p0_mod_p2_shoup = avx._mm512_set1_epi32(P0_MOD_P2_SHOUP as i32);
    let p0_mod_p3_shoup = avx._mm512_set1_epi32(P0_MOD_P3_SHOUP as i32);
    let p1_mod_p3_shoup = avx._mm512_set1_epi32(P1_MOD_P3_SHOUP as i32);
    let p0_mod_p4_shoup = avx._mm512_set1_epi32(P0_MOD_P4_SHOUP as i32);
    let p1_mod_p4_shoup = avx._mm512_set1_epi32(P1_MOD_P4_SHOUP as i32);
    let p2_mod_p4_shoup = avx._mm512_set1_epi32(P2_MOD_P4_SHOUP as i32);

    let p01_inv_mod_p2 = avx._mm512_set1_epi32(P01_INV_MOD_P2 as i32);
    let p01_inv_mod_p2_shoup = avx._mm512_set1_epi32(P01_INV_MOD_P2_SHOUP as i32);
    let p012_inv_mod_p3 = avx._mm512_set1_epi32(P012_INV_MOD_P3 as i32);
    let p012_inv_mod_p3_shoup = avx._mm512_set1_epi32(P012_INV_MOD_P3_SHOUP as i32);
    let p0123_inv_mod_p4 = avx._mm512_set1_epi32(P0123_INV_MOD_P4 as i32);
    let p0123_inv_mod_p4_shoup = avx._mm512_set1_epi32(P0123_INV_MOD_P4_SHOUP as i32);

    let p01 = avx._mm512_set1_epi64((P0 as u64).wrapping_mul(P1 as u64) as i64);
    let p012 =
        avx._mm512_set1_epi64((P0 as u64).wrapping_mul(P1 as u64).wrapping_mul(P2 as u64) as i64);
    let p0123 = avx._mm512_set1_epi64(
        (P0 as u64)
            .wrapping_mul(P1 as u64)
            .wrapping_mul(P2 as u64)
            .wrapping_mul(P3 as u64) as i64,
    );
    let p01234 = avx._mm512_set1_epi64(
        (P0 as u64)
            .wrapping_mul(P1 as u64)
            .wrapping_mul(P2 as u64)
            .wrapping_mul(P3 as u64)
            .wrapping_mul(P4 as u64) as i64,
    );

    let v0 = mod_p0;
    let v1 = mul_mod32_avx512(
        simd,
        p1,
        avx._mm512_sub_epi32(avx._mm512_add_epi32(two_p1, mod_p1), v0),
        p0_inv_mod_p1,
        p0_inv_mod_p1_shoup,
    );
    let v2 = mul_mod32_avx512(
        simd,
        p2,
        avx._mm512_sub_epi32(
            avx._mm512_add_epi32(two_p2, mod_p2),
            avx._mm512_add_epi32(v0, mul_mod32_avx512(simd, p2, v1, p0, p0_mod_p2_shoup)),
        ),
        p01_inv_mod_p2,
        p01_inv_mod_p2_shoup,
    );
    let v3 = mul_mod32_avx512(
        simd,
        p3,
        avx._mm512_sub_epi32(
            avx._mm512_add_epi32(two_p3, mod_p3),
            avx._mm512_add_epi32(
                v0,
                mul_mod32_avx512(
                    simd,
                    p3,
                    avx._mm512_add_epi32(v1, mul_mod32_avx512(simd, p3, v2, p1, p1_mod_p3_shoup)),
                    p0,
                    p0_mod_p3_shoup,
                ),
            ),
        ),
        p012_inv_mod_p3,
        p012_inv_mod_p3_shoup,
    );
    let v4 = mul_mod32_avx512(
        simd,
        p4,
        avx._mm512_sub_epi32(
            avx._mm512_add_epi32(two_p4, mod_p4),
            avx._mm512_add_epi32(
                v0,
                mul_mod32_avx512(
                    simd,
                    p4,
                    avx._mm512_add_epi32(
                        v1,
                        mul_mod32_avx512(
                            simd,
                            p4,
                            avx._mm512_add_epi32(
                                v2,
                                mul_mod32_avx512(simd, p4, v3, p2, p2_mod_p4_shoup),
                            ),
                            p1,
                            p1_mod_p4_shoup,
                        ),
                    ),
                    p0,
                    p0_mod_p4_shoup,
                ),
            ),
        ),
        p0123_inv_mod_p4,
        p0123_inv_mod_p4_shoup,
    );

    let sign = avx._mm512_cmpgt_epu32_mask(v4, half_p4);
    let sign0 = sign as u8;
    let sign1 = (sign >> 8) as u8;
    let v00 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v0));
    let v01 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v0));
    let v10 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v1));
    let v11 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v1));
    let v20 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v2));
    let v21 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v2));
    let v30 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v3));
    let v31 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v3));
    let v40 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v4));
    let v41 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v4));

    let pos0 = v00;
    let pos0 = avx._mm512_add_epi64(pos0, avx._mm512_mul_epi32(p0, v10));
    let pos0 = avx._mm512_add_epi64(pos0, avx._mm512_mullox_epi64(p01, v20));
    let pos0 = avx._mm512_add_epi64(pos0, avx._mm512_mullox_epi64(p012, v30));
    let pos0 = avx._mm512_add_epi64(pos0, avx._mm512_mullox_epi64(p0123, v40));

    let pos1 = v01;
    let pos1 = avx._mm512_add_epi64(pos1, avx._mm512_mul_epi32(p0, v11));
    let pos1 = avx._mm512_add_epi64(pos1, avx._mm512_mullox_epi64(p01, v21));
    let pos1 = avx._mm512_add_epi64(pos1, avx._mm512_mullox_epi64(p012, v31));
    let pos1 = avx._mm512_add_epi64(pos1, avx._mm512_mullox_epi64(p0123, v41));

    let neg0 = avx._mm512_sub_epi64(pos0, p01234);
    let neg1 = avx._mm512_sub_epi64(pos1, p01234);

    [
        avx._mm512_mask_blend_epi64(sign0, pos0, neg0),
        avx._mm512_mask_blend_epi64(sign1, pos1, neg1),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn reconstruct_32bit_01234_v2_avx512(
    simd: crate::Avx512,
    mod_p0: __m256i,
    mod_p1: __m256i,
    mod_p2: __m256i,
    mod_p3: __m256i,
    mod_p4: __m256i,
) -> __m512i {
    use crate::primes32::*;

    let avx = simd.avx512f;

    let p0 = avx._mm512_set1_epi64(P0 as i64);
    let p1 = avx._mm512_set1_epi64(P1 as i64);
    let p2 = avx._mm512_set1_epi64(P2 as i64);
    let p3 = avx._mm512_set1_epi64(P3 as i64);
    let p4 = avx._mm512_set1_epi64(P4 as i64);
    let p12 = avx._mm512_set1_epi64(P12 as i64);
    let p34 = avx._mm512_set1_epi64(P34 as i64);
    let p012 = avx._mm512_set1_epi64((P0 as u64).wrapping_mul(P12) as i64);
    let p01234 = avx._mm512_set1_epi64((P0 as u64).wrapping_mul(P12).wrapping_mul(P34) as i64);

    let two_p2 = avx._mm512_set1_epi64((2 * P2) as i64);
    let two_p4 = avx._mm512_set1_epi64((2 * P4) as i64);
    let two_p12 = avx._mm512_set1_epi64((2 * P12) as i64);
    let two_p34 = avx._mm512_set1_epi64((2 * P34) as i64);
    let half_p34 = avx._mm512_set1_epi64((P34 / 2) as i64);

    let p0_inv_mod_p12 = avx._mm512_set1_epi64(P0_INV_MOD_P12 as i64);
    let p0_inv_mod_p12_shoup = avx._mm512_set1_epi64(P0_INV_MOD_P12_SHOUP as i64);
    let p1_inv_mod_p2 = avx._mm512_set1_epi64(P1_INV_MOD_P2 as i64);
    let p1_inv_mod_p2_shoup = avx._mm512_set1_epi64(P1_INV_MOD_P2_SHOUP as i64);
    let p3_inv_mod_p4 = avx._mm512_set1_epi64(P3_INV_MOD_P4 as i64);
    let p3_inv_mod_p4_shoup = avx._mm512_set1_epi64(P3_INV_MOD_P4_SHOUP as i64);

    let p012_inv_mod_p34 = avx._mm512_set1_epi64(P012_INV_MOD_P34 as i64);
    let p012_inv_mod_p34_shoup = avx._mm512_set1_epi64(P012_INV_MOD_P34_SHOUP as i64);
    let p0_mod_p34_shoup = avx._mm512_set1_epi64(P0_MOD_P34_SHOUP as i64);

    let mod_p0 = avx._mm512_cvtepu32_epi64(mod_p0);
    let mod_p1 = avx._mm512_cvtepu32_epi64(mod_p1);
    let mod_p2 = avx._mm512_cvtepu32_epi64(mod_p2);
    let mod_p3 = avx._mm512_cvtepu32_epi64(mod_p3);
    let mod_p4 = avx._mm512_cvtepu32_epi64(mod_p4);

    let mod_p12 = {
        let v1 = mod_p1;
        let v2 = mul_mod32_v2_avx512(
            simd,
            p2,
            avx._mm512_sub_epi32(avx._mm512_add_epi32(two_p2, mod_p2), v1),
            p1_inv_mod_p2,
            p1_inv_mod_p2_shoup,
        );
        avx._mm512_add_epi64(v1, avx._mm512_mullox_epi64(v2, p1))
    };
    let mod_p34 = {
        let v3 = mod_p3;
        let v4 = mul_mod32_v2_avx512(
            simd,
            p4,
            avx._mm512_sub_epi32(avx._mm512_add_epi32(two_p4, mod_p4), v3),
            p3_inv_mod_p4,
            p3_inv_mod_p4_shoup,
        );
        avx._mm512_add_epi64(v3, avx._mm512_mullox_epi64(v4, p3))
    };

    let v0 = mod_p0;
    let v12 = mul_mod64_avx512(
        simd,
        p12,
        avx._mm512_sub_epi64(avx._mm512_add_epi64(two_p12, mod_p12), v0),
        p0_inv_mod_p12,
        p0_inv_mod_p12_shoup,
    );
    let v34 = mul_mod64_avx512(
        simd,
        p34,
        avx._mm512_sub_epi64(
            avx._mm512_add_epi64(two_p34, mod_p34),
            avx._mm512_add_epi64(v0, mul_mod64_avx512(simd, p34, v12, p0, p0_mod_p34_shoup)),
        ),
        p012_inv_mod_p34,
        p012_inv_mod_p34_shoup,
    );

    let sign = avx._mm512_cmpgt_epu64_mask(v34, half_p34);
    let pos = v0;
    let pos = avx._mm512_add_epi64(pos, avx._mm512_mullox_epi64(v12, p0));
    let pos = avx._mm512_add_epi64(pos, simd._mm512_mul_u64_u64_epu64(v34, p012).0);

    let neg = avx._mm512_sub_epi64(pos, p01234);

    avx._mm512_mask_blend_epi64(sign, pos, neg)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn reconstruct_52bit_012_avx512(
    simd: crate::Avx512,
    mod_p0: __m512i,
    mod_p1: __m512i,
    mod_p2: __m512i,
) -> __m512i {
    use crate::primes52::*;

    let avx = simd.avx512f;

    let p0 = avx._mm512_set1_epi64(P0 as i64);
    let p1 = avx._mm512_set1_epi64(P1 as i64);
    let p2 = avx._mm512_set1_epi64(P2 as i64);
    let neg_p1 = avx._mm512_set1_epi64(P1.wrapping_neg() as i64);
    let neg_p2 = avx._mm512_set1_epi64(P2.wrapping_neg() as i64);
    let two_p1 = avx._mm512_set1_epi64((2 * P1) as i64);
    let two_p2 = avx._mm512_set1_epi64((2 * P2) as i64);
    let half_p2 = avx._mm512_set1_epi64((P2 / 2) as i64);

    let p0_inv_mod_p1 = avx._mm512_set1_epi64(P0_INV_MOD_P1 as i64);
    let p0_inv_mod_p1_shoup = avx._mm512_set1_epi64(P0_INV_MOD_P1_SHOUP as i64);
    let p0_mod_p2_shoup = avx._mm512_set1_epi64(P0_MOD_P2_SHOUP as i64);
    let p01_inv_mod_p2 = avx._mm512_set1_epi64(P01_INV_MOD_P2 as i64);
    let p01_inv_mod_p2_shoup = avx._mm512_set1_epi64(P01_INV_MOD_P2_SHOUP as i64);

    let p01 = avx._mm512_set1_epi64(P0.wrapping_mul(P1) as i64);
    let p012 = avx._mm512_set1_epi64(P0.wrapping_mul(P1).wrapping_mul(P2) as i64);

    let v0 = mod_p0;
    let v1 = mul_mod52_avx512(
        simd,
        p1,
        neg_p1,
        avx._mm512_sub_epi64(avx._mm512_add_epi64(two_p1, mod_p1), v0),
        p0_inv_mod_p1,
        p0_inv_mod_p1_shoup,
    );
    let v2 = mul_mod52_avx512(
        simd,
        p2,
        neg_p2,
        avx._mm512_sub_epi64(
            avx._mm512_add_epi64(two_p2, mod_p2),
            avx._mm512_add_epi64(
                v0,
                mul_mod52_avx512(simd, p2, neg_p2, v1, p0, p0_mod_p2_shoup),
            ),
        ),
        p01_inv_mod_p2,
        p01_inv_mod_p2_shoup,
    );

    let sign = avx._mm512_cmpgt_epu64_mask(v2, half_p2);

    let pos = avx._mm512_add_epi64(
        avx._mm512_add_epi64(v0, avx._mm512_mullox_epi64(v1, p0)),
        avx._mm512_mullox_epi64(v2, p01),
    );
    let neg = avx._mm512_sub_epi64(pos, p012);

    avx._mm512_mask_blend_epi64(sign, pos, neg)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn reconstruct_slice_32bit_01234_avx2(
    simd: crate::Avx2,
    value: &mut [u64],
    mod_p0: &[u32],
    mod_p1: &[u32],
    mod_p2: &[u32],
    mod_p3: &[u32],
    mod_p4: &[u32],
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<4, _>(value).0;
            let mod_p0 = pulp::as_arrays::<4, _>(mod_p0).0;
            let mod_p1 = pulp::as_arrays::<4, _>(mod_p1).0;
            let mod_p2 = pulp::as_arrays::<4, _>(mod_p2).0;
            let mod_p3 = pulp::as_arrays::<4, _>(mod_p3).0;
            let mod_p4 = pulp::as_arrays::<4, _>(mod_p4).0;
            for (value, &mod_p0, &mod_p1, &mod_p2, &mod_p3, &mod_p4) in
                crate::izip!(value, mod_p0, mod_p1, mod_p2, mod_p3, mod_p4)
            {
                use pulp::cast;
                *value = cast(reconstruct_32bit_01234_v2_avx2(
                    simd,
                    cast(mod_p0),
                    cast(mod_p1),
                    cast(mod_p2),
                    cast(mod_p3),
                    cast(mod_p4),
                ));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
fn reconstruct_slice_32bit_01234_avx512(
    simd: crate::Avx512,
    value: &mut [u64],
    mod_p0: &[u32],
    mod_p1: &[u32],
    mod_p2: &[u32],
    mod_p3: &[u32],
    mod_p4: &[u32],
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<8, _>(value).0;
            let mod_p0 = pulp::as_arrays::<8, _>(mod_p0).0;
            let mod_p1 = pulp::as_arrays::<8, _>(mod_p1).0;
            let mod_p2 = pulp::as_arrays::<8, _>(mod_p2).0;
            let mod_p3 = pulp::as_arrays::<8, _>(mod_p3).0;
            let mod_p4 = pulp::as_arrays::<8, _>(mod_p4).0;
            for (value, &mod_p0, &mod_p1, &mod_p2, &mod_p3, &mod_p4) in
                crate::izip!(value, mod_p0, mod_p1, mod_p2, mod_p3, mod_p4)
            {
                use pulp::cast;
                *value = cast(reconstruct_32bit_01234_v2_avx512(
                    simd,
                    cast(mod_p0),
                    cast(mod_p1),
                    cast(mod_p2),
                    cast(mod_p3),
                    cast(mod_p4),
                ));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
fn reconstruct_slice_52bit_012_avx512(
    simd: crate::Avx512,
    value: &mut [u64],
    mod_p0: &[u64],
    mod_p1: &[u64],
    mod_p2: &[u64],
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<8, _>(value).0;
            let mod_p0 = pulp::as_arrays::<8, _>(mod_p0).0;
            let mod_p1 = pulp::as_arrays::<8, _>(mod_p1).0;
            let mod_p2 = pulp::as_arrays::<8, _>(mod_p2).0;
            for (value, &mod_p0, &mod_p1, &mod_p2) in crate::izip!(value, mod_p0, mod_p1, mod_p2) {
                use pulp::cast;
                *value = cast(reconstruct_52bit_012_avx512(
                    simd,
                    cast(mod_p0),
                    cast(mod_p1),
                    cast(mod_p2),
                ));
            }
        },
    );
}

impl Plan32 {
    pub fn try_new(n: usize) -> Option<Self> {
        use crate::{primes32::*, _32::Plan};
        Some(Self(
            Plan::try_new(n, P0)?,
            Plan::try_new(n, P1)?,
            Plan::try_new(n, P2)?,
            Plan::try_new(n, P3)?,
            Plan::try_new(n, P4)?,
        ))
    }

    pub fn fwd(
        &self,
        value: &[u64],
        mod_p0: &mut [u32],
        mod_p1: &mut [u32],
        mod_p2: &mut [u32],
        mod_p3: &mut [u32],
        mod_p4: &mut [u32],
    ) {
        for (value, mod_p0, mod_p1, mod_p2, mod_p3, mod_p4) in crate::izip!(
            value,
            &mut *mod_p0,
            &mut *mod_p1,
            &mut *mod_p2,
            &mut *mod_p3,
            &mut *mod_p4
        ) {
            *mod_p0 = (value % crate::primes32::P0 as u64) as u32;
            *mod_p1 = (value % crate::primes32::P1 as u64) as u32;
            *mod_p2 = (value % crate::primes32::P2 as u64) as u32;
            *mod_p3 = (value % crate::primes32::P3 as u64) as u32;
            *mod_p4 = (value % crate::primes32::P4 as u64) as u32;
        }
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
        self.2.fwd(mod_p2);
        self.3.fwd(mod_p3);
        self.4.fwd(mod_p4);
    }

    pub fn inv(
        &self,
        value: &mut [u64],
        mod_p0: &mut [u32],
        mod_p1: &mut [u32],
        mod_p2: &mut [u32],
        mod_p3: &mut [u32],
        mod_p4: &mut [u32],
    ) {
        self.0.inv(mod_p0);
        self.1.inv(mod_p1);
        self.2.inv(mod_p2);
        self.3.inv(mod_p3);
        self.4.inv(mod_p4);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                reconstruct_slice_32bit_01234_avx512(
                    simd, value, mod_p0, mod_p1, mod_p2, mod_p3, mod_p4,
                );
                return;
            }
            if let Some(simd) = crate::Avx2::try_new() {
                reconstruct_slice_32bit_01234_avx2(
                    simd, value, mod_p0, mod_p1, mod_p2, mod_p3, mod_p4,
                );
                return;
            }
        }

        for (value, &mod_p0, &mod_p1, &mod_p2, &mod_p3, &mod_p4) in
            crate::izip!(value, &*mod_p0, &*mod_p1, &*mod_p2, &*mod_p3, &*mod_p4)
        {
            *value = reconstruct_32bit_01234_v2(mod_p0, mod_p1, mod_p2, mod_p3, mod_p4);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl Plan52 {
    pub fn try_new(n: usize) -> Option<Self> {
        use crate::{primes52::*, _64::Plan};
        let simd = crate::Avx512::try_new()?;
        Some(Self(
            Plan::try_new(n, P0)?,
            Plan::try_new(n, P1)?,
            Plan::try_new(n, P2)?,
            simd,
        ))
    }

    pub fn fwd(&self, value: &[u64], mod_p0: &mut [u64], mod_p1: &mut [u64], mod_p2: &mut [u64]) {
        use crate::primes52::*;
        self.3.vectorize(
            #[inline(always)]
            || {
                for (&value, mod_p0, mod_p1, mod_p2) in
                    crate::izip!(value, &mut *mod_p0, &mut *mod_p1, &mut *mod_p2)
                {
                    *mod_p0 = value % P0;
                    *mod_p1 = value % P1;
                    *mod_p2 = value % P2;
                }
            },
        );
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
        self.2.fwd(mod_p2);
    }

    pub fn inv(
        &self,
        value: &mut [u64],
        mod_p0: &mut [u64],
        mod_p1: &mut [u64],
        mod_p2: &mut [u64],
    ) {
        self.0.inv(mod_p0);
        self.1.inv(mod_p1);
        self.2.inv(mod_p2);

        reconstruct_slice_52bit_012_avx512(self.3, value, mod_p0, mod_p1, mod_p2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random;

    #[test]
    fn reconstruct_32bit() {
        for n in [32, 64, 256, 1024, 2048] {
            let value = (0..n).map(|_| random::<u64>()).collect::<Vec<_>>();
            let mut value_roundtrip = vec![0; n];
            let mut mod_p0 = vec![0; n];
            let mut mod_p1 = vec![0; n];
            let mut mod_p2 = vec![0; n];
            let mut mod_p3 = vec![0; n];
            let mut mod_p4 = vec![0; n];

            let plan = Plan32::try_new(n).unwrap();
            plan.fwd(
                &value,
                &mut mod_p0,
                &mut mod_p1,
                &mut mod_p2,
                &mut mod_p3,
                &mut mod_p4,
            );
            plan.inv(
                &mut value_roundtrip,
                &mut mod_p0,
                &mut mod_p1,
                &mut mod_p2,
                &mut mod_p3,
                &mut mod_p4,
            );
            for (&value, &value_roundtrip) in crate::izip!(&value, &value_roundtrip) {
                assert_eq!(value_roundtrip, value.wrapping_mul(n as u64));
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly")]
    #[test]
    fn reconstruct_52bit() {
        for n in [32, 64, 256, 1024, 2048] {
            if let Some(plan) = Plan52::try_new(n) {
                let value = (0..n).map(|_| random::<u64>()).collect::<Vec<_>>();
                let mut value_roundtrip = vec![0; n];
                let mut mod_p0 = vec![0; n];
                let mut mod_p1 = vec![0; n];
                let mut mod_p2 = vec![0; n];

                plan.fwd(&value, &mut mod_p0, &mut mod_p1, &mut mod_p2);
                plan.inv(&mut value_roundtrip, &mut mod_p0, &mut mod_p1, &mut mod_p2);
                for (&value, &value_roundtrip) in crate::izip!(&value, &value_roundtrip) {
                    assert_eq!(value_roundtrip, value.wrapping_mul(n as u64));
                }
            }
        }
    }

    #[test]
    fn reconstruct_32bit_avx() {
        for n in [16, 32, 64, 256, 1024, 2048] {
            use crate::primes32::*;

            let mut value = vec![0; n];
            let mut value_avx2 = vec![0; n];
            let mut value_avx512 = vec![0; n];
            let mod_p0 = (0..n).map(|_| random::<u32>() % P0).collect::<Vec<_>>();
            let mod_p1 = (0..n).map(|_| random::<u32>() % P1).collect::<Vec<_>>();
            let mod_p2 = (0..n).map(|_| random::<u32>() % P2).collect::<Vec<_>>();
            let mod_p3 = (0..n).map(|_| random::<u32>() % P3).collect::<Vec<_>>();
            let mod_p4 = (0..n).map(|_| random::<u32>() % P4).collect::<Vec<_>>();

            for (value, &mod_p0, &mod_p1, &mod_p2, &mod_p3, &mod_p4) in
                crate::izip!(&mut value, &mod_p0, &mod_p1, &mod_p2, &mod_p3, &mod_p4)
            {
                *value = reconstruct_32bit_01234_v2(mod_p0, mod_p1, mod_p2, mod_p3, mod_p4);
            }

            if let Some(simd) = crate::Avx2::try_new() {
                reconstruct_slice_32bit_01234_avx2(
                    simd,
                    &mut value_avx2,
                    &mod_p0,
                    &mod_p1,
                    &mod_p2,
                    &mod_p3,
                    &mod_p4,
                );
                assert_eq!(value, value_avx2);
            }
            if let Some(simd) = crate::Avx512::try_new() {
                reconstruct_slice_32bit_01234_avx512(
                    simd,
                    &mut value_avx512,
                    &mod_p0,
                    &mod_p1,
                    &mod_p2,
                    &mod_p3,
                    &mod_p4,
                );
                assert_eq!(value, value_avx512);
            }
        }
    }
}
