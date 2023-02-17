#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
use crate::Avx512;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub fn fwd_butterfly_avx512(
    simd: Avx512,
    z0: __m512i,
    z1: __m512i,
    w: __m512i,
    w_shoup: __m512i,
    p: __m512i,
    neg_p: __m512i,
    two_p: __m512i,
) -> (__m512i, __m512i) {
    let _ = two_p;
    let avx = simd.avx512f;
    let z0 = avx._mm512_mask_blend_epi32(
        avx._mm512_cmpge_epu32_mask(z0, p),
        z0,
        avx._mm512_sub_epi32(z0, p),
    );
    let shoup_q = simd._mm512_mul_u32_u32_epu32(z1, w_shoup).1;
    let t = avx._mm512_add_epi32(
        avx._mm512_mullo_epi32(z1, w),
        avx._mm512_mullo_epi32(shoup_q, neg_p),
    );
    let t = avx._mm512_mask_blend_epi32(
        avx._mm512_cmpge_epu32_mask(t, p),
        t,
        avx._mm512_sub_epi32(t, p),
    );
    (
        avx._mm512_add_epi32(z0, t),
        avx._mm512_add_epi32(avx._mm512_sub_epi32(z0, t), p),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn fwd_butterfly_avx2(
    simd: Avx2,
    z0: __m256i,
    z1: __m256i,
    w: __m256i,
    w_shoup: __m256i,
    p: __m256i,
    neg_p: __m256i,
    two_p: __m256i,
) -> (__m256i, __m256i) {
    let _ = two_p;
    let avx = simd.avx2;
    let z0 = avx._mm256_blendv_epi8(
        avx._mm256_sub_epi32(z0, p),
        z0,
        simd._mm256_cmpgt_epu32(p, z0),
    );
    let shoup_q = simd._mm256_mul_u32_u32_epu32(z1, w_shoup).1;
    let t = avx._mm256_add_epi32(
        avx._mm256_mullo_epi32(z1, w),
        avx._mm256_mullo_epi32(shoup_q, neg_p),
    );
    let t = avx._mm256_blendv_epi8(avx._mm256_sub_epi32(t, p), t, simd._mm256_cmpgt_epu32(p, t));
    (
        avx._mm256_add_epi32(z0, t),
        avx._mm256_add_epi32(avx._mm256_sub_epi32(z0, t), p),
    )
}

#[inline(always)]
pub fn fwd_butterfly_scalar(
    z0: u32,
    z1: u32,
    w: u32,
    w_shoup: u32,
    p: u32,
    neg_p: u32,
    two_p: u32,
) -> (u32, u32) {
    let _ = two_p;
    let z0 = z0.min(z0.wrapping_sub(p));
    let shoup_q = ((z1 as u64 * w_shoup as u64) >> 32) as u32;
    let t = u32::wrapping_add(z1.wrapping_mul(w), shoup_q.wrapping_mul(neg_p));
    let t = t.min(t.wrapping_sub(p));
    (z0.wrapping_add(t), z0.wrapping_sub(t).wrapping_add(p))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub fn inv_butterfly_avx512(
    simd: Avx512,
    z0: __m512i,
    z1: __m512i,
    w: __m512i,
    w_shoup: __m512i,
    p: __m512i,
    neg_p: __m512i,
    two_p: __m512i,
) -> (__m512i, __m512i) {
    let _ = two_p;
    let avx = simd.avx512f;

    let y0 = avx._mm512_add_epi32(z0, z1);
    let y0 = avx._mm512_mask_blend_epi32(
        avx._mm512_cmpge_epu32_mask(y0, p),
        y0,
        avx._mm512_sub_epi32(y0, p),
    );
    let t = avx._mm512_add_epi32(avx._mm512_sub_epi32(z0, z1), p);

    let shoup_q = simd._mm512_mul_u32_u32_epu32(t, w_shoup).1;
    let y1 = avx._mm512_add_epi32(
        avx._mm512_mullo_epi32(t, w),
        avx._mm512_mullo_epi32(shoup_q, neg_p),
    );
    let y1 = avx._mm512_mask_blend_epi32(
        avx._mm512_cmpge_epu32_mask(y1, p),
        y1,
        avx._mm512_sub_epi32(y1, p),
    );

    (y0, y1)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn inv_butterfly_avx2(
    simd: Avx2,
    z0: __m256i,
    z1: __m256i,
    w: __m256i,
    w_shoup: __m256i,
    p: __m256i,
    neg_p: __m256i,
    two_p: __m256i,
) -> (__m256i, __m256i) {
    let _ = two_p;
    let avx = simd.avx2;

    let y0 = avx._mm256_add_epi32(z0, z1);
    let y0 = avx._mm256_blendv_epi8(
        avx._mm256_sub_epi32(y0, p),
        y0,
        simd._mm256_cmpgt_epu32(p, y0),
    );
    let t = avx._mm256_add_epi32(avx._mm256_sub_epi32(z0, z1), p);

    let shoup_q = simd._mm256_mul_u32_u32_epu32(t, w_shoup).1;
    let y1 = avx._mm256_add_epi32(
        avx._mm256_mullo_epi32(t, w),
        avx._mm256_mullo_epi32(shoup_q, neg_p),
    );
    let y1 = avx._mm256_blendv_epi8(
        avx._mm256_sub_epi32(y1, p),
        y1,
        simd._mm256_cmpgt_epu32(p, y1),
    );

    (y0, y1)
}

#[inline(always)]
pub fn inv_butterfly_scalar(
    z0: u32,
    z1: u32,
    w: u32,
    w_shoup: u32,
    p: u32,
    neg_p: u32,
    two_p: u32,
) -> (u32, u32) {
    let _ = two_p;

    let y0 = z0.wrapping_add(z1);
    let y0 = y0.min(y0.wrapping_sub(p));
    let t = z0.wrapping_sub(z1).wrapping_add(p);
    let shoup_q = ((t as u64 * w_shoup as u64) >> 32) as u32;
    let y1 = u32::wrapping_add(t.wrapping_mul(w), shoup_q.wrapping_mul(neg_p));
    let y1 = y1.min(y1.wrapping_sub(p));
    (y0, y1)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn fwd_avx512(simd: Avx512, p: u32, data: &mut [u32], twid: &[u32], twid_shoup: &[u32]) {
    super::shoup::fwd_depth_first_avx512(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        0,
        0,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            fwd_butterfly_avx512(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn inv_avx512(simd: Avx512, p: u32, data: &mut [u32], twid: &[u32], twid_shoup: &[u32]) {
    super::shoup::inv_depth_first_avx512(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        0,
        0,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            inv_butterfly_avx512(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_avx2(simd: Avx2, p: u32, data: &mut [u32], twid: &[u32], twid_shoup: &[u32]) {
    super::shoup::fwd_depth_first_avx2(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        0,
        0,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            fwd_butterfly_avx2(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn inv_avx2(simd: Avx2, p: u32, data: &mut [u32], twid: &[u32], twid_shoup: &[u32]) {
    super::shoup::inv_depth_first_avx2(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        0,
        0,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            inv_butterfly_avx2(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

pub fn fwd_scalar(p: u32, data: &mut [u32], twid: &[u32], twid_shoup: &[u32]) {
    super::shoup::fwd_depth_first_scalar(
        p,
        data,
        twid,
        twid_shoup,
        0,
        0,
        #[inline(always)]
        |z0, z1, w, w_shoup, p, neg_p, two_p| {
            fwd_butterfly_scalar(z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

pub fn inv_scalar(p: u32, data: &mut [u32], twid: &[u32], twid_shoup: &[u32]) {
    super::shoup::inv_depth_first_scalar(
        p,
        data,
        twid,
        twid_shoup,
        0,
        0,
        #[inline(always)]
        |z0, z1, w, w_shoup, p, neg_p, two_p| {
            inv_butterfly_scalar(z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fastdiv::Div32, prime::largest_prime_in_arithmetic_progression64,
        _32::init_negacyclic_twiddles_shoup,
    };
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

    #[test]
    fn test_product() {
        for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 30, 1 << 31).unwrap()
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
            let mut twid_shoup = vec![0u32; n];
            let mut inv_twid = vec![0u32; n];
            let mut inv_twid_shoup = vec![0u32; n];
            init_negacyclic_twiddles_shoup(
                p,
                n,
                &mut twid,
                &mut twid_shoup,
                &mut inv_twid,
                &mut inv_twid_shoup,
            );

            let mut prod = vec![0u32; n];
            let mut lhs_fourier = lhs.clone();
            let mut rhs_fourier = rhs.clone();

            fwd_scalar(p, &mut lhs_fourier, &twid, &twid_shoup);
            fwd_scalar(p, &mut rhs_fourier, &twid, &twid_shoup);

            for i in 0..n {
                prod[i] = mul(Div32::new(p), lhs_fourier[i], rhs_fourier[i]);
            }

            inv_scalar(p, &mut prod, &inv_twid, &inv_twid_shoup);
            let result = prod;

            for i in 0..n {
                assert_eq!(
                    result[i] % p,
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
                let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 30, 1 << 31)
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
                let mut twid_shoup = vec![0u32; n];
                let mut inv_twid = vec![0u32; n];
                let mut inv_twid_shoup = vec![0u32; n];
                init_negacyclic_twiddles_shoup(
                    p,
                    n,
                    &mut twid,
                    &mut twid_shoup,
                    &mut inv_twid,
                    &mut inv_twid_shoup,
                );

                let mut prod = vec![0u32; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_avx2(simd, p, &mut lhs_fourier, &twid, &twid_shoup);
                fwd_avx2(simd, p, &mut rhs_fourier, &twid, &twid_shoup);

                for i in 0..n {
                    prod[i] = mul(Div32::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_avx2(simd, p, &mut prod, &inv_twid, &inv_twid_shoup);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i] % p,
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
                let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 30, 1 << 31)
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
                let mut twid_shoup = vec![0u32; n];
                let mut inv_twid = vec![0u32; n];
                let mut inv_twid_shoup = vec![0u32; n];
                init_negacyclic_twiddles_shoup(
                    p,
                    n,
                    &mut twid,
                    &mut twid_shoup,
                    &mut inv_twid,
                    &mut inv_twid_shoup,
                );

                let mut prod = vec![0u32; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_avx512(simd, p, &mut lhs_fourier, &twid, &twid_shoup);
                fwd_avx512(simd, p, &mut rhs_fourier, &twid, &twid_shoup);

                for i in 0..n {
                    prod[i] = mul(Div32::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_avx512(simd, p, &mut prod, &inv_twid, &inv_twid_shoup);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i] % p,
                        mul(Div32::new(p), negacyclic_convolution[i], n as u32),
                    );
                }
            }
        }
    }
}
