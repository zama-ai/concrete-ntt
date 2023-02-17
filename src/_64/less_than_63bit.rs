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
    let z0 = avx._mm512_mask_blend_epi64(
        avx._mm512_cmpge_epu64_mask(z0, p),
        z0,
        avx._mm512_sub_epi64(z0, p),
    );
    let shoup_q = simd._mm512_mul_u64_u64_epu64(z1, w_shoup).1;
    let t = avx._mm512_add_epi64(
        avx._mm512_mullox_epi64(z1, w),
        avx._mm512_mullox_epi64(shoup_q, neg_p),
    );
    let t = avx._mm512_mask_blend_epi64(
        avx._mm512_cmpge_epu64_mask(t, p),
        t,
        avx._mm512_sub_epi64(t, p),
    );
    (
        avx._mm512_add_epi64(z0, t),
        avx._mm512_add_epi64(avx._mm512_sub_epi64(z0, t), p),
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
        avx._mm256_sub_epi64(z0, p),
        z0,
        simd._mm256_cmpgt_epu64(p, z0),
    );
    let shoup_q = simd._mm256_mul_u64_u64_epu64(z1, w_shoup).1;
    let t = avx._mm256_add_epi64(
        simd._mm256_mul_u64_u64_epu64(z1, w).0,
        simd._mm256_mul_u64_u64_epu64(shoup_q, neg_p).0,
    );
    let t = avx._mm256_blendv_epi8(
        avx._mm256_sub_epi64(t, p),
        t,
        simd._mm256_cmpgt_epu64(p, t),
    );
    (
        avx._mm256_add_epi64(z0, t),
        avx._mm256_add_epi64(avx._mm256_sub_epi64(z0, t), p),
    )
}

#[inline(always)]
pub fn fwd_butterfly_scalar(
    z0: u64,
    z1: u64,
    w: u64,
    w_shoup: u64,
    p: u64,
    neg_p: u64,
    two_p: u64,
) -> (u64, u64) {
    let _ = two_p;
    let z0 = z0.min(z0.wrapping_sub(p));
    let shoup_q = ((z1 as u128 * w_shoup as u128) >> 64) as u64;
    let t = u64::wrapping_add(z1.wrapping_mul(w), shoup_q.wrapping_mul(neg_p));
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

    let y0 = avx._mm512_add_epi64(z0, z1);
    let y0 = avx._mm512_mask_blend_epi64(
        avx._mm512_cmpge_epu64_mask(y0, p),
        y0,
        avx._mm512_sub_epi64(y0, p),
    );
    let t = avx._mm512_add_epi64(avx._mm512_sub_epi64(z0, z1), p);

    let shoup_q = simd._mm512_mul_u64_u64_epu64(t, w_shoup).1;
    let y1 = avx._mm512_add_epi64(
        avx._mm512_mullox_epi64(t, w),
        avx._mm512_mullox_epi64(shoup_q, neg_p),
    );
    let y1 = avx._mm512_mask_blend_epi64(
        avx._mm512_cmpge_epu64_mask(y1, p),
        y1,
        avx._mm512_sub_epi64(y1, p),
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

    let y0 = avx._mm256_add_epi64(z0, z1);
    let y0 = avx._mm256_blendv_epi8(
        avx._mm256_sub_epi64(y0, p),
        y0,
        simd._mm256_cmpgt_epu64(p, y0),
    );
    let t = avx._mm256_add_epi64(avx._mm256_sub_epi64(z0, z1), p);

    let shoup_q = simd._mm256_mul_u64_u64_epu64(t, w_shoup).1;
    let y1 = avx._mm256_add_epi64(
        simd._mm256_mul_u64_u64_epu64(t, w).0,
        simd._mm256_mul_u64_u64_epu64(shoup_q, neg_p).0,
    );
    let y1 = avx._mm256_blendv_epi8(
        avx._mm256_sub_epi64(y1, p),
        y1,
        simd._mm256_cmpgt_epu64(p, y1),
    );

    (y0, y1)
}

#[inline(always)]
pub fn inv_butterfly_scalar(
    z0: u64,
    z1: u64,
    w: u64,
    w_shoup: u64,
    p: u64,
    neg_p: u64,
    two_p: u64,
) -> (u64, u64) {
    let _ = two_p;

    let y0 = z0.wrapping_add(z1);
    let y0 = y0.min(y0.wrapping_sub(p));
    let t = z0.wrapping_sub(z1).wrapping_add(p);
    let shoup_q = ((t as u128 * w_shoup as u128) >> 64) as u64;
    let y1 = u64::wrapping_add(t.wrapping_mul(w), shoup_q.wrapping_mul(neg_p));
    let y1 = y1.min(y1.wrapping_sub(p));
    (y0, y1)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn fwd_breadth_first_avx512(
    simd: Avx512,
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    super::shoup::fwd_breadth_first_avx512(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        recursion_depth,
        recursion_half,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            fwd_butterfly_avx512(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn inv_breadth_first_avx512(
    simd: Avx512,
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    super::shoup::inv_breadth_first_avx512(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        recursion_depth,
        recursion_half,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            inv_butterfly_avx512(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_breadth_first_avx2(
    simd: Avx2,
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    super::shoup::fwd_breadth_first_avx2(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        recursion_depth,
        recursion_half,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            fwd_butterfly_avx2(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn inv_breadth_first_avx2(
    simd: Avx2,
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    super::shoup::inv_breadth_first_avx2(
        simd,
        p,
        data,
        twid,
        twid_shoup,
        recursion_depth,
        recursion_half,
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            inv_butterfly_avx2(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

pub fn fwd_breadth_first_scalar(
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    super::shoup::fwd_breadth_first_scalar(
        p,
        data,
        twid,
        twid_shoup,
        recursion_depth,
        recursion_half,
        #[inline(always)]
        |z0, z1, w, w_shoup, p, neg_p, two_p| {
            fwd_butterfly_scalar(z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

pub fn inv_breadth_first_scalar(
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
) {
    super::shoup::inv_breadth_first_scalar(
        p,
        data,
        twid,
        twid_shoup,
        recursion_depth,
        recursion_half,
        #[inline(always)]
        |z0, z1, w, w_shoup, p, neg_p, two_p| {
            inv_butterfly_scalar(z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        fastdiv::Div64,
        prime::largest_prime_in_arithmetic_progression64,
        _64::{generic_solinas::PrimeModulus, init_negacyclic_twiddles_shoup},
    };

    use super::*;
    use alloc::vec;
    use rand::random;

    extern crate alloc;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly")]
    #[test]
    fn test_product_avx512() {
        if let Some(simd) = Avx512::try_new() {
            for n in [16, 32, 64, 128, 256, 512, 1024] {
                let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63)
                    .unwrap();

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
                let mut twid_shoup = vec![0u64; n];
                let mut inv_twid = vec![0u64; n];
                let mut inv_twid_shoup = vec![0u64; n];
                init_negacyclic_twiddles_shoup(
                    p,
                    n,
                    64,
                    &mut twid,
                    &mut twid_shoup,
                    &mut inv_twid,
                    &mut inv_twid_shoup,
                );

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_breadth_first_avx512(simd, p, &mut lhs_fourier, &twid, &twid_shoup, 0, 0);
                fwd_breadth_first_avx512(simd, p, &mut rhs_fourier, &twid, &twid_shoup, 0, 0);

                for i in 0..n {
                    prod[i] =
                        <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_breadth_first_avx512(simd, p, &mut prod, &inv_twid, &inv_twid_shoup, 0, 0);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i] % p,
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
    #[test]
    fn test_product_avx2() {
        if let Some(simd) = Avx2::try_new() {
            for n in [16, 32, 64, 128, 256, 512, 1024] {
                let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63)
                    .unwrap();

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
                let mut twid_shoup = vec![0u64; n];
                let mut inv_twid = vec![0u64; n];
                let mut inv_twid_shoup = vec![0u64; n];
                init_negacyclic_twiddles_shoup(
                    p,
                    n,
                    64,
                    &mut twid,
                    &mut twid_shoup,
                    &mut inv_twid,
                    &mut inv_twid_shoup,
                );

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_breadth_first_avx2(simd, p, &mut lhs_fourier, &twid, &twid_shoup, 0, 0);
                fwd_breadth_first_avx2(simd, p, &mut rhs_fourier, &twid, &twid_shoup, 0, 0);

                for i in 0..n {
                    prod[i] =
                        <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
                }

                inv_breadth_first_avx2(simd, p, &mut prod, &inv_twid, &inv_twid_shoup, 0, 0);
                let result = prod;

                for i in 0..n {
                    assert_eq!(
                        result[i] % p,
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
    fn test_product_scalar() {
        for n in [16, 32, 64, 128, 256, 512, 1024] {
            let p =
                largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63).unwrap();

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
            let mut twid_shoup = vec![0u64; n];
            let mut inv_twid = vec![0u64; n];
            let mut inv_twid_shoup = vec![0u64; n];
            init_negacyclic_twiddles_shoup(
                p,
                n,
                64,
                &mut twid,
                &mut twid_shoup,
                &mut inv_twid,
                &mut inv_twid_shoup,
            );

            let mut prod = vec![0u64; n];
            let mut lhs_fourier = lhs.clone();
            let mut rhs_fourier = rhs.clone();

            fwd_breadth_first_scalar(p, &mut lhs_fourier, &twid, &twid_shoup, 0, 0);
            fwd_breadth_first_scalar(p, &mut rhs_fourier, &twid, &twid_shoup, 0, 0);

            for i in 0..n {
                prod[i] = <u64 as PrimeModulus>::mul(Div64::new(p), lhs_fourier[i], rhs_fourier[i]);
            }

            inv_breadth_first_scalar(p, &mut prod, &inv_twid, &inv_twid_shoup, 0, 0);
            let result = prod;

            for i in 0..n {
                assert_eq!(
                    result[i] % p,
                    <u64 as PrimeModulus>::mul(Div64::new(p), negacyclic_convolution[i], n as u64),
                );
            }
        }
    }
}
