use crate::Avx512;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
pub(crate) fn fwd_butterfly_avx512(
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
    let fma = simd.avx512ifma;
    let z0 = simd.small_mod_epu64(p, z0);
    let zero = avx._mm512_setzero_si512();
    let shoup_q = fma._mm512_madd52hi_epu64(zero, z1, w_shoup);
    let t = avx._mm512_and_si512(
        avx._mm512_set1_epi64(((1u64 << 52) - 1u64) as i64),
        fma._mm512_madd52lo_epu64(fma._mm512_madd52lo_epu64(zero, z1, w), shoup_q, neg_p),
    );
    let t = simd.small_mod_epu64(p, t);
    (
        avx._mm512_add_epi64(z0, t),
        avx._mm512_add_epi64(avx._mm512_sub_epi64(z0, t), p),
    )
}

#[inline(always)]
pub(crate) fn fwd_last_butterfly_avx512(
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
    let fma = simd.avx512ifma;
    let z0 = simd.small_mod_epu64(p, z0);
    let zero = avx._mm512_setzero_si512();
    let shoup_q = fma._mm512_madd52hi_epu64(zero, z1, w_shoup);
    let t = avx._mm512_and_si512(
        avx._mm512_set1_epi64(((1u64 << 52) - 1u64) as i64),
        fma._mm512_madd52lo_epu64(fma._mm512_madd52lo_epu64(zero, z1, w), shoup_q, neg_p),
    );
    let t = simd.small_mod_epu64(p, t);
    (
        simd.small_mod_epu64(p, avx._mm512_add_epi64(z0, t)),
        simd.small_mod_epu64(p, avx._mm512_add_epi64(avx._mm512_sub_epi64(z0, t), p)),
    )
}

#[inline(always)]
pub(crate) fn inv_butterfly_avx512(
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
    let fma = simd.avx512ifma;

    let y0 = avx._mm512_add_epi64(z0, z1);
    let y0 = simd.small_mod_epu64(p, y0);
    let t = avx._mm512_add_epi64(avx._mm512_sub_epi64(z0, z1), p);

    let zero = avx._mm512_setzero_si512();
    let shoup_q = fma._mm512_madd52hi_epu64(zero, t, w_shoup);
    let y1 = avx._mm512_and_si512(
        avx._mm512_set1_epi64(((1u64 << 52) - 1u64) as i64),
        fma._mm512_madd52lo_epu64(fma._mm512_madd52lo_epu64(zero, t, w), shoup_q, neg_p),
    );
    let y1 = simd.small_mod_epu64(p, y1);

    (y0, y1)
}

pub(crate) fn fwd_avx512(simd: Avx512, p: u64, data: &mut [u64], twid: &[u64], twid_shoup: &[u64]) {
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
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            fwd_last_butterfly_avx512(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

pub(crate) fn inv_avx512(simd: Avx512, p: u64, data: &mut [u64], twid: &[u64], twid_shoup: &[u64]) {
    super::shoup::inv_breadth_first_avx512(
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
        #[inline(always)]
        |simd, z0, z1, w, w_shoup, p, neg_p, two_p| {
            inv_butterfly_avx512(simd, z0, z1, w, w_shoup, p, neg_p, two_p)
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        fastdiv::Div64,
        prime::largest_prime_in_arithmetic_progression64,
        prime64::{generic_solinas::PrimeModulus, init_negacyclic_twiddles_shoup},
    };

    use super::*;
    use alloc::vec;
    use rand::random;

    extern crate alloc;

    #[test]
    fn test_product() {
        if let Some(simd) = Avx512::try_new() {
            for n in [16, 32, 64, 128, 256, 512, 1024] {
                let p = largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 50, 1 << 51)
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
                    52,
                    &mut twid,
                    &mut twid_shoup,
                    &mut inv_twid,
                    &mut inv_twid_shoup,
                );

                let mut prod = vec![0u64; n];
                let mut lhs_fourier = lhs.clone();
                let mut rhs_fourier = rhs.clone();

                fwd_avx512(simd, p, &mut lhs_fourier, &twid, &twid_shoup);
                fwd_avx512(simd, p, &mut rhs_fourier, &twid, &twid_shoup);
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

                inv_avx512(simd, p, &mut prod, &inv_twid, &inv_twid_shoup);
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
