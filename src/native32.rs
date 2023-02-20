use aligned_vec::avec;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Negacyclic NTT plan for multiplying two 32bit polynomials.
#[derive(Clone, Debug)]
pub struct Plan32(
    crate::prime32::Plan,
    crate::prime32::Plan,
    crate::prime32::Plan,
);

/// Negacyclic NTT plan for multiplying two 32bit polynomials.  
/// This can be more efficient than [`Plan32`], but requires the AVX512 instruction set.
#[cfg(any(
    doc,
    all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64"))
))]
#[cfg_attr(docsrs, doc(cfg(feature = "nightly")))]
#[derive(Clone, Debug)]
pub struct Plan52(crate::prime64::Plan, crate::prime64::Plan, crate::Avx512);

#[inline(always)]
pub(crate) fn mul_mod32(p: u32, a: u32, b: u32) -> u32 {
    let wide = a as u64 * b as u64;
    (wide % p as u64) as u32
}

#[inline(always)]
pub(crate) fn reconstruct_32bit_012(mod_p0: u32, mod_p1: u32, mod_p2: u32) -> u32 {
    use crate::primes32::*;

    let v0 = mod_p0;
    let v1 = mul_mod32(P1, P0_INV_MOD_P1, 2 * P1 + mod_p1 - v0);
    let v2 = mul_mod32(
        P2,
        P01_INV_MOD_P2,
        2 * P2 + mod_p2 - (v0 + mul_mod32(P2, P0, v1)),
    );

    let sign = v2 > (P2 / 2);

    const _0: u32 = P0;
    const _01: u32 = _0.wrapping_mul(P1);
    const _012: u32 = _01.wrapping_mul(P2);

    let pos = v0
        .wrapping_add(v1.wrapping_mul(_0))
        .wrapping_add(v2.wrapping_mul(_01));

    let neg = pos.wrapping_sub(_012);

    if sign {
        neg
    } else {
        pos
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn mul_mod32_avx2(
    simd: crate::Avx2,
    p: __m256i,
    a: __m256i,
    b: __m256i,
    b_shoup: __m256i,
) -> __m256i {
    let shoup_q = simd._mm256_mul_u32_u32_epu32(a, b_shoup).1;
    let t = simd.avx2._mm256_sub_epi32(
        simd.avx2._mm256_mullo_epi32(a, b),
        simd.avx2._mm256_mullo_epi32(shoup_q, p),
    );
    simd.small_mod_epu32(p, t)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub(crate) fn mul_mod32_avx512(
    simd: crate::Avx512,
    p: __m512i,
    a: __m512i,
    b: __m512i,
    b_shoup: __m512i,
) -> __m512i {
    let shoup_q = simd._mm512_mul_u32_u32_epu32(a, b_shoup).1;
    let t = simd.avx512f._mm512_sub_epi32(
        simd.avx512f._mm512_mullo_epi32(a, b),
        simd.avx512f._mm512_mullo_epi32(shoup_q, p),
    );
    simd.small_mod_epu32(p, t)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub(crate) fn mul_mod52_avx512(
    simd: crate::Avx512,
    p: __m512i,
    neg_p: __m512i,
    a: __m512i,
    b: __m512i,
    b_shoup: __m512i,
) -> __m512i {
    let avx = simd.avx512f;
    let fma = simd.avx512ifma;

    let z = avx._mm512_setzero_si512();
    let shoup_q = fma._mm512_madd52hi_epu64(z, a, b_shoup);
    let t = avx._mm512_and_si512(
        avx._mm512_set1_epi64(((1u64 << 52) - 1) as i64),
        fma._mm512_madd52lo_epu64(fma._mm512_madd52lo_epu64(z, a, b), shoup_q, neg_p),
    );
    simd.small_mod_epu64(p, t)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn reconstruct_32bit_012_avx2(
    simd: crate::Avx2,
    mod_p0: __m256i,
    mod_p1: __m256i,
    mod_p2: __m256i,
) -> __m256i {
    use crate::primes32::*;

    let avx = simd.avx2;

    let p0 = simd.avx._mm256_set1_epi32(P0 as i32);
    let p1 = simd.avx._mm256_set1_epi32(P1 as i32);
    let p2 = simd.avx._mm256_set1_epi32(P2 as i32);
    let two_p1 = simd.avx._mm256_set1_epi32((2 * P1) as i32);
    let two_p2 = simd.avx._mm256_set1_epi32((2 * P2) as i32);
    let half_p2 = simd.avx._mm256_set1_epi32((P2 / 2) as i32);

    let p0_inv_mod_p1 = simd.avx._mm256_set1_epi32(P0_INV_MOD_P1 as i32);
    let p0_inv_mod_p1_shoup = simd.avx._mm256_set1_epi32(P0_INV_MOD_P1_SHOUP as i32);
    let p01_inv_mod_p2 = simd.avx._mm256_set1_epi32(P01_INV_MOD_P2 as i32);
    let p01_inv_mod_p2_shoup = simd.avx._mm256_set1_epi32(P01_INV_MOD_P2_SHOUP as i32);
    let p0_mod_p2_shoup = simd.avx._mm256_set1_epi32(P0_MOD_P2_SHOUP as i32);

    let p01 = simd.avx._mm256_set1_epi32(P0.wrapping_mul(P1) as i32);
    let p012 = simd
        .avx
        ._mm256_set1_epi32(P0.wrapping_mul(P1).wrapping_mul(P2) as i32);

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

    let sign = simd._mm256_cmpgt_epu32(v2, half_p2);
    let pos = avx._mm256_add_epi32(
        avx._mm256_add_epi32(v0, avx._mm256_mullo_epi32(v1, p0)),
        avx._mm256_mullo_epi32(v2, p01),
    );
    let neg = avx._mm256_sub_epi32(pos, p012);

    avx._mm256_blendv_epi8(pos, neg, sign)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn reconstruct_32bit_012_avx512(
    simd: crate::Avx512,
    mod_p0: __m512i,
    mod_p1: __m512i,
    mod_p2: __m512i,
) -> __m512i {
    use crate::primes32::*;

    let avx = simd.avx512f;

    let p0 = avx._mm512_set1_epi32(P0 as i32);
    let p1 = avx._mm512_set1_epi32(P1 as i32);
    let p2 = avx._mm512_set1_epi32(P2 as i32);
    let two_p1 = avx._mm512_set1_epi32((2 * P1) as i32);
    let two_p2 = avx._mm512_set1_epi32((2 * P2) as i32);
    let half_p2 = avx._mm512_set1_epi32((P2 / 2) as i32);

    let p0_inv_mod_p1 = avx._mm512_set1_epi32(P0_INV_MOD_P1 as i32);
    let p0_inv_mod_p1_shoup = avx._mm512_set1_epi32(P0_INV_MOD_P1_SHOUP as i32);
    let p01_inv_mod_p2 = avx._mm512_set1_epi32(P01_INV_MOD_P2 as i32);
    let p01_inv_mod_p2_shoup = avx._mm512_set1_epi32(P01_INV_MOD_P2_SHOUP as i32);
    let p0_mod_p2_shoup = avx._mm512_set1_epi32(P0_MOD_P2_SHOUP as i32);

    let p01 = avx._mm512_set1_epi32(P0.wrapping_mul(P1) as i32);
    let p012 = avx._mm512_set1_epi32(P0.wrapping_mul(P1).wrapping_mul(P2) as i32);

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

    let sign = avx._mm512_cmpgt_epu32_mask(v2, half_p2);
    let pos = avx._mm512_add_epi32(
        avx._mm512_add_epi32(v0, avx._mm512_mullo_epi32(v1, p0)),
        avx._mm512_mullo_epi32(v2, p01),
    );
    let neg = avx._mm512_sub_epi32(pos, p012);

    avx._mm512_mask_blend_epi32(sign, pos, neg)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn reconstruct_52bit_01_avx512(simd: crate::Avx512, mod_p0: __m512i, mod_p1: __m512i) -> __m256i {
    use crate::primes52::*;

    let avx = simd.avx512f;

    let p0 = avx._mm512_set1_epi64(P0 as i64);
    let p1 = avx._mm512_set1_epi64(P1 as i64);
    let neg_p1 = avx._mm512_set1_epi64(P1.wrapping_neg() as i64);
    let two_p1 = avx._mm512_set1_epi64((2 * P1) as i64);
    let half_p1 = avx._mm512_set1_epi64((P1 / 2) as i64);

    let p0_inv_mod_p1 = avx._mm512_set1_epi64(P0_INV_MOD_P1 as i64);
    let p0_inv_mod_p1_shoup = avx._mm512_set1_epi64(P0_INV_MOD_P1_SHOUP as i64);

    let p01 = avx._mm512_set1_epi64(P0.wrapping_mul(P1) as i64);

    let v0 = mod_p0;
    let v1 = mul_mod52_avx512(
        simd,
        p1,
        neg_p1,
        avx._mm512_sub_epi64(avx._mm512_add_epi64(two_p1, mod_p1), v0),
        p0_inv_mod_p1,
        p0_inv_mod_p1_shoup,
    );

    let sign = avx._mm512_cmpgt_epu64_mask(v1, half_p1);

    let pos = avx._mm512_add_epi64(v0, avx._mm512_mullox_epi64(v1, p0));
    let neg = avx._mm512_sub_epi64(pos, p01);

    avx._mm512_cvtepi64_epi32(avx._mm512_mask_blend_epi64(sign, pos, neg))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn reconstruct_slice_32bit_012_avx2(
    simd: crate::Avx2,
    value: &mut [u32],
    mod_p0: &[u32],
    mod_p1: &[u32],
    mod_p2: &[u32],
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
                *value = cast(reconstruct_32bit_012_avx2(
                    simd,
                    cast(mod_p0),
                    cast(mod_p1),
                    cast(mod_p2),
                ));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
fn reconstruct_slice_32bit_012_avx512(
    simd: crate::Avx512,
    value: &mut [u32],
    mod_p0: &[u32],
    mod_p1: &[u32],
    mod_p2: &[u32],
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<16, _>(value).0;
            let mod_p0 = pulp::as_arrays::<16, _>(mod_p0).0;
            let mod_p1 = pulp::as_arrays::<16, _>(mod_p1).0;
            let mod_p2 = pulp::as_arrays::<16, _>(mod_p2).0;
            for (value, &mod_p0, &mod_p1, &mod_p2) in crate::izip!(value, mod_p0, mod_p1, mod_p2) {
                use pulp::cast;
                *value = cast(reconstruct_32bit_012_avx512(
                    simd,
                    cast(mod_p0),
                    cast(mod_p1),
                    cast(mod_p2),
                ));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
fn reconstruct_slice_52bit_01_avx512(
    simd: crate::Avx512,
    value: &mut [u32],
    mod_p0: &[u64],
    mod_p1: &[u64],
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<8, _>(value).0;
            let mod_p0 = pulp::as_arrays::<8, _>(mod_p0).0;
            let mod_p1 = pulp::as_arrays::<8, _>(mod_p1).0;
            for (value, &mod_p0, &mod_p1) in crate::izip!(value, mod_p0, mod_p1) {
                use pulp::cast;
                *value = cast(reconstruct_52bit_01_avx512(
                    simd,
                    cast(mod_p0),
                    cast(mod_p1),
                ));
            }
        },
    );
}

impl Plan32 {
    /// Returns a negacyclic NTT plan for the given polynomial size, or `None` if no
    /// suitable roots of unity can be found for the wanted parameters.
    pub fn try_new(n: usize) -> Option<Self> {
        use crate::{prime32::Plan, primes32::*};
        Some(Self(
            Plan::try_new(n, P0)?,
            Plan::try_new(n, P1)?,
            Plan::try_new(n, P2)?,
        ))
    }

    /// Returns the polynomial size of the negacyclic NTT plan.
    #[inline]
    pub fn ntt_size(&self) -> usize {
        self.0.ntt_size()
    }

    pub fn fwd(&self, value: &[u32], mod_p0: &mut [u32], mod_p1: &mut [u32], mod_p2: &mut [u32]) {
        for (value, mod_p0, mod_p1, mod_p2) in
            crate::izip!(value, &mut *mod_p0, &mut *mod_p1, &mut *mod_p2)
        {
            *mod_p0 = value % crate::primes32::P0;
            *mod_p1 = value % crate::primes32::P1;
            *mod_p2 = value % crate::primes32::P2;
        }
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
        self.2.fwd(mod_p2);
    }

    pub fn inv(
        &self,
        value: &mut [u32],
        mod_p0: &mut [u32],
        mod_p1: &mut [u32],
        mod_p2: &mut [u32],
    ) {
        self.0.inv(mod_p0);
        self.1.inv(mod_p1);
        self.2.inv(mod_p2);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                reconstruct_slice_32bit_012_avx512(simd, value, mod_p0, mod_p1, mod_p2);
                return;
            }
            if let Some(simd) = crate::Avx2::try_new() {
                reconstruct_slice_32bit_012_avx2(simd, value, mod_p0, mod_p1, mod_p2);
                return;
            }
        }

        for (value, &mod_p0, &mod_p1, &mod_p2) in crate::izip!(value, &*mod_p0, &*mod_p1, &*mod_p2)
        {
            *value = reconstruct_32bit_012(mod_p0, mod_p1, mod_p2);
        }
    }

    /// Computes the negacyclic polynomial product of `lhs` and `rhs`, and stores the result in
    /// `prod`.
    pub fn negacyclic_polymul(&self, prod: &mut [u32], lhs: &[u32], rhs: &[u32]) {
        let n = prod.len();
        assert_eq!(n, lhs.len());
        assert_eq!(n, rhs.len());

        let mut lhs0 = avec![0; n];
        let mut lhs1 = avec![0; n];
        let mut lhs2 = avec![0; n];

        let mut rhs0 = avec![0; n];
        let mut rhs1 = avec![0; n];
        let mut rhs2 = avec![0; n];

        self.fwd(lhs, &mut lhs0, &mut lhs1, &mut lhs2);
        self.fwd(rhs, &mut rhs0, &mut rhs1, &mut rhs2);

        self.0.mul_assign_normalize(&mut lhs0, &rhs0);
        self.1.mul_assign_normalize(&mut lhs1, &rhs1);
        self.2.mul_assign_normalize(&mut lhs2, &rhs2);

        self.inv(prod, &mut lhs0, &mut lhs1, &mut lhs2);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl Plan52 {
    /// Returns a negacyclic NTT plan for the given polynomial size, or `None` if no
    /// suitable roots of unity can be found for the wanted parameters, or if the AVX512
    /// instruction set isn't detected.
    pub fn try_new(n: usize) -> Option<Self> {
        use crate::{prime64::Plan, primes52::*};
        let simd = crate::Avx512::try_new()?;
        Some(Self(Plan::try_new(n, P0)?, Plan::try_new(n, P1)?, simd))
    }

    /// Returns the polynomial size of the negacyclic NTT plan.
    #[inline]
    pub fn ntt_size(&self) -> usize {
        self.0.ntt_size()
    }

    pub fn fwd(&self, value: &[u32], mod_p0: &mut [u64], mod_p1: &mut [u64]) {
        self.2.vectorize(
            #[inline(always)]
            || {
                for (value, mod_p0, mod_p1) in crate::izip!(value, &mut *mod_p0, &mut *mod_p1) {
                    *mod_p0 = *value as u64;
                    *mod_p1 = *value as u64;
                }
            },
        );
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
    }

    pub fn inv(&self, value: &mut [u32], mod_p0: &mut [u64], mod_p1: &mut [u64]) {
        self.0.inv(mod_p0);
        self.1.inv(mod_p1);

        let simd = self.2;
        simd.vectorize(
            #[inline(always)]
            || {
                reconstruct_slice_52bit_01_avx512(simd, value, mod_p0, mod_p1);
            },
        )
    }

    /// Computes the negacyclic polynomial product of `lhs` and `rhs`, and stores the result in
    /// `prod`.
    pub fn negacyclic_polymul(&self, prod: &mut [u32], lhs: &[u32], rhs: &[u32]) {
        let n = prod.len();
        assert_eq!(n, lhs.len());
        assert_eq!(n, rhs.len());

        let mut lhs0 = avec![0; n];
        let mut lhs1 = avec![0; n];

        let mut rhs0 = avec![0; n];
        let mut rhs1 = avec![0; n];

        self.fwd(lhs, &mut lhs0, &mut lhs1);
        self.fwd(rhs, &mut rhs0, &mut rhs1);

        self.0.mul_assign_normalize(&mut lhs0, &rhs0);
        self.1.mul_assign_normalize(&mut lhs1, &rhs1);

        self.inv(prod, &mut lhs0, &mut lhs1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random;

    #[test]
    fn reconstruct_32bit() {
        for n in [32, 64, 256, 1024, 2048] {
            let plan = Plan32::try_new(n).unwrap();

            let value = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
            let mut value_roundtrip = vec![0u32; n];
            let mut mod_p0 = vec![0u32; n];
            let mut mod_p1 = vec![0u32; n];
            let mut mod_p2 = vec![0u32; n];

            plan.fwd(&value, &mut mod_p0, &mut mod_p1, &mut mod_p2);
            plan.inv(&mut value_roundtrip, &mut mod_p0, &mut mod_p1, &mut mod_p2);
            for (&value, &value_roundtrip) in crate::izip!(&value, &value_roundtrip) {
                assert_eq!(value_roundtrip, value.wrapping_mul(n as u32));
            }

            let lhs = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
            let mut full_convolution = vec![0u32; 2 * n];
            let mut negacyclic_convolution = vec![0u32; n];
            for i in 0..n {
                for j in 0..n {
                    full_convolution[i + j] =
                        full_convolution[i + j].wrapping_add(lhs[i].wrapping_mul(rhs[j]));
                }
            }
            for i in 0..n {
                negacyclic_convolution[i] =
                    full_convolution[i].wrapping_sub(full_convolution[i + n]);
            }

            let mut prod = vec![0; n];
            plan.negacyclic_polymul(&mut prod, &lhs, &rhs);
            assert_eq!(prod, negacyclic_convolution);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly")]
    #[test]
    fn reconstruct_52bit() {
        for n in [32, 64, 256, 1024, 2048] {
            if let Some(plan) = Plan52::try_new(n) {
                let value = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
                let mut value_roundtrip = vec![0u32; n];
                let mut mod_p0 = vec![0u64; n];
                let mut mod_p1 = vec![0u64; n];

                plan.fwd(&value, &mut mod_p0, &mut mod_p1);
                plan.inv(&mut value_roundtrip, &mut mod_p0, &mut mod_p1);
                for (&value, &value_roundtrip) in crate::izip!(&value, &value_roundtrip) {
                    assert_eq!(value_roundtrip, value.wrapping_mul(n as u32));
                }

                let lhs = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
                let rhs = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
                let mut full_convolution = vec![0u32; 2 * n];
                let mut negacyclic_convolution = vec![0u32; n];
                for i in 0..n {
                    for j in 0..n {
                        full_convolution[i + j] =
                            full_convolution[i + j].wrapping_add(lhs[i].wrapping_mul(rhs[j]));
                    }
                }
                for i in 0..n {
                    negacyclic_convolution[i] =
                        full_convolution[i].wrapping_sub(full_convolution[i + n]);
                }

                let mut prod = vec![0; n];
                plan.negacyclic_polymul(&mut prod, &lhs, &rhs);
                assert_eq!(prod, negacyclic_convolution);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn reconstruct_32bit_avx() {
        for n in [16, 32, 64, 256, 1024, 2048] {
            use crate::primes32::*;

            let mut value = vec![0u32; n];
            let mut value_avx2 = vec![0u32; n];
            #[cfg(feature = "nightly")]
            let mut value_avx512 = vec![0u32; n];
            let mod_p0 = (0..n).map(|_| random::<u32>() % P0).collect::<Vec<_>>();
            let mod_p1 = (0..n).map(|_| random::<u32>() % P1).collect::<Vec<_>>();
            let mod_p2 = (0..n).map(|_| random::<u32>() % P2).collect::<Vec<_>>();

            for (value, &mod_p0, &mod_p1, &mod_p2) in
                crate::izip!(&mut value, &mod_p0, &mod_p1, &mod_p2)
            {
                *value = reconstruct_32bit_012(mod_p0, mod_p1, mod_p2);
            }

            if let Some(simd) = crate::Avx2::try_new() {
                reconstruct_slice_32bit_012_avx2(simd, &mut value_avx2, &mod_p0, &mod_p1, &mod_p2);
                assert_eq!(value, value_avx2);
            }
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                reconstruct_slice_32bit_012_avx512(
                    simd,
                    &mut value_avx512,
                    &mod_p0,
                    &mod_p1,
                    &mod_p2,
                );
                assert_eq!(value, value_avx512);
            }
        }
    }
}
