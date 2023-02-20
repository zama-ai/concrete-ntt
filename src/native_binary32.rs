use crate::native32::mul_mod32;
use aligned_vec::avec;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Negacyclic NTT plan for multiplying two 32bit polynomials, where the RHS contains binary
/// coefficients.
#[derive(Clone, Debug)]
pub struct Plan32(crate::prime32::Plan, crate::prime32::Plan);

/// Negacyclic NTT plan for multiplying two 32bit polynomials, where the RHS contains binary
/// coefficients.  
/// This can be more efficient than [`Plan32`], but requires the AVX512 instruction set.
#[cfg(any(
    doc,
    all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64"))
))]
#[cfg_attr(docsrs, doc(cfg(feature = "nightly")))]
#[derive(Clone, Debug)]
pub struct Plan52(crate::prime64::Plan, crate::Avx512);

#[inline(always)]
pub(crate) fn reconstruct_32bit_01(mod_p0: u32, mod_p1: u32) -> u32 {
    use crate::primes32::*;

    let v0 = mod_p0;
    let v1 = mul_mod32(P1, P0_INV_MOD_P1, 2 * P1 + mod_p1 - v0);

    let sign = v1 > (P1 / 2);

    const _0: u32 = P0;
    const _01: u32 = _0.wrapping_mul(P1);

    let pos = v0.wrapping_add(v1.wrapping_mul(_0));
    let neg = pos.wrapping_sub(_01);

    if sign {
        neg
    } else {
        pos
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn reconstruct_32bit_01_avx2(
    simd: crate::Avx2,
    mod_p0: __m256i,
    mod_p1: __m256i,
) -> __m256i {
    use crate::{native32::mul_mod32_avx2, primes32::*};

    let avx = simd.avx2;

    let p0 = simd.avx._mm256_set1_epi32(P0 as i32);
    let p1 = simd.avx._mm256_set1_epi32(P1 as i32);
    let two_p1 = simd.avx._mm256_set1_epi32((2 * P1) as i32);
    let half_p1 = simd.avx._mm256_set1_epi32((P1 / 2) as i32);

    let p0_inv_mod_p1 = simd.avx._mm256_set1_epi32(P0_INV_MOD_P1 as i32);
    let p0_inv_mod_p1_shoup = simd.avx._mm256_set1_epi32(P0_INV_MOD_P1_SHOUP as i32);

    let p01 = simd.avx._mm256_set1_epi32(P0.wrapping_mul(P1) as i32);

    let v0 = mod_p0;
    let v1 = mul_mod32_avx2(
        simd,
        p1,
        avx._mm256_sub_epi32(avx._mm256_add_epi32(two_p1, mod_p1), v0),
        p0_inv_mod_p1,
        p0_inv_mod_p1_shoup,
    );

    let sign = simd._mm256_cmpgt_epu32(v1, half_p1);
    let pos = avx._mm256_add_epi32(v0, avx._mm256_mullo_epi32(v1, p0));

    let neg = avx._mm256_sub_epi32(pos, p01);

    avx._mm256_blendv_epi8(pos, neg, sign)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn reconstruct_32bit_01_avx512(simd: crate::Avx512, mod_p0: __m512i, mod_p1: __m512i) -> __m512i {
    use crate::{native32::mul_mod32_avx512, primes32::*};

    let avx = simd.avx512f;

    let p0 = avx._mm512_set1_epi32(P0 as i32);
    let p1 = avx._mm512_set1_epi32(P1 as i32);
    let two_p1 = avx._mm512_set1_epi32((2 * P1) as i32);
    let half_p1 = avx._mm512_set1_epi32((P1 / 2) as i32);

    let p0_inv_mod_p1 = avx._mm512_set1_epi32(P0_INV_MOD_P1 as i32);
    let p0_inv_mod_p1_shoup = avx._mm512_set1_epi32(P0_INV_MOD_P1_SHOUP as i32);

    let p01 = avx._mm512_set1_epi32(P0.wrapping_mul(P1) as i32);

    let v0 = mod_p0;
    let v1 = mul_mod32_avx512(
        simd,
        p1,
        avx._mm512_sub_epi32(avx._mm512_add_epi32(two_p1, mod_p1), v0),
        p0_inv_mod_p1,
        p0_inv_mod_p1_shoup,
    );

    let sign = avx._mm512_cmpgt_epu32_mask(v1, half_p1);
    let pos = avx._mm512_add_epi32(v0, avx._mm512_mullo_epi32(v1, p0));

    let neg = avx._mm512_sub_epi32(pos, p01);

    avx._mm512_mask_blend_epi32(sign, pos, neg)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn reconstruct_52bit_0_avx512(simd: crate::Avx512, mod_p0: __m512i) -> __m256i {
    use crate::primes52::*;

    let avx = simd.avx512f;

    let p0 = avx._mm512_set1_epi64(P0 as i64);
    let half_p0 = avx._mm512_set1_epi64((P0 / 2) as i64);

    let v0 = mod_p0;

    let sign = avx._mm512_cmpgt_epu64_mask(v0, half_p0);

    let pos = v0;
    let neg = avx._mm512_sub_epi64(pos, p0);

    avx._mm512_cvtepi64_epi32(avx._mm512_mask_blend_epi64(sign, pos, neg))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn reconstruct_slice_32bit_01_avx2(
    simd: crate::Avx2,
    value: &mut [u32],
    mod_p0: &[u32],
    mod_p1: &[u32],
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<8, _>(value).0;
            let mod_p0 = pulp::as_arrays::<8, _>(mod_p0).0;
            let mod_p1 = pulp::as_arrays::<8, _>(mod_p1).0;
            for (value, &mod_p0, &mod_p1) in crate::izip!(value, mod_p0, mod_p1) {
                use pulp::cast;
                *value = cast(reconstruct_32bit_01_avx2(simd, cast(mod_p0), cast(mod_p1)));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
fn reconstruct_slice_32bit_01_avx512(
    simd: crate::Avx512,
    value: &mut [u32],
    mod_p0: &[u32],
    mod_p1: &[u32],
) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<16, _>(value).0;
            let mod_p0 = pulp::as_arrays::<16, _>(mod_p0).0;
            let mod_p1 = pulp::as_arrays::<16, _>(mod_p1).0;
            for (value, &mod_p0, &mod_p1) in crate::izip!(value, mod_p0, mod_p1) {
                use pulp::cast;
                *value = cast(reconstruct_32bit_01_avx512(
                    simd,
                    cast(mod_p0),
                    cast(mod_p1),
                ));
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
fn reconstruct_slice_52bit_0_avx512(simd: crate::Avx512, value: &mut [u32], mod_p0: &[u64]) {
    simd.vectorize(
        #[inline(always)]
        move || {
            let value = pulp::as_arrays_mut::<8, _>(value).0;
            let mod_p0 = pulp::as_arrays::<8, _>(mod_p0).0;
            for (value, &mod_p0) in crate::izip!(value, mod_p0) {
                use pulp::cast;
                *value = cast(reconstruct_52bit_0_avx512(simd, cast(mod_p0)));
            }
        },
    );
}

impl Plan32 {
    /// Returns a negacyclic NTT plan for the given polynomial size, or `None` if no
    /// suitable roots of unity can be found for the wanted parameters.
    pub fn try_new(n: usize) -> Option<Self> {
        use crate::{prime32::Plan, primes32::*};
        Some(Self(Plan::try_new(n, P0)?, Plan::try_new(n, P1)?))
    }

    /// Returns the polynomial size of the negacyclic NTT plan.
    #[inline]
    pub fn ntt_size(&self) -> usize {
        self.0.ntt_size()
    }

    pub fn fwd(&self, value: &[u32], mod_p0: &mut [u32], mod_p1: &mut [u32]) {
        for (value, mod_p0, mod_p1) in crate::izip!(value, &mut *mod_p0, &mut *mod_p1) {
            *mod_p0 = value % crate::primes32::P0;
            *mod_p1 = value % crate::primes32::P1;
        }
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
    }

    pub fn fwd_binary(&self, value: &[u32], mod_p0: &mut [u32], mod_p1: &mut [u32]) {
        for (value, mod_p0, mod_p1) in crate::izip!(value, &mut *mod_p0, &mut *mod_p1) {
            *mod_p0 = *value;
            *mod_p1 = *value;
        }
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
    }

    pub fn inv(&self, value: &mut [u32], mod_p0: &mut [u32], mod_p1: &mut [u32]) {
        self.0.inv(mod_p0);
        self.1.inv(mod_p1);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly")]
            if let Some(simd) = crate::Avx512::try_new() {
                reconstruct_slice_32bit_01_avx512(simd, value, mod_p0, mod_p1);
                return;
            }
            if let Some(simd) = crate::Avx2::try_new() {
                reconstruct_slice_32bit_01_avx2(simd, value, mod_p0, mod_p1);
                return;
            }
        }

        for (value, &mod_p0, &mod_p1) in crate::izip!(value, &*mod_p0, &*mod_p1) {
            *value = reconstruct_32bit_01(mod_p0, mod_p1);
        }
    }

    /// Computes the negacyclic polynomial product of `lhs` and `rhs`, and stores the result in
    /// `prod`.
    pub fn negacyclic_polymul(&self, prod: &mut [u32], lhs: &[u32], rhs_binary: &[u32]) {
        let n = prod.len();
        assert_eq!(n, lhs.len());
        assert_eq!(n, rhs_binary.len());

        let mut lhs0 = avec![0; n];
        let mut lhs1 = avec![0; n];

        let mut rhs0 = avec![0; n];
        let mut rhs1 = avec![0; n];

        self.fwd(lhs, &mut lhs0, &mut lhs1);
        self.fwd_binary(rhs_binary, &mut rhs0, &mut rhs1);

        self.0.mul_assign_normalize(&mut lhs0, &rhs0);
        self.1.mul_assign_normalize(&mut lhs1, &rhs1);

        self.inv(prod, &mut lhs0, &mut lhs1);
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
        Some(Self(Plan::try_new(n, P0)?, simd))
    }

    /// Returns the polynomial size of the negacyclic NTT plan.
    #[inline]
    pub fn ntt_size(&self) -> usize {
        self.0.ntt_size()
    }

    pub fn fwd(&self, value: &[u32], mod_p0: &mut [u64]) {
        self.1.vectorize(
            #[inline(always)]
            || {
                for (value, mod_p0) in crate::izip!(value, &mut *mod_p0) {
                    *mod_p0 = *value as u64;
                }
            },
        );
        self.0.fwd(mod_p0);
    }

    pub fn fwd_binary(&self, value: &[u32], mod_p0: &mut [u64]) {
        self.fwd(value, mod_p0);
    }

    pub fn inv(&self, value: &mut [u32], mod_p0: &mut [u64]) {
        self.0.inv(mod_p0);

        let simd = self.1;
        simd.vectorize(
            #[inline(always)]
            || {
                reconstruct_slice_52bit_0_avx512(simd, value, mod_p0);
            },
        )
    }

    /// Computes the negacyclic polynomial product of `lhs` and `rhs`, and stores the result in
    /// `prod`.
    pub fn negacyclic_polymul(&self, prod: &mut [u32], lhs: &[u32], rhs_binary: &[u32]) {
        let n = prod.len();
        assert_eq!(n, lhs.len());
        assert_eq!(n, rhs_binary.len());

        let mut lhs0 = avec![0; n];
        let mut rhs0 = avec![0; n];

        self.fwd(lhs, &mut lhs0);
        self.fwd_binary(rhs_binary, &mut rhs0);

        self.0.mul_assign_normalize(&mut lhs0, &rhs0);

        self.inv(prod, &mut lhs0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random;

    extern crate alloc;
    use alloc::{vec, vec::Vec};

    #[test]
    fn reconstruct_32bit() {
        for n in [32, 64, 256, 1024, 2048] {
            let plan = Plan32::try_new(n).unwrap();

            let lhs = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| random::<u32>() % 2).collect::<Vec<_>>();
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
                let lhs = (0..n).map(|_| random::<u32>()).collect::<Vec<_>>();
                let rhs = (0..n).map(|_| random::<u32>() % 2).collect::<Vec<_>>();
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
}
