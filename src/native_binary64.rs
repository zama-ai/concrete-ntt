use aligned_vec::avec;
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

pub struct Plan32(crate::_32::Plan, crate::_32::Plan, crate::_32::Plan);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub struct Plan52(crate::_64::Plan, crate::_64::Plan, crate::Avx512);

#[inline(always)]
#[allow(dead_code)]
fn reconstruct_32bit_012(mod_p0: u32, mod_p1: u32, mod_p2: u32) -> u64 {
    use crate::primes32::*;

    let v0 = mod_p0;
    let v1 = mul_mod32(P1, P0_INV_MOD_P1, 2 * P1 + mod_p1 - v0);
    let v2 = mul_mod32(
        P2,
        P01_INV_MOD_P2,
        2 * P2 + mod_p2 - (v0 + mul_mod32(P2, P0, v1)),
    );

    let sign = v2 > (P2 / 2);

    const _0: u64 = P0 as u64;
    const _01: u64 = _0.wrapping_mul(P1 as u64);
    const _012: u64 = _01.wrapping_mul(P2 as u64);

    let pos = (v0 as u64)
        .wrapping_add((v1 as u64).wrapping_mul(_0))
        .wrapping_add((v2 as u64).wrapping_mul(_01));

    let neg = pos.wrapping_sub(_012);

    if sign {
        neg
    } else {
        pos
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(dead_code)]
#[inline(always)]
fn reconstruct_32bit_012_avx2(
    simd: crate::Avx2,
    mod_p0: __m256i,
    mod_p1: __m256i,
    mod_p2: __m256i,
) -> [__m256i; 2] {
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
    let p0_mod_p2_shoup = simd.avx._mm256_set1_epi32(P0_MOD_P2_SHOUP as i32);

    let p01_inv_mod_p2 = simd.avx._mm256_set1_epi32(P01_INV_MOD_P2 as i32);
    let p01_inv_mod_p2_shoup = simd.avx._mm256_set1_epi32(P01_INV_MOD_P2_SHOUP as i32);

    let p01 = simd
        .avx
        ._mm256_set1_epi64x((P0 as u64).wrapping_mul(P1 as u64) as i64);
    let p012 = simd
        .avx
        ._mm256_set1_epi64x((P0 as u64).wrapping_mul(P1 as u64).wrapping_mul(P2 as u64) as i64);

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
    // sign extend so that -1i32 becomes -1i64
    let sign0 = avx._mm256_cvtepi32_epi64(simd.avx._mm256_castsi256_si128(sign));
    let sign1 = avx._mm256_cvtepi32_epi64(simd.avx._mm256_extractf128_si256::<1>(sign));

    let v00 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v0));
    let v01 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v0));
    let v10 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v1));
    let v11 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v1));
    let v20 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_castsi256_si128(v2));
    let v21 = avx._mm256_cvtepu32_epi64(simd.avx._mm256_extractf128_si256::<1>(v2));

    let pos0 = v00;
    let pos0 = avx._mm256_add_epi64(pos0, avx._mm256_mul_epi32(p0, v10));
    let pos0 = avx._mm256_add_epi64(pos0, simd._mm256_mullo_u64_u32_epu64(p01, v20));

    let pos1 = v01;
    let pos1 = avx._mm256_add_epi64(pos1, avx._mm256_mul_epi32(p0, v11));
    let pos1 = avx._mm256_add_epi64(pos1, simd._mm256_mullo_u64_u32_epu64(p01, v21));

    let neg0 = avx._mm256_sub_epi64(pos0, p012);
    let neg1 = avx._mm256_sub_epi64(pos1, p012);

    [
        avx._mm256_blendv_epi8(pos0, neg0, sign0),
        avx._mm256_blendv_epi8(pos1, neg1, sign1),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[allow(dead_code)]
#[inline(always)]
fn reconstruct_32bit_012_avx512(
    simd: crate::Avx512,
    mod_p0: __m512i,
    mod_p1: __m512i,
    mod_p2: __m512i,
) -> [__m512i; 2] {
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
    let p0_mod_p2_shoup = avx._mm512_set1_epi32(P0_MOD_P2_SHOUP as i32);

    let p01_inv_mod_p2 = avx._mm512_set1_epi32(P01_INV_MOD_P2 as i32);
    let p01_inv_mod_p2_shoup = avx._mm512_set1_epi32(P01_INV_MOD_P2_SHOUP as i32);

    let p01 = avx._mm512_set1_epi64((P0 as u64).wrapping_mul(P1 as u64) as i64);
    let p012 =
        avx._mm512_set1_epi64((P0 as u64).wrapping_mul(P1 as u64).wrapping_mul(P2 as u64) as i64);

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
    let sign0 = sign as u8;
    let sign1 = (sign >> 8) as u8;
    let v00 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v0));
    let v01 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v0));
    let v10 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v1));
    let v11 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v1));
    let v20 = avx._mm512_cvtepu32_epi64(avx._mm512_castsi512_si256(v2));
    let v21 = avx._mm512_cvtepu32_epi64(avx._mm512_extracti64x4_epi64::<1>(v2));

    let pos0 = v00;
    let pos0 = avx._mm512_add_epi64(pos0, avx._mm512_mul_epi32(p0, v10));
    let pos0 = avx._mm512_add_epi64(pos0, avx._mm512_mullox_epi64(p01, v20));

    let pos1 = v01;
    let pos1 = avx._mm512_add_epi64(pos1, avx._mm512_mul_epi32(p0, v11));
    let pos1 = avx._mm512_add_epi64(pos1, avx._mm512_mullox_epi64(p01, v21));

    let neg0 = avx._mm512_sub_epi64(pos0, p012);
    let neg1 = avx._mm512_sub_epi64(pos1, p012);

    [
        avx._mm512_mask_blend_epi64(sign0, pos0, neg0),
        avx._mm512_mask_blend_epi64(sign1, pos1, neg1),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
fn reconstruct_52bit_01_avx512(simd: crate::Avx512, mod_p0: __m512i, mod_p1: __m512i) -> __m512i {
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

    avx._mm512_mask_blend_epi64(sign, pos, neg)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn reconstruct_slice_32bit_012_avx2(
    simd: crate::Avx2,
    value: &mut [u64],
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
    value: &mut [u64],
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
    value: &mut [u64],
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
    pub fn try_new(n: usize) -> Option<Self> {
        use crate::{primes32::*, _32::Plan};
        Some(Self(
            Plan::try_new(n, P0)?,
            Plan::try_new(n, P1)?,
            Plan::try_new(n, P2)?,
        ))
    }

    pub fn fwd(&self, value: &[u64], mod_p0: &mut [u32], mod_p1: &mut [u32], mod_p2: &mut [u32]) {
        for (value, mod_p0, mod_p1, mod_p2) in
            crate::izip!(value, &mut *mod_p0, &mut *mod_p1, &mut *mod_p2)
        {
            *mod_p0 = (value % crate::primes32::P0 as u64) as u32;
            *mod_p1 = (value % crate::primes32::P1 as u64) as u32;
            *mod_p2 = (value % crate::primes32::P2 as u64) as u32;
        }
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
        self.2.fwd(mod_p2);
    }
    pub fn fwd_binary(
        &self,
        value: &[u64],
        mod_p0: &mut [u32],
        mod_p1: &mut [u32],
        mod_p2: &mut [u32],
    ) {
        for (value, mod_p0, mod_p1, mod_p2) in
            crate::izip!(value, &mut *mod_p0, &mut *mod_p1, &mut *mod_p2)
        {
            *mod_p0 = *value as u32;
            *mod_p1 = *value as u32;
            *mod_p2 = *value as u32;
        }
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
        self.2.fwd(mod_p2);
    }

    pub fn inv(
        &self,
        value: &mut [u64],
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

    pub fn negacyclic_polymul(&self, prod: &mut [u64], lhs: &[u64], rhs_binary: &[u64]) {
        let n = prod.len();
        assert_eq!(n, lhs.len());
        assert_eq!(n, rhs_binary.len());

        let mut lhs0 = avec![0; n];
        let mut lhs1 = avec![0; n];
        let mut lhs2 = avec![0; n];

        let mut rhs0 = avec![0; n];
        let mut rhs1 = avec![0; n];
        let mut rhs2 = avec![0; n];

        self.fwd(lhs, &mut lhs0, &mut lhs1, &mut lhs2);
        self.fwd_binary(rhs_binary, &mut rhs0, &mut rhs1, &mut rhs2);

        self.0.mul_assign_normalize(&mut lhs0, &rhs0);
        self.1.mul_assign_normalize(&mut lhs1, &rhs1);
        self.2.mul_assign_normalize(&mut lhs2, &rhs2);

        self.inv(prod, &mut lhs0, &mut lhs1, &mut lhs2);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
impl Plan52 {
    pub fn try_new(n: usize) -> Option<Self> {
        use crate::{primes52::*, _64::Plan};
        let simd = crate::Avx512::try_new()?;
        Some(Self(Plan::try_new(n, P0)?, Plan::try_new(n, P1)?, simd))
    }

    pub fn fwd(&self, value: &[u64], mod_p0: &mut [u64], mod_p1: &mut [u64]) {
        use crate::primes52::*;
        self.2.vectorize(
            #[inline(always)]
            || {
                for (&value, mod_p0, mod_p1) in crate::izip!(value, &mut *mod_p0, &mut *mod_p1) {
                    *mod_p0 = value % P0;
                    *mod_p1 = value % P1;
                }
            },
        );
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
    }
    pub fn fwd_binary(&self, value: &[u64], mod_p0: &mut [u64], mod_p1: &mut [u64]) {
        self.2.vectorize(
            #[inline(always)]
            || {
                for (&value, mod_p0, mod_p1) in crate::izip!(value, &mut *mod_p0, &mut *mod_p1) {
                    *mod_p0 = value;
                    *mod_p1 = value;
                }
            },
        );
        self.0.fwd(mod_p0);
        self.1.fwd(mod_p1);
    }

    pub fn inv(&self, value: &mut [u64], mod_p0: &mut [u64], mod_p1: &mut [u64]) {
        self.0.inv(mod_p0);
        self.1.inv(mod_p1);

        reconstruct_slice_52bit_01_avx512(self.2, value, mod_p0, mod_p1);
    }

    pub fn negacyclic_polymul(&self, prod: &mut [u64], lhs: &[u64], rhs_binary: &[u64]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random;

    #[test]
    fn reconstruct_32bit() {
        for n in [32, 64, 256, 1024, 2048] {
            let plan = Plan32::try_new(n).unwrap();

            let lhs = (0..n).map(|_| random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| random::<u64>() % 2).collect::<Vec<_>>();
            let mut full_convolution = vec![0u64; 2 * n];
            let mut negacyclic_convolution = vec![0u64; n];
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
                let lhs = (0..n).map(|_| random::<u64>()).collect::<Vec<_>>();
                let rhs = (0..n).map(|_| random::<u64>() % 2).collect::<Vec<_>>();
                let mut full_convolution = vec![0u64; 2 * n];
                let mut negacyclic_convolution = vec![0u64; n];
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
