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
use pulp::{as_arrays, as_arrays_mut, cast};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub fn fwd_breadth_first_avx512(
    simd: Avx512,
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Fn(
        Avx512,
        /* z0 */ __m512i,
        /* z1 */ __m512i,
        /* w */ __m512i,
        /* w_shoup */ __m512i,
        /* p */ __m512i,
        /* neg_p */ __m512i,
        /* two_p */ __m512i,
    ) -> (__m512i, __m512i),
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            let avx = simd.avx512f;
            debug_assert!(n.is_power_of_two());

            let mut t = n;
            let mut m = 1;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;

            let neg_p = avx._mm512_set1_epi64(p.wrapping_neg() as i64);
            let two_p = avx._mm512_set1_epi64((2 * p) as i64);
            let p = avx._mm512_set1_epi64(p as i64);

            while m < n / 8 {
                t /= 2;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<8, _>(z0).0;
                    let z1 = as_arrays_mut::<8, _>(z1).0;
                    let w = avx._mm512_set1_epi64(w as i64);
                    let w_shoup = avx._mm512_set1_epi64(w_shoup as i64);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                        *__z0 = cast(z0);
                        *__z1 = cast(z1);
                    }
                }

                m *= 2;
                w_idx *= 2;
            }

            // m = n / 8
            // t = 4
            {
                let w = as_arrays::<2, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<2, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute4_epu64(*w);
                    let w_shoup = simd.permute4_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave4_epu64(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu64([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 4
            // t = 2
            {
                let w = as_arrays::<4, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<4, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute2_epu64(*w);
                    let w_shoup = simd.permute2_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu64(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z1z1 = cast(simd.interleave2_epu64([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 2
            // t = 1
            {
                let w = as_arrays::<8, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<8, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute1_epu64(*w);
                    let w_shoup = simd.permute1_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu64(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z1 = cast(simd.interleave1_epu64([z0, z1]));
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub fn fwd_breadth_first_avx2(
    simd: Avx2,
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Fn(
        Avx2,
        /* z0 */ __m256i,
        /* z1 */ __m256i,
        /* w */ __m256i,
        /* w_shoup */ __m256i,
        /* p */ __m256i,
        /* neg_p */ __m256i,
        /* two_p */ __m256i,
    ) -> (__m256i, __m256i),
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            let avx = simd.avx;
            debug_assert!(n.is_power_of_two());

            let mut t = n;
            let mut m = 1;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;

            let neg_p = avx._mm256_set1_epi64x(p.wrapping_neg() as i64);
            let two_p = avx._mm256_set1_epi64x((2 * p) as i64);
            let p = avx._mm256_set1_epi64x(p as i64);

            while m < n / 4 {
                t /= 2;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<4, _>(z0).0;
                    let z1 = as_arrays_mut::<4, _>(z1).0;
                    let w = avx._mm256_set1_epi64x(w as i64);
                    let w_shoup = avx._mm256_set1_epi64x(w_shoup as i64);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                        *__z0 = cast(z0);
                        *__z1 = cast(z1);
                    }
                }

                m *= 2;
                w_idx *= 2;
            }

            // m = n / 4
            // t = 2
            {
                let w = as_arrays::<2, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<2, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<4, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute2_epu64(*w);
                    let w_shoup = simd.permute2_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu64(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z1z1 = cast(simd.interleave2_epu64([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 2
            // t = 1
            {
                let w = as_arrays::<4, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<4, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<4, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute1_epu64(*w);
                    let w_shoup = simd.permute1_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu64(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z1 = cast(simd.interleave1_epu64([z0, z1]));
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_breadth_first_scalar(
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Fn(
        /* z0 */ u64,
        /* z1 */ u64,
        /* w */ u64,
        /* w_shoup */ u64,
        /* p */ u64,
        /* neg_p */ u64,
        /* two_p */ u64,
    ) -> (u64, u64),
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = n;
    let mut m = 1;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    let neg_p = p.wrapping_neg();
    let two_p = 2 * p;

    while m < n {
        t /= 2;

        let w = &twid[w_idx..];
        let w_shoup = &twid_shoup[w_idx..];

        for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
            let (z0, z1) = data.split_at_mut(t);
            for (__z0, __z1) in zip(z0, z1) {
                let mut z0 = *__z0;
                let mut z1 = *__z1;
                (z0, z1) = butterfly(z0, z1, w, w_shoup, p, neg_p, two_p);
                *__z0 = z0;
                *__z1 = z1;
            }
        }

        m *= 2;
        w_idx *= 2;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
#[inline(always)]
pub fn inv_breadth_first_avx512(
    simd: Avx512,
    p: u64,
    data: &mut [u64],
    inv_twid: &[u64],
    inv_twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Fn(
        Avx512,
        /* z0 */ __m512i,
        /* z1 */ __m512i,
        /* w */ __m512i,
        /* w_shoup */ __m512i,
        /* p */ __m512i,
        /* neg_p */ __m512i,
        /* two_p */ __m512i,
    ) -> (__m512i, __m512i),
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            let avx = simd.avx512f;
            debug_assert!(n.is_power_of_two());

            let mut t = 1;
            let mut m = n;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;

            let neg_p = avx._mm512_set1_epi64(p.wrapping_neg() as i64);
            let two_p = avx._mm512_set1_epi64((2 * p) as i64);
            let p = avx._mm512_set1_epi64(p as i64);

            // m = n / 2
            // t = 1
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<8, _>(&inv_twid[w_idx..]).0;
                let w_shoup = as_arrays::<8, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute1_epu64(*w);
                    let w_shoup = simd.permute1_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu64(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z1 = cast(simd.interleave1_epu64([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 4
            // t = 2
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<4, _>(&inv_twid[w_idx..]).0;
                let w_shoup = as_arrays::<4, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute2_epu64(*w);
                    let w_shoup = simd.permute2_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu64(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z1z1 = cast(simd.interleave2_epu64([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 8
            // t = 4
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<2, _>(&inv_twid[w_idx..]).0;
                let w_shoup = as_arrays::<2, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute4_epu64(*w);
                    let w_shoup = simd.permute4_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave4_epu64(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu64([z0, z1]));
                }

                t *= 2;
            }

            while m > 1 {
                m /= 2;
                w_idx /= 2;

                let w = &inv_twid[w_idx..];
                let w_shoup = &inv_twid_shoup[w_idx..];

                for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<8, _>(z0).0;
                    let z1 = as_arrays_mut::<8, _>(z1).0;
                    let w = avx._mm512_set1_epi64(w as i64);
                    let w_shoup = avx._mm512_set1_epi64(w_shoup as i64);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
#[inline(always)]
pub fn inv_breadth_first_avx2(
    simd: Avx2,
    p: u64,
    data: &mut [u64],
    inv_twid: &[u64],
    inv_twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Fn(
        Avx2,
        /* z0 */ __m256i,
        /* z1 */ __m256i,
        /* w */ __m256i,
        /* w_shoup */ __m256i,
        /* p */ __m256i,
        /* neg_p */ __m256i,
        /* two_p */ __m256i,
    ) -> (__m256i, __m256i),
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let n = data.len();
            let avx = simd.avx;
            debug_assert!(n.is_power_of_two());

            let mut t = 1;
            let mut m = n;
            let mut w_idx = (m << recursion_depth) + recursion_half * m;

            let neg_p = avx._mm256_set1_epi64x(p.wrapping_neg() as i64);
            let two_p = avx._mm256_set1_epi64x((2 * p) as i64);
            let p = avx._mm256_set1_epi64x(p as i64);

            // m = n / 2
            // t = 1
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<4, _>(&inv_twid[w_idx..]).0;
                let w_shoup = as_arrays::<4, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<4, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute1_epu64(*w);
                    let w_shoup = simd.permute1_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu64(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z1 = cast(simd.interleave1_epu64([z0, z1]));
                }

                t *= 2;
            }

            // m = n / 4
            // t = 2
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<2, _>(&inv_twid[w_idx..]).0;
                let w_shoup = as_arrays::<2, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<4, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute2_epu64(*w);
                    let w_shoup = simd.permute2_epu64(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu64(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z1z1 = cast(simd.interleave2_epu64([z0, z1]));
                }

                t *= 2;
            }

            while m > 1 {
                m /= 2;
                w_idx /= 2;

                let w = &inv_twid[w_idx..];
                let w_shoup = &inv_twid_shoup[w_idx..];

                for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<4, _>(z0).0;
                    let z1 = as_arrays_mut::<4, _>(z1).0;
                    let w = avx._mm256_set1_epi64x(w as i64);
                    let w_shoup = avx._mm256_set1_epi64x(w_shoup as i64);

                    for (__z0, __z1) in zip(z0, z1) {
                        let mut z0 = cast(*__z0);
                        let mut z1 = cast(*__z1);
                        (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
pub fn inv_breadth_first_scalar(
    p: u64,
    data: &mut [u64],
    twid: &[u64],
    twid_shoup: &[u64],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Fn(
        /* z0 */ u64,
        /* z1 */ u64,
        /* w */ u64,
        /* w_shoup */ u64,
        /* p */ u64,
        /* neg_p */ u64,
        /* two_p */ u64,
    ) -> (u64, u64),
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut t = 1;
    let mut m = n;
    let mut w_idx = (m << recursion_depth) + recursion_half * m;

    let neg_p = p.wrapping_neg();
    let two_p = 2 * p;

    while m > 1 {
        m /= 2;
        w_idx /= 2;

        let w = &twid[w_idx..];
        let w_shoup = &twid_shoup[w_idx..];

        for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
            let (z0, z1) = data.split_at_mut(t);
            for (__z0, __z1) in zip(z0, z1) {
                let mut z0 = *__z0;
                let mut z1 = *__z1;
                (z0, z1) = butterfly(z0, z1, w, w_shoup, p, neg_p, two_p);
                *__z0 = z0;
                *__z1 = z1;
            }
        }

        t *= 2;
    }
}