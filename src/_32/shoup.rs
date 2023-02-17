use super::RECURSION_THRESHOLD;
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
pub fn fwd_breadth_first_avx512(
    simd: Avx512,
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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

            let neg_p = avx._mm512_set1_epi32(p.wrapping_neg() as i32);
            let two_p = avx._mm512_set1_epi32((2 * p) as i32);
            let p = avx._mm512_set1_epi32(p as i32);

            while m < n / 16 {
                t /= 2;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<16, _>(z0).0;
                    let z1 = as_arrays_mut::<16, _>(z1).0;
                    let w = avx._mm512_set1_epi32(w as i32);
                    let w_shoup = avx._mm512_set1_epi32(w_shoup as i32);

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

            // m = n / 16
            // t = 8
            {
                let w = as_arrays::<2, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<2, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute8_epu32(*w);
                    let w_shoup = simd.permute8_epu32(*w_shoup);
                    let [mut z0, mut z1] =
                        simd.interleave8_epu32(cast(*z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1 = cast(simd.interleave8_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 8
            // t = 4
            {
                let w = as_arrays::<4, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<4, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute4_epu32(*w);
                    let w_shoup = simd.permute4_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 4
            // t = 2
            {
                let w = as_arrays::<8, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<8, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute2_epu32(*w);
                    let w_shoup = simd.permute2_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z1z1 = cast(simd.interleave2_epu32([z0, z1]));
                }

                w_idx *= 2;
            }

            // m = n / 2
            // t = 1
            {
                let w = as_arrays::<16, _>(&twid[w_idx..]).0;
                let w_shoup = as_arrays::<16, _>(&twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute1_epu32(*w);
                    let w_shoup = simd.permute1_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                fwd_breadth_first_avx512(
                    simd,
                    p,
                    data,
                    twid,
                    twid_shoup,
                    recursion_depth,
                    recursion_half,
                    butterfly,
                );
            } else {
                let t = n / 2;
                let m = 1;
                let w_idx = (m << recursion_depth) + m * recursion_half;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                {
                    let avx = simd.avx512f;
                    let neg_p = avx._mm512_set1_epi32(p.wrapping_neg() as i32);
                    let two_p = avx._mm512_set1_epi32((2 * p) as i32);
                    let p = avx._mm512_set1_epi32(p as i32);

                    for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup))
                    {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<16, _>(z0).0;
                        let z1 = as_arrays_mut::<16, _>(z1).0;
                        let w = avx._mm512_set1_epi32(w as i32);
                        let w_shoup = avx._mm512_set1_epi32(w_shoup as i32);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                            *__z0 = cast(z0);
                            *__z1 = cast(z1);
                        }
                    }
                }

                let (data0, data1) = data.split_at_mut(n / 2);
                fwd_depth_first_avx512(
                    simd,
                    p,
                    data0,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2,
                    butterfly,
                );
                fwd_depth_first_avx512(
                    simd,
                    p,
                    data1,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                    butterfly,
                );
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_breadth_first_avx2(
    simd: Avx2,
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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

            let neg_p = avx._mm256_set1_epi32(p.wrapping_neg() as i32);
            let two_p = avx._mm256_set1_epi32((2 * p) as i32);
            let p = avx._mm256_set1_epi32(p as i32);

            while m < n / 8 {
                t /= 2;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                    let (z0, z1) = data.split_at_mut(t);
                    let z0 = as_arrays_mut::<8, _>(z0).0;
                    let z1 = as_arrays_mut::<8, _>(z1).0;
                    let w = avx._mm256_set1_epi32(w as i32);
                    let w_shoup = avx._mm256_set1_epi32(w_shoup as i32);

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
                    let w = simd.permute4_epu32(*w);
                    let w_shoup = simd.permute4_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu32([z0, z1]));
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
                    let w = simd.permute2_epu32(*w);
                    let w_shoup = simd.permute2_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z1z1 = cast(simd.interleave2_epu32([z0, z1]));
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
                    let w = simd.permute1_epu32(*w);
                    let w_shoup = simd.permute1_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z1 = cast(simd.interleave1_epu32([z0, z1]));
                }
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_depth_first_avx2(
    simd: Avx2,
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                fwd_breadth_first_avx2(
                    simd,
                    p,
                    data,
                    twid,
                    twid_shoup,
                    recursion_depth,
                    recursion_half,
                    butterfly,
                );
            } else {
                let t = n / 2;
                let m = 1;
                let w_idx = (m << recursion_depth) + m * recursion_half;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                {
                    let avx = simd.avx;
                    let neg_p = avx._mm256_set1_epi32(p.wrapping_neg() as i32);
                    let two_p = avx._mm256_set1_epi32((2 * p) as i32);
                    let p = avx._mm256_set1_epi32(p as i32);

                    for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup))
                    {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<8, _>(z0).0;
                        let z1 = as_arrays_mut::<8, _>(z1).0;
                        let w = avx._mm256_set1_epi32(w as i32);
                        let w_shoup = avx._mm256_set1_epi32(w_shoup as i32);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                            *__z0 = cast(z0);
                            *__z1 = cast(z1);
                        }
                    }
                }

                let (data0, data1) = data.split_at_mut(n / 2);
                fwd_depth_first_avx2(
                    simd,
                    p,
                    data0,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2,
                    butterfly,
                );
                fwd_depth_first_avx2(
                    simd,
                    p,
                    data1,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                    butterfly,
                );
            }
        },
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn fwd_breadth_first_scalar(
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
            /* z0 */ u32,
            /* z1 */ u32,
            /* w */ u32,
            /* w_shoup */ u32,
            /* p */ u32,
            /* neg_p */ u32,
            /* two_p */ u32,
        ) -> (u32, u32),
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

pub fn fwd_depth_first_scalar(
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
            /* z0 */ u32,
            /* z1 */ u32,
            /* w */ u32,
            /* w_shoup */ u32,
            /* p */ u32,
            /* neg_p */ u32,
            /* two_p */ u32,
        ) -> (u32, u32),
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    if n <= RECURSION_THRESHOLD {
        fwd_breadth_first_scalar(
            p,
            data,
            twid,
            twid_shoup,
            recursion_depth,
            recursion_half,
            butterfly,
        );
    } else {
        let t = n / 2;
        let m = 1;
        let w_idx = (m << recursion_depth) + m * recursion_half;

        let w = &twid[w_idx..];
        let w_shoup = &twid_shoup[w_idx..];

        {
            let neg_p = p.wrapping_neg();
            let two_p = 2 * p;

            for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                let (z0, z1) = data.split_at_mut(t);

                for (__z0, __z1) in zip(z0, z1) {
                    let mut z0 = cast(*__z0);
                    let mut z1 = cast(*__z1);
                    (z0, z1) = butterfly(z0, z1, w, w_shoup, p, neg_p, two_p);
                    *__z0 = cast(z0);
                    *__z1 = cast(z1);
                }
            }
        }

        let (data0, data1) = data.split_at_mut(n / 2);
        fwd_depth_first_scalar(
            p,
            data0,
            twid,
            twid_shoup,
            recursion_depth + 1,
            recursion_half * 2,
            butterfly,
        );
        fwd_depth_first_scalar(
            p,
            data1,
            twid,
            twid_shoup,
            recursion_depth + 1,
            recursion_half * 2 + 1,
            butterfly,
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly")]
pub fn inv_breadth_first_avx512(
    simd: Avx512,
    p: u32,
    data: &mut [u32],
    inv_twid: &[u32],
    inv_twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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

            let neg_p = avx._mm512_set1_epi32(p.wrapping_neg() as i32);
            let two_p = avx._mm512_set1_epi32((2 * p) as i32);
            let p = avx._mm512_set1_epi32(p as i32);

            // m = n / 2
            // t = 1
            {
                m /= 2;
                w_idx /= 2;

                let w = as_arrays::<16, _>(&inv_twid[w_idx..]).0;
                let w_shoup = as_arrays::<16, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute1_epu32(*w);
                    let w_shoup = simd.permute1_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
                let w_shoup = as_arrays::<8, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute2_epu32(*w);
                    let w_shoup = simd.permute2_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
                let w_shoup = as_arrays::<4, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute4_epu32(*w);
                    let w_shoup = simd.permute4_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
                let w_shoup = as_arrays::<2, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<16, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute8_epu32(*w);
                    let w_shoup = simd.permute8_epu32(*w_shoup);
                    let [mut z0, mut z1] =
                        simd.interleave8_epu32(cast(*z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z0z0z0z0z0z0z1z1z1z1z1z1z1z1 = cast(simd.interleave8_epu32([z0, z1]));
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
                    let z0 = as_arrays_mut::<16, _>(z0).0;
                    let z1 = as_arrays_mut::<16, _>(z1).0;
                    let w = avx._mm512_set1_epi32(w as i32);
                    let w_shoup = avx._mm512_set1_epi32(w_shoup as i32);

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
#[cfg(feature = "nightly")]
pub fn inv_depth_first_avx512(
    simd: Avx512,
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                inv_breadth_first_avx512(
                    simd,
                    p,
                    data,
                    twid,
                    twid_shoup,
                    recursion_depth,
                    recursion_half,
                    butterfly,
                );
            } else {
                let (data0, data1) = data.split_at_mut(n / 2);
                inv_depth_first_avx512(
                    simd,
                    p,
                    data0,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2,
                    butterfly,
                );
                inv_depth_first_avx512(
                    simd,
                    p,
                    data1,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                    butterfly,
                );

                let t = n / 2;
                let m = 1;
                let w_idx = (m << recursion_depth) + m * recursion_half;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                {
                    let avx = simd.avx512f;
                    let neg_p = avx._mm512_set1_epi32(p.wrapping_neg() as i32);
                    let two_p = avx._mm512_set1_epi32((2 * p) as i32);
                    let p = avx._mm512_set1_epi32(p as i32);

                    for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup))
                    {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<16, _>(z0).0;
                        let z1 = as_arrays_mut::<16, _>(z1).0;
                        let w = avx._mm512_set1_epi32(w as i32);
                        let w_shoup = avx._mm512_set1_epi32(w_shoup as i32);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
pub fn inv_breadth_first_avx2(
    simd: Avx2,
    p: u32,
    data: &mut [u32],
    inv_twid: &[u32],
    inv_twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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

            let neg_p = avx._mm256_set1_epi32(p.wrapping_neg() as i32);
            let two_p = avx._mm256_set1_epi32((2 * p) as i32);
            let p = avx._mm256_set1_epi32(p as i32);

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
                    let w = simd.permute1_epu32(*w);
                    let w_shoup = simd.permute1_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave1_epu32(cast(*z0z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
                let w_shoup = as_arrays::<4, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute2_epu32(*w);
                    let w_shoup = simd.permute2_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave2_epu32(cast(*z0z0z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
                let w_shoup = as_arrays::<2, _>(&inv_twid_shoup[w_idx..]).0;
                let data = as_arrays_mut::<8, _>(data).0;
                let data = as_arrays_mut::<2, _>(data).0;

                for (z0z0z0z0z1z1z1z1, (w, w_shoup)) in zip(data, zip(w, w_shoup)) {
                    let w = simd.permute4_epu32(*w);
                    let w_shoup = simd.permute4_epu32(*w_shoup);
                    let [mut z0, mut z1] = simd.interleave4_epu32(cast(*z0z0z0z0z1z1z1z1));
                    (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
                    *z0z0z0z0z1z1z1z1 = cast(simd.interleave4_epu32([z0, z1]));
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
                    let w = avx._mm256_set1_epi32(w as i32);
                    let w_shoup = avx._mm256_set1_epi32(w_shoup as i32);

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
pub fn inv_depth_first_avx2(
    simd: Avx2,
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
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
            debug_assert!(n.is_power_of_two());

            if n <= RECURSION_THRESHOLD {
                inv_breadth_first_avx2(
                    simd,
                    p,
                    data,
                    twid,
                    twid_shoup,
                    recursion_depth,
                    recursion_half,
                    butterfly,
                );
            } else {
                let (data0, data1) = data.split_at_mut(n / 2);
                inv_depth_first_avx2(
                    simd,
                    p,
                    data0,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2,
                    butterfly,
                );
                inv_depth_first_avx2(
                    simd,
                    p,
                    data1,
                    twid,
                    twid_shoup,
                    recursion_depth + 1,
                    recursion_half * 2 + 1,
                    butterfly,
                );

                let t = n / 2;
                let m = 1;
                let w_idx = (m << recursion_depth) + m * recursion_half;

                let w = &twid[w_idx..];
                let w_shoup = &twid_shoup[w_idx..];

                {
                    let avx = simd.avx;
                    let neg_p = avx._mm256_set1_epi32(p.wrapping_neg() as i32);
                    let two_p = avx._mm256_set1_epi32((2 * p) as i32);
                    let p = avx._mm256_set1_epi32(p as i32);

                    for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup))
                    {
                        let (z0, z1) = data.split_at_mut(t);
                        let z0 = as_arrays_mut::<8, _>(z0).0;
                        let z1 = as_arrays_mut::<8, _>(z1).0;
                        let w = avx._mm256_set1_epi32(w as i32);
                        let w_shoup = avx._mm256_set1_epi32(w_shoup as i32);

                        for (__z0, __z1) in zip(z0, z1) {
                            let mut z0 = cast(*__z0);
                            let mut z1 = cast(*__z1);
                            (z0, z1) = butterfly(simd, z0, z1, w, w_shoup, p, neg_p, two_p);
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
pub fn inv_breadth_first_scalar(
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
            /* z0 */ u32,
            /* z1 */ u32,
            /* w */ u32,
            /* w_shoup */ u32,
            /* p */ u32,
            /* neg_p */ u32,
            /* two_p */ u32,
        ) -> (u32, u32),
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

pub fn inv_depth_first_scalar(
    p: u32,
    data: &mut [u32],
    twid: &[u32],
    twid_shoup: &[u32],
    recursion_depth: usize,
    recursion_half: usize,
    butterfly: impl Copy
        + Fn(
            /* z0 */ u32,
            /* z1 */ u32,
            /* w */ u32,
            /* w_shoup */ u32,
            /* p */ u32,
            /* neg_p */ u32,
            /* two_p */ u32,
        ) -> (u32, u32),
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    if n <= RECURSION_THRESHOLD {
        inv_breadth_first_scalar(
            p,
            data,
            twid,
            twid_shoup,
            recursion_depth,
            recursion_half,
            butterfly,
        );
    } else {
        let (data0, data1) = data.split_at_mut(n / 2);
        inv_depth_first_scalar(
            p,
            data0,
            twid,
            twid_shoup,
            recursion_depth + 1,
            recursion_half * 2,
            butterfly,
        );
        inv_depth_first_scalar(
            p,
            data1,
            twid,
            twid_shoup,
            recursion_depth + 1,
            recursion_half * 2 + 1,
            butterfly,
        );

        let t = n / 2;
        let m = 1;
        let w_idx = (m << recursion_depth) + m * recursion_half;

        let w = &twid[w_idx..];
        let w_shoup = &twid_shoup[w_idx..];

        {
            let neg_p = p.wrapping_neg();
            let two_p = 2 * p;

            for (data, (&w, &w_shoup)) in zip(data.chunks_exact_mut(2 * t), zip(w, w_shoup)) {
                let (z0, z1) = data.split_at_mut(t);

                for (__z0, __z1) in zip(z0, z1) {
                    let mut z0 = cast(*__z0);
                    let mut z1 = cast(*__z1);
                    (z0, z1) = butterfly(z0, z1, w, w_shoup, p, neg_p, two_p);
                    *__z0 = cast(z0);
                    *__z1 = cast(z1);
                }
            }
        }
    }
}
