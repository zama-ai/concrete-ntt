#![allow(dead_code, unused_macros)]

mod prototype;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    use crate::{
        izip,
        prime64::{generic_solinas::PrimeModulusV3, Solinas},
        solinas::prototype::{OMEGA_64_SHL_AMOUNT, P, R96},
    };

    use crate::V3;
    use pulp::{i32x4, i32x8, i64x4, m32, m32x8, u32x8, u64x2, u64x4};

    #[cfg(feature = "nightly")]
    use crate::V4;
    #[cfg(feature = "nightly")]
    use pulp::{i32x16, i64x8, u32x16, u64x8};

    const fn u32x8_from_array(data: [u32; 8]) -> u32x8 {
        let u = data;
        u32x8(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7])
    }

    const fn u32_max(a: u32, b: u32) -> u32 {
        if a > b {
            a
        } else {
            b
        }
    }

    #[inline]
    fn split_2_mut<T>(slice: &mut [T]) -> (&mut [T], &mut [T]) {
        let mid = slice.len() / 2;
        slice.split_at_mut(mid)
    }

    #[inline]
    fn split_4_mut<T>(slice: &mut [T]) -> (&mut [T], &mut [T], &mut [T], &mut [T]) {
        let (x01, x23) = split_2_mut(slice);
        let (x0, x1) = split_2_mut(x01);
        let (x2, x3) = split_2_mut(x23);
        (x0, x1, x2, x3)
    }

    #[inline]
    fn split_8_mut<T>(
        slice: &mut [T],
    ) -> (
        &mut [T],
        &mut [T],
        &mut [T],
        &mut [T],
        &mut [T],
        &mut [T],
        &mut [T],
        &mut [T],
    ) {
        let (x01, x23, x45, x67) = split_4_mut(slice);
        let (x0, x1) = split_2_mut(x01);
        let (x2, x3) = split_2_mut(x23);
        let (x4, x5) = split_2_mut(x45);
        let (x6, x7) = split_2_mut(x67);
        (x0, x1, x2, x3, x4, x5, x6, x7)
    }

    #[derive(Copy, Clone, Debug)]
    struct Wrapper<T, U>(T, U);

    impl<T, U> core::ops::Deref for Wrapper<T, U> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    trait Apply<Value>: Copy {
        type Output;

        fn apply(self, value: Value) -> Self::Output;
        #[doc(hidden)]
        fn __opaque(self) -> impl Copy + FnOnce(Value) -> Self::Output {
            #[inline(always)]
            move |value| self.apply(value)
        }
    }

    impl<Value, T: Copy, U: Apply<Value>> Apply<Value> for Wrapper<T, U> {
        type Output = U::Output;

        #[inline(always)]
        fn apply(self, value: Value) -> Self::Output {
            self.1.apply(value)
        }
    }

    macro_rules! shl_const_new {
        ($simd: expr, $amount: expr $(,)?) => {{
            use $crate::solinas::x86::Apply;

            #[derive(Copy, Clone, Debug)]
            struct ShlConst<S> {
                __inner: S,
            }

            impl Apply<$crate::solinas::x86::i32x4x2> for ShlConst<$crate::V3> {
                type Output = $crate::solinas::x86::i32x4x2;

                #[inline(always)]
                fn apply(
                    self,
                    value: $crate::solinas::x86::i32x4x2,
                ) -> $crate::solinas::x86::i32x4x2 {
                    const AMOUNT: i32 = $amount;

                    self.__inner
                        .shl_const_i32x4x2::<AMOUNT, { AMOUNT % 24 }, { 24 - AMOUNT % 24 }>(value)
                }
            }

            let shl = ShlConst { __inner: $simd };
            $crate::solinas::x86::Wrapper(shl.__opaque(), shl)
        }};
    }

    macro_rules! shl_const {
        ($simd: expr, $amount: expr $(,)?) => {{
            use $crate::solinas::x86::Apply;

            #[derive(Copy, Clone, Debug)]
            struct ShlConst<S> {
                __inner: S,
            }

            impl Apply<$crate::solinas::x86::R96x8> for ShlConst<$crate::V3> {
                type Output = $crate::solinas::x86::R96x8;

                #[inline(always)]
                fn apply(self, value: $crate::solinas::x86::R96x8) -> $crate::solinas::x86::R96x8 {
                    const AMOUNT: i32 = $amount;

                    self.__inner
                        .shl_const_r96x8::<AMOUNT, { AMOUNT % 24 }, { 24 - AMOUNT % 24 }>(value)
                }
            }

            #[cfg(feature = "nightly")]
            impl Apply<$crate::solinas::x86::R96x16> for ShlConst<$crate::V4> {
                type Output = $crate::solinas::x86::R96x16;

                #[inline(always)]
                fn apply(
                    self,
                    value: $crate::solinas::x86::R96x16,
                ) -> $crate::solinas::x86::R96x16 {
                    const AMOUNT: u32 = $amount;

                    self.__inner
                        .shl_const_r96x16::<AMOUNT, { AMOUNT % 24 }, { 24 - AMOUNT % 24 }>(value)
                }
            }

            let shl = ShlConst { __inner: $simd };
            $crate::solinas::x86::Wrapper(shl.__opaque(), shl)
        }};
    }

    macro_rules! mul_by_w8 {
        ($simd: expr, $pow: expr $(,)?) => {
            shl_const!(
                $simd,
                ($pow * crate::solinas::prototype::OMEGA_8_SHL_AMOUNT) as _
            )
        };
    }

    macro_rules! mul_by_w64 {
        ($simd: expr, $pow: expr $(,)?) => {
            shl_const!(
                $simd,
                ($pow * crate::solinas::prototype::OMEGA_64_SHL_AMOUNT) as _
            )
        };
    }

    macro_rules! mul_by_w8_new {
        ($simd: expr, $pow: expr $(,)?) => {
            shl_const_new!(
                $simd,
                ($pow * crate::solinas::prototype::OMEGA_8_SHL_AMOUNT) as _
            )
        };
    }

    macro_rules! mul_by_w64_new {
        ($simd: expr, $pow: expr $(,)?) => {
            shl_const_new!(
                $simd,
                ($pow * crate::solinas::prototype::OMEGA_64_SHL_AMOUNT) as _
            )
        };
    }

    #[inline]
    fn max_u32s(x: &[u32]) -> u32 {
        x.iter().copied().max().unwrap_or(0)
    }

    #[derive(Copy, Clone, Debug)]
    pub struct R96x8 {
        pub x: i32x8,
        pub y: i32x8,
        pub z: i32x8,
        pub w: i32x8,
    }

    #[allow(non_camel_case_types)]
    pub type i32x4x2 = [i32x4; 2];

    impl R96x8 {
        #[inline(always)]
        pub fn new(x: i32x8, y: i32x8, z: i32x8, w: i32x8) -> Self {
            Self { x, y, z, w }
        }
    }

    #[cfg(feature = "nightly")]
    #[derive(Copy, Clone, Debug)]
    pub struct R96x16 {
        pub x: i32x16,
        pub y: i32x16,
        pub z: i32x16,
        pub w: i32x16,
    }

    impl PartialEq for R96x8 {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            let R96x8 { x, y, z, w } = *self;
            let lhs_x: [i32; 8] = pulp::cast(x);
            let lhs_y: [i32; 8] = pulp::cast(y);
            let lhs_z: [i32; 8] = pulp::cast(z);
            let lhs_w: [i32; 8] = pulp::cast(w);

            let R96x8 { x, y, z, w } = *other;
            let rhs_x: [i32; 8] = pulp::cast(x);
            let rhs_y: [i32; 8] = pulp::cast(y);
            let rhs_z: [i32; 8] = pulp::cast(z);
            let rhs_w: [i32; 8] = pulp::cast(w);

            let idx = [0, 1, 2, 3, 4, 5, 6, 7usize];

            let lhs = idx.map(|i| R96 {
                x: lhs_x[i],
                y: lhs_y[i],
                z: lhs_z[i],
                w: lhs_w[i],
                available_bits: 0,
            });

            let rhs = idx.map(|i| R96 {
                x: rhs_x[i],
                y: rhs_y[i],
                z: rhs_z[i],
                w: rhs_w[i],
                available_bits: 0,
            });

            lhs == rhs
        }
    }

    impl V3 {
        const ZERO: i32x8 = i32x8(0, 0, 0, 0, 0, 0, 0, 0);
        const MASK: i32x8 = i32x8(
            0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF,
        );

        #[inline(always)]
        pub fn i32x4x2_from_zpx2(self, value: u64x2) -> i32x4x2 {
            let value = pulp::cast([value, value]);
            // aabb
            let value: u64x4 = pulp::cast(self.avx._mm256_shuffle_pd::<0b_1100>(value, value));

            let xz: i32x8 = pulp::cast(self.and_u64x4(
                self.shr_dyn_u64x4(value, u64x4(0, 48, 0, 48)),
                self.splat_u64x4(0xFFFFFF),
            ));
            let y: i32x8 = pulp::cast(self.and_u64x4(
                self.shl_const_u64x4::<8>(value),
                u64x4(0xFFFFFF << 32, 0, 0xFFFFFF << 32, 0),
            ));

            pulp::cast(self.or_i32x8(xz, y))
        }

        #[inline(always)]
        pub fn i32x4x4_from_interleaved_zpx4(self, value: u64x4) -> [i32x4x2; 2] {
            let value = pulp::cast(value);
            // aabb
            let value0: u64x4 = pulp::cast(self.avx._mm256_shuffle_pd::<0b_0000>(value, value));
            let value1: u64x4 = pulp::cast(self.avx._mm256_shuffle_pd::<0b_1111>(value, value));

            let xz0: i32x8 = pulp::cast(self.and_u64x4(
                self.shr_dyn_u64x4(value0, u64x4(0, 48, 0, 48)),
                self.splat_u64x4(0xFFFFFF),
            ));
            let xz1: i32x8 = pulp::cast(self.and_u64x4(
                self.shr_dyn_u64x4(value1, u64x4(0, 48, 0, 48)),
                self.splat_u64x4(0xFFFFFF),
            ));
            let y0: i32x8 = pulp::cast(self.and_u64x4(
                self.shl_const_u64x4::<8>(value0),
                u64x4(0xFFFFFF << 32, 0, 0xFFFFFF << 32, 0),
            ));
            let y1: i32x8 = pulp::cast(self.and_u64x4(
                self.shl_const_u64x4::<8>(value1),
                u64x4(0xFFFFFF << 32, 0, 0xFFFFFF << 32, 0),
            ));

            [
                pulp::cast(self.or_i32x8(xz0, y0)),
                pulp::cast(self.or_i32x8(xz1, y1)),
            ]
        }

        #[inline(always)]
        pub fn interleaved_zpx4_from_i32x4x4(self, value: [i32x4x2; 2]) -> u64x4 {
            let value0: u64x4 = pulp::cast(value[0]);
            let value1: u64x4 = pulp::cast(value[1]);

            let x0: i64x4;
            let y0: i64x4;
            let z0: i64x4;
            let w0: i64x4;

            let x1: i64x4;
            let y1: i64x4;
            let z1: i64x4;
            let w1: i64x4;

            {
                let xz = self.and_u64x4(value0, self.splat_u64x4(0xFFFF_FFFF));
                let xz_hi = self.shl_const_u64x4::<32>(xz);
                let xz_hi_sign = pulp::cast(self.shr_const_i32x8::<32>(pulp::cast(xz_hi)));
                let xz = self.or_u64x4(xz, xz_hi_sign);

                let yw = self.shr_const_u64x4::<32>(value0);
                let yw_hi = self.shl_const_u64x4::<32>(yw);
                let yw_hi_sign = pulp::cast(self.shr_const_i32x8::<32>(pulp::cast(yw_hi)));
                let yw = self.or_u64x4(yw, yw_hi_sign);

                x0 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b0000>(pulp::cast(xz), pulp::cast(xz)),
                );
                z0 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b1111>(pulp::cast(xz), pulp::cast(xz)),
                );
                y0 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b0000>(pulp::cast(yw), pulp::cast(yw)),
                );
                w0 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b1111>(pulp::cast(yw), pulp::cast(yw)),
                );
            }

            {
                let xz = self.and_u64x4(value1, self.splat_u64x4(0xFFFF_FFFF));
                let xz_hi = self.shl_const_u64x4::<32>(xz);
                let xz_hi_sign = pulp::cast(self.shr_const_i32x8::<32>(pulp::cast(xz_hi)));
                let xz = self.or_u64x4(xz, xz_hi_sign);

                let yw = self.shr_const_u64x4::<32>(value1);
                let yw_hi = self.shl_const_u64x4::<32>(yw);
                let yw_hi_sign = pulp::cast(self.shr_const_i32x8::<32>(pulp::cast(yw_hi)));
                let yw = self.or_u64x4(yw, yw_hi_sign);

                x1 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b0000>(pulp::cast(xz), pulp::cast(xz)),
                );
                z1 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b1111>(pulp::cast(xz), pulp::cast(xz)),
                );
                y1 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b0000>(pulp::cast(yw), pulp::cast(yw)),
                );
                w1 = pulp::cast(
                    self.avx
                        ._mm256_shuffle_pd::<0b1111>(pulp::cast(yw), pulp::cast(yw)),
                );
            }

            let x: i64x4 = pulp::cast(
                self.avx2
                    ._mm256_unpacklo_epi64(pulp::cast(x0), pulp::cast(x1)),
            );
            let y: i64x4 = pulp::cast(
                self.avx2
                    ._mm256_unpacklo_epi64(pulp::cast(y0), pulp::cast(y1)),
            );
            let z: i64x4 = pulp::cast(
                self.avx2
                    ._mm256_unpacklo_epi64(pulp::cast(z0), pulp::cast(z1)),
            );
            let w: i64x4 = pulp::cast(
                self.avx2
                    ._mm256_unpacklo_epi64(pulp::cast(w0), pulp::cast(w1)),
            );

            let x = x;
            let y = self.shl_const_i64x4::<24>(y);
            let w0 = self.and_i64x4(self.shl_const_i64x4::<8>(w), self.splat_i64x4(0xFFFF_FFFF));
            let w1 = self.or_i64x4(
                pulp::cast(self.shr_const_u64x4::<24>(pulp::cast(w))),
                self.and_i64x4(self.splat_i64x4(0xFFFF_FFFF << 32), w),
            );
            let z0 = self.shl_const_i64x4::<48>(self.and_i64x4(z, self.splat_i64x4(0xFFFF)));
            let z1 = self.or_i64x4(
                pulp::cast(self.shr_const_u64x4::<16>(pulp::cast(z))),
                self.and_i64x4(self.splat_i64x4(0xFFFF_FFFF << 32), z),
            );

            #[inline(always)]
            fn add_p_if_negative(simd: V3, a: i64x4) -> u64x4 {
                let p: i64x4 = pulp::cast(const { [P; 4] });

                pulp::cast(simd.avx._mm256_blendv_pd(
                    pulp::cast(a),
                    pulp::cast(simd.wrapping_add_i64x4(a, p)),
                    pulp::cast(a),
                ))
            }

            let a = self.wrapping_sub_i64x4(self.wrapping_add_i64x4(x, y), w1);
            let a = add_p_if_negative(self, a);
            let b: u64x4 = pulp::cast(z0);

            let c = self.wrapping_sub_i64x4(self.shl_const_i64x4::<32>(z1), z1);
            let c = add_p_if_negative(self, c);

            let d: u64x4 = pulp::cast(self.wrapping_sub_i64x4(self.shl_const_i64x4::<32>(w0), w0));

            let p: u64x4 = pulp::cast(const { [P; 4] });

            let ab = self.wrapping_add_u64x4(a, b);
            let o: u64x4 = pulp::cast(self.cmp_lt_u64x4(ab, a));
            let ab = self.wrapping_sub_u64x4(ab, self.and_u64x4(o, p));

            let cd = self.wrapping_add_u64x4(c, d);
            let o: u64x4 = pulp::cast(self.cmp_lt_u64x4(cd, c));
            let cd = self.wrapping_sub_u64x4(cd, self.and_u64x4(o, p));

            let abcd = self.wrapping_add_u64x4(ab, cd);
            let o: u64x4 = pulp::cast(self.cmp_lt_u64x4(abcd, ab));
            let abcd = self.wrapping_sub_u64x4(abcd, self.and_u64x4(o, p));

            let too_big: u64x4 = pulp::cast(self.cmp_ge_u64x4(abcd, p));
            let abcd = self.wrapping_sub_u64x4(abcd, self.and_u64x4(too_big, p));

            abcd
        }

        #[inline(always)]
        pub fn add_i32x4x2(self, lhs: i32x4x2, rhs: i32x4x2) -> i32x4x2 {
            pulp::cast(self.wrapping_add_i32x8(pulp::cast(lhs), pulp::cast(rhs)))
        }

        #[inline(always)]
        pub fn sub_i32x4x2(self, lhs: i32x4x2, rhs: i32x4x2) -> i32x4x2 {
            pulp::cast(self.wrapping_sub_i32x8(pulp::cast(lhs), pulp::cast(rhs)))
        }

        #[inline(always)]
        pub fn propagate_carries_i32x4x2(self, value: i32x4x2) -> i32x4x2 {
            let value = pulp::cast(value);
            let lo = self.and_i32x8(value, self.splat_i32x8(0xFFFFFF));
            let hi = self.shr_const_i32x8::<24>(value);
            let hi = self.wrapping_sub_i32x8(
                pulp::cast(self.avx2._mm256_bslli_epi128::<4>(pulp::cast(hi))),
                pulp::cast(self.avx2._mm256_bsrli_epi128::<12>(pulp::cast(hi))),
            );
            pulp::cast(self.wrapping_add_i32x8(lo, hi))
        }

        #[inline(always)]
        pub fn shl_const_i32x4x2<
            const AMOUNT: i32,
            const AMOUNT_MOD_24: i32,
            const TWENTY_FOUR_MINUS_AMOUNT_MOD_24: i32,
        >(
            self,
            value: i32x4x2,
        ) -> i32x4x2 {
            const {
                assert!(AMOUNT >= 0);
                assert!(AMOUNT_MOD_24 == AMOUNT % 24);
                assert!(TWENTY_FOUR_MINUS_AMOUNT_MOD_24 == 24 - AMOUNT_MOD_24);
            };

            let mut value = pulp::cast(value);

            match const { (AMOUNT % 192) / 24 } {
                0 => {}
                1 => {
                    value = self.wrapping_sub_i32x8(
                        pulp::cast(self.avx2._mm256_bslli_epi128::<4>(pulp::cast(value))),
                        pulp::cast(self.avx2._mm256_bsrli_epi128::<12>(pulp::cast(value))),
                    );
                }
                2 => {
                    value = self.wrapping_sub_i32x8(
                        pulp::cast(self.avx2._mm256_bslli_epi128::<8>(pulp::cast(value))),
                        pulp::cast(self.avx2._mm256_bsrli_epi128::<8>(pulp::cast(value))),
                    );
                }
                3 => {
                    value = self.wrapping_sub_i32x8(
                        pulp::cast(self.avx2._mm256_bslli_epi128::<12>(pulp::cast(value))),
                        pulp::cast(self.avx2._mm256_bsrli_epi128::<4>(pulp::cast(value))),
                    );
                }
                4 => value = self.wrapping_sub_i32x8(Self::ZERO, value),
                5 => {
                    value = self.wrapping_sub_i32x8(
                        pulp::cast(self.avx2._mm256_bsrli_epi128::<12>(pulp::cast(value))),
                        pulp::cast(self.avx2._mm256_bslli_epi128::<4>(pulp::cast(value))),
                    );
                }
                6 => {
                    value = self.wrapping_sub_i32x8(
                        pulp::cast(self.avx2._mm256_bsrli_epi128::<8>(pulp::cast(value))),
                        pulp::cast(self.avx2._mm256_bslli_epi128::<8>(pulp::cast(value))),
                    );
                }
                7 => {
                    value = self.wrapping_sub_i32x8(
                        pulp::cast(self.avx2._mm256_bsrli_epi128::<4>(pulp::cast(value))),
                        pulp::cast(self.avx2._mm256_bslli_epi128::<12>(pulp::cast(value))),
                    );
                }
                _ => unreachable!(),
            }

            if const { AMOUNT_MOD_24 == 0 } {
                return pulp::cast(value);
            }

            let mask: i32x8 =
                pulp::cast(const { [((1u32 << TWENTY_FOUR_MINUS_AMOUNT_MOD_24) - 1) as i32; 8] });

            let l = self.shl_const_i32x8::<AMOUNT_MOD_24>(self.and_i32x8(value, mask));
            let r = self.shr_const_i32x8::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value);
            let r = self.wrapping_sub_i32x8(
                pulp::cast(self.avx2._mm256_bslli_epi128::<4>(pulp::cast(r))),
                pulp::cast(self.avx2._mm256_bsrli_epi128::<12>(pulp::cast(r))),
            );

            let mut value = pulp::cast(self.wrapping_add_i32x8(l, r));

            if const { AMOUNT_MOD_24 } > 16 {
                value = self.propagate_carries_i32x4x2(value)
            }

            value
        }

        #[inline(always)]
        pub fn shl_dyn_i32x4x2<const MAX_ROT: u32>(self, value: i32x4x2, amount: u32x8) -> i32x4x2 {
            const { assert!(MAX_ROT <= 2) };

            #[cfg(debug_assertions)]
            {
                let amount_: [u32; 8] = pulp::cast(amount);
                if max_u32s(&amount_) > 24 * (1 + MAX_ROT) {
                    panic!("shift amount exceeds 24: {:?}", amount);
                }
            }

            let mut value = pulp::cast(value);
            let mut amount = amount;

            let one: i32x8 = pulp::cast(const { [1_i32; 8] });
            let twenty_four: i32x8 = pulp::cast(const { [24_i32; 8] });

            for _ in 0..MAX_ROT {
                let old = value;
                let rot = self.wrapping_sub_i32x8(
                    pulp::cast(self.avx2._mm256_bslli_epi128::<4>(pulp::cast(value))),
                    pulp::cast(self.avx2._mm256_bsrli_epi128::<12>(pulp::cast(value))),
                );

                // with avx2,
                // cmp_lt_i32x8 is more efficient than cmp_lt_u32x8
                let keep_old = self.cmp_lt_i32x8(pulp::cast(amount), twenty_four);

                amount = self.wrapping_sub_u32x8(
                    amount,
                    self.andnot_u32x8(pulp::cast(keep_old), pulp::cast(twenty_four)),
                );

                value = self.select_i32x8(keep_old, old, rot);
            }

            let twenty_four_minus_amount = self.wrapping_sub_i32x8(twenty_four, pulp::cast(amount));

            let mask: i32x8 = self.wrapping_sub_i32x8(
                self.shl_dyn_i32x8(one, pulp::cast(twenty_four_minus_amount)),
                one,
            );

            let l = self.shl_dyn_i32x8(self.and_i32x8(value, mask), amount);
            let r = self.shr_dyn_i32x8(value, twenty_four_minus_amount);
            let r = self.wrapping_sub_i32x8(
                pulp::cast(self.avx2._mm256_bslli_epi128::<4>(pulp::cast(r))),
                pulp::cast(self.avx2._mm256_bsrli_epi128::<12>(pulp::cast(r))),
            );

            let value = pulp::cast(self.wrapping_add_i32x8(l, r));

            self.propagate_carries_i32x4x2(value)
        }

        #[inline(always)]
        pub fn r96x8_from_zpx8(self, lo: u32x8, hi: u32x8) -> R96x8 {
            let mask_lo = pulp::cast(const { [0xFFFFFF_u32; 8] });
            let mask_hi = pulp::cast(const { [0xFFFF_u32; 8] });

            R96x8 {
                x: pulp::cast(self.and_u32x8(lo, mask_lo)),
                y: pulp::cast(
                    self.or_u32x8(self.shr_const_u32x8::<24>(lo), self.and_u32x8(hi, mask_hi)),
                ),
                z: pulp::cast(self.shr_const_u32x8::<16>(hi)),
                w: Self::ZERO,
            }
        }

        #[inline(always)]
        pub fn zpx8_from_r96x8(self, value: R96x8) -> (u32x8, u32x8) {
            let R96x8 { x, y, z, w } = value;

            let x: [i32x4; 2] = pulp::cast(x);
            let y: [i32x4; 2] = pulp::cast(y);
            let z: [i32x4; 2] = pulp::cast(z);
            let w: [i32x4; 2] = pulp::cast(w);

            let x = [
                self.convert_i32x4_to_i64x4(x[0]),
                self.convert_i32x4_to_i64x4(x[1]),
            ];
            let y = [
                self.shl_const_i64x4::<24>(self.convert_i32x4_to_i64x4(y[0])),
                self.shl_const_i64x4::<24>(self.convert_i32x4_to_i64x4(y[1])),
            ];

            let mask = pulp::cast(const { [0xFFFF_u64; 4] });
            let z0 = [
                self.shl_const_u64x4::<48>(self.and_u64x4(
                    self.convert_u32x4_to_u64x4(self.convert_i32x4_to_u32x4(z[0])),
                    mask,
                )),
                self.shl_const_u64x4::<48>(self.and_u64x4(
                    self.convert_u32x4_to_u64x4(self.convert_i32x4_to_u32x4(z[1])),
                    mask,
                )),
            ];

            let z1 = [
                self.convert_i32x4_to_i64x4(self.shr_const_i32x4::<16>(z[0])),
                self.convert_i32x4_to_i64x4(self.shr_const_i32x4::<16>(z[1])),
            ];

            let w0 = [
                self.shl_const_u64x4::<8>(
                    self.convert_u32x4_to_u64x4(self.convert_i32x4_to_u32x4(w[0])),
                ),
                self.shl_const_u64x4::<8>(
                    self.convert_u32x4_to_u64x4(self.convert_i32x4_to_u32x4(w[1])),
                ),
            ];

            let w1 = [
                self.convert_i32x4_to_i64x4(self.shr_const_i32x4::<24>(w[0])),
                self.convert_i32x4_to_i64x4(self.shr_const_i32x4::<24>(w[1])),
            ];

            let a = [
                self.wrapping_sub_i64x4(self.wrapping_add_i64x4(x[0], y[0]), w1[0]),
                self.wrapping_sub_i64x4(self.wrapping_add_i64x4(x[1], y[1]), w1[1]),
            ];

            let p: u64x4 = pulp::cast(const { [P; 4] });

            #[inline(always)]
            fn add_p_if_negative(simd: V3, a: i64x4) -> u64x4 {
                let p: i64x4 = pulp::cast(const { [P; 4] });

                pulp::cast(simd.avx._mm256_blendv_pd(
                    pulp::cast(a),
                    pulp::cast(simd.wrapping_add_i64x4(a, p)),
                    pulp::cast(a),
                ))
            }

            #[inline(always)]
            fn add(simd: V3, a: u64x4, b: u64x4) -> u64x4 {
                let p: u64x4 = pulp::cast(const { [P; 4] });
                let ab = simd.wrapping_add_u64x4(a, b);
                let overflow: u64x4 = pulp::cast(simd.cmp_lt_u64x4(ab, a));
                simd.wrapping_sub_u64x4(ab, simd.and_u64x4(overflow, p))
            }

            // we use the top bit (sign bit in 2's complement) of each 64-bit chunk as the mask bit
            let a = [add_p_if_negative(self, a[0]), add_p_if_negative(self, a[1])];

            let b = z0;

            let c = [
                add_p_if_negative(
                    self,
                    self.wrapping_sub_i64x4(self.shl_const_i64x4::<32>(z1[0]), z1[0]),
                ),
                add_p_if_negative(
                    self,
                    self.wrapping_sub_i64x4(self.shl_const_i64x4::<32>(z1[1]), z1[1]),
                ),
            ];

            let d = [
                self.wrapping_sub_u64x4(self.shl_const_u64x4::<32>(w0[0]), w0[0]),
                self.wrapping_sub_u64x4(self.shl_const_u64x4::<32>(w0[1]), w0[1]),
            ];

            let ab = [add(self, a[0], b[0]), add(self, a[1], b[1])];
            let cd = [add(self, c[0], d[0]), add(self, c[1], d[1])];

            let abcd = [add(self, ab[0], cd[0]), add(self, ab[1], cd[1])];
            let smaller_than_p: [u64x4; 2] = [
                pulp::cast(self.cmp_lt_u64x4(abcd[0], p)),
                pulp::cast(self.cmp_lt_u64x4(abcd[1], p)),
            ];

            let abcd = [
                self.wrapping_sub_u64x4(abcd[0], self.andnot_u64x4(smaller_than_p[0], p)),
                self.wrapping_sub_u64x4(abcd[1], self.andnot_u64x4(smaller_than_p[1], p)),
            ];

            self.u32x8x2_from_u64x4x2(abcd[0], abcd[1])
        }

        #[inline(always)]
        fn u32x8x2_from_u64x4x2(self, values_0: u64x4, values_1: u64x4) -> (u32x8, u32x8) {
            let a = pulp::cast(values_0);
            let b = pulp::cast(values_1);
            // a = [0, 1, 2, 3, 4, 5, 6, 7]
            // b = [8, 9, a, b, c, d, e, f]
            //
            // we want
            // - [0, 2, 4, 6, 8, a, c, e]
            // - [1, 3, 5, 7, 9, b, d, f]

            let idx = pulp::cast(const { u32x8(0, 2, 4, 6, 1, 3, 5, 7) });

            // [0, 2, 4, 6, 1, 3, 5, 7]
            let a = self.avx2._mm256_permutevar8x32_epi32(a, idx);
            // [8, a, c, e, 9, b, d, f]
            let b = self.avx2._mm256_permutevar8x32_epi32(b, idx);

            let (a, b) = (
                self.avx2._mm256_permute2x128_si256::<0b_0010_0000>(a, b),
                self.avx2._mm256_permute2x128_si256::<0b_0011_0001>(a, b),
            );

            (pulp::cast(a), pulp::cast(b))
        }

        #[inline(always)]
        pub fn r96x8_from_u64x4x2(self, values_0: u64x4, values_1: u64x4) -> R96x8 {
            let (a, b) = self.u32x8x2_from_u64x4x2(values_0, values_1);
            self.r96x8_from_zpx8(a, b)
        }

        #[inline(always)]
        #[track_caller]
        pub fn add_r96x8(self, lhs: R96x8, rhs: R96x8) -> R96x8 {
            R96x8 {
                x: self.wrapping_add_i32x8(lhs.x, rhs.x),
                y: self.wrapping_add_i32x8(lhs.y, rhs.y),
                z: self.wrapping_add_i32x8(lhs.z, rhs.z),
                w: self.wrapping_add_i32x8(lhs.w, rhs.w),
            }
        }

        #[inline(always)]
        #[track_caller]
        pub fn sub_r96x8(self, lhs: R96x8, rhs: R96x8) -> R96x8 {
            R96x8 {
                x: self.wrapping_sub_i32x8(lhs.x, rhs.x),
                y: self.wrapping_sub_i32x8(lhs.y, rhs.y),
                z: self.wrapping_sub_i32x8(lhs.z, rhs.z),
                w: self.wrapping_sub_i32x8(lhs.w, rhs.w),
            }
        }

        #[inline(always)]
        pub fn neg_r96x8(self, value: R96x8) -> R96x8 {
            R96x8 {
                x: self.wrapping_sub_i32x8(Self::ZERO, value.x),
                y: self.wrapping_sub_i32x8(Self::ZERO, value.y),
                z: self.wrapping_sub_i32x8(Self::ZERO, value.z),
                w: self.wrapping_sub_i32x8(Self::ZERO, value.w),
            }
        }

        #[inline(always)]
        pub fn propagate_carries_r96x8(self, value: R96x8) -> R96x8 {
            let lo = R96x8 {
                x: self.and_i32x8(value.x, Self::MASK),
                y: self.and_i32x8(value.y, Self::MASK),
                z: self.and_i32x8(value.z, Self::MASK),
                w: self.and_i32x8(value.w, Self::MASK),
            };
            let hi = R96x8 {
                x: self.shr_const_i32x8::<24>(self.wrapping_sub_i32x8(Self::ZERO, value.w)),
                y: self.shr_const_i32x8::<24>(value.x),
                z: self.shr_const_i32x8::<24>(value.y),
                w: self.shr_const_i32x8::<24>(value.z),
            };

            self.add_r96x8(lo, hi)
        }

        #[inline(always)]
        pub fn shl_const_r96x8<
            const AMOUNT: i32,
            const AMOUNT_MOD_24: i32,
            const TWENTY_FOUR_MINUS_AMOUNT_MOD_24: i32,
        >(
            self,
            value: R96x8,
        ) -> R96x8 {
            const {
                assert!(AMOUNT >= 0);
                assert!(AMOUNT_MOD_24 == AMOUNT % 24);
                assert!(TWENTY_FOUR_MINUS_AMOUNT_MOD_24 == 24 - AMOUNT_MOD_24);
            };

            let mut value = value;

            {
                let R96x8 { x, y, z, w } = value;
                match const { (AMOUNT % 192) / 24 } {
                    0 => {}
                    1 => {
                        value.x = self.wrapping_sub_i32x8(Self::ZERO, w);
                        value.y = x;
                        value.z = y;
                        value.w = z;
                    }
                    2 => {
                        value.x = self.wrapping_sub_i32x8(Self::ZERO, z);
                        value.y = self.wrapping_sub_i32x8(Self::ZERO, w);
                        value.z = x;
                        value.w = y;
                    }
                    3 => {
                        value.x = self.wrapping_sub_i32x8(Self::ZERO, y);
                        value.y = self.wrapping_sub_i32x8(Self::ZERO, z);
                        value.z = self.wrapping_sub_i32x8(Self::ZERO, w);
                        value.w = x;
                    }
                    4 => {
                        value.x = self.wrapping_sub_i32x8(Self::ZERO, x);
                        value.y = self.wrapping_sub_i32x8(Self::ZERO, y);
                        value.z = self.wrapping_sub_i32x8(Self::ZERO, z);
                        value.w = self.wrapping_sub_i32x8(Self::ZERO, w);
                    }
                    5 => {
                        value.x = w;
                        value.y = self.wrapping_sub_i32x8(Self::ZERO, x);
                        value.z = self.wrapping_sub_i32x8(Self::ZERO, y);
                        value.w = self.wrapping_sub_i32x8(Self::ZERO, z);
                    }
                    6 => {
                        value.x = z;
                        value.y = w;
                        value.z = self.wrapping_sub_i32x8(Self::ZERO, x);
                        value.w = self.wrapping_sub_i32x8(Self::ZERO, y);
                    }
                    7 => {
                        value.x = y;
                        value.y = z;
                        value.z = w;
                        value.w = self.wrapping_sub_i32x8(Self::ZERO, x);
                    }
                    _ => unreachable!(),
                }
            }

            if const { AMOUNT_MOD_24 == 0 } {
                return value;
            }

            let mask: i32x8 =
                pulp::cast(const { [((1u32 << TWENTY_FOUR_MINUS_AMOUNT_MOD_24) - 1) as i32; 8] });

            value = R96x8 {
                x: self.wrapping_sub_i32x8(
                    self.shl_const_i32x8::<AMOUNT_MOD_24>(self.and_i32x8(value.x, mask)),
                    self.shr_const_i32x8::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.w),
                ),
                y: self.wrapping_add_i32x8(
                    self.shl_const_i32x8::<AMOUNT_MOD_24>(self.and_i32x8(value.y, mask)),
                    self.shr_const_i32x8::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.x),
                ),
                z: self.wrapping_add_i32x8(
                    self.shl_const_i32x8::<AMOUNT_MOD_24>(self.and_i32x8(value.z, mask)),
                    self.shr_const_i32x8::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.y),
                ),
                w: self.wrapping_add_i32x8(
                    self.shl_const_i32x8::<AMOUNT_MOD_24>(self.and_i32x8(value.w, mask)),
                    self.shr_const_i32x8::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.z),
                ),
            };

            if const { AMOUNT_MOD_24 } > 16 {
                value = self.propagate_carries_r96x8(value)
            }
            value
        }

        #[inline(always)]
        pub fn shl_dyn_r96x8<const MAX_ROT: u32>(
            self,
            value: R96x8,
            amount: u32x8,
            amount_div_3: u32x8,
        ) -> R96x8 {
            #[cfg(debug_assertions)]
            {
                let amount_: [u32; 8] = pulp::cast(amount);
                if max_u32s(&amount_) > 24 * (1 + MAX_ROT) {
                    panic!("shift amount exceeds 24: {:?}", amount);
                }
                assert_eq!(
                    amount,
                    self.wrapping_add_u32x8(amount_div_3, self.shl_const_u32x8::<1>(amount_div_3)),
                );
            }

            let mut value = value;
            let mut amount = amount;

            if const { MAX_ROT > 4 } {
                let rot = R96x8 {
                    x: self.wrapping_sub_i32x8(Self::ZERO, value.x),
                    y: self.wrapping_sub_i32x8(Self::ZERO, value.y),
                    z: self.wrapping_sub_i32x8(Self::ZERO, value.z),
                    w: self.wrapping_sub_i32x8(Self::ZERO, value.w),
                };

                // we select `rot` or `value` depending on the value of bit 5 in `amount_div_3`
                // shift it up to the msb
                let mask = self.shl_const_u32x8::<26>(amount_div_3);

                let amount_div_3 = self.and_u32x8(amount_div_3, self.splat_u32x8(0b11111));
                amount =
                    self.wrapping_add_u32x8(amount_div_3, self.shl_const_u32x8::<1>(amount_div_3));

                value.x = pulp::cast(self.avx._mm256_blendv_ps(
                    pulp::cast(value.x),
                    pulp::cast(rot.x),
                    pulp::cast(mask),
                ));
                value.y = pulp::cast(self.avx._mm256_blendv_ps(
                    pulp::cast(value.y),
                    pulp::cast(rot.y),
                    pulp::cast(mask),
                ));
                value.z = pulp::cast(self.avx._mm256_blendv_ps(
                    pulp::cast(value.z),
                    pulp::cast(rot.z),
                    pulp::cast(mask),
                ));
                value.w = pulp::cast(self.avx._mm256_blendv_ps(
                    pulp::cast(value.w),
                    pulp::cast(rot.w),
                    pulp::cast(mask),
                ));
            }

            let one: i32x8 = pulp::cast(const { [1i32; 8] });
            let twenty_four: i32x8 = pulp::cast(const { [24_i32; 8] });

            for _ in 0..const { u32_max(4, MAX_ROT) } {
                let old = value;
                let rot = R96x8 {
                    x: self.wrapping_sub_i32x8(Self::ZERO, value.w),
                    y: value.x,
                    z: value.y,
                    w: value.z,
                };

                // with avx2,
                // cmp_lt_i32x8 is more efficient than cmp_lt_u32x8
                let keep_old = self.cmp_lt_i32x8(pulp::cast(amount), twenty_four);

                amount = self.wrapping_sub_u32x8(
                    amount,
                    self.andnot_u32x8(pulp::cast(keep_old), pulp::cast(twenty_four)),
                );

                value.x = self.select_i32x8(keep_old, old.x, rot.x);
                value.y = self.select_i32x8(keep_old, old.y, rot.y);
                value.z = self.select_i32x8(keep_old, old.z, rot.z);
                value.w = self.select_i32x8(keep_old, old.w, rot.w);
            }

            let twenty_four_minus_amount = self.wrapping_sub_i32x8(twenty_four, pulp::cast(amount));

            let mask: i32x8 = self.wrapping_sub_i32x8(
                self.shl_dyn_i32x8(one, pulp::cast(twenty_four_minus_amount)),
                one,
            );

            let value = R96x8 {
                x: self.wrapping_sub_i32x8(
                    self.shl_dyn_i32x8(self.and_i32x8(value.x, mask), amount),
                    self.shr_dyn_i32x8(value.w, twenty_four_minus_amount),
                ),
                y: self.wrapping_add_i32x8(
                    self.shl_dyn_i32x8(self.and_i32x8(value.y, mask), amount),
                    self.shr_dyn_i32x8(value.x, twenty_four_minus_amount),
                ),
                z: self.wrapping_add_i32x8(
                    self.shl_dyn_i32x8(self.and_i32x8(value.z, mask), amount),
                    self.shr_dyn_i32x8(value.y, twenty_four_minus_amount),
                ),
                w: self.wrapping_add_i32x8(
                    self.shl_dyn_i32x8(self.and_i32x8(value.w, mask), amount),
                    self.shr_dyn_i32x8(value.z, twenty_four_minus_amount),
                ),
            };

            self.propagate_carries_r96x8(value)
        }

        #[inline(always)]
        pub fn mul_zpx8(
            self,
            lhs_lo: u32x8,
            lhs_hi: u32x8,
            rhs_lo: u32x8,
            rhs_hi: u32x8,
        ) -> (u32x8, u32x8) {
            // pl = 1
            let pl: u32x8 = pulp::cast(const { [P as u32; 8] });
            // ph = u32::MAX
            let ph: u32x8 = pulp::cast(const { [(P >> 32) as u32; 8] });
            let zero: u32x8 = pulp::cast(Self::ZERO);
            let max = ph;

            let lolo = self.widening_mul_u32x8(lhs_lo, rhs_lo);
            let lohi = self.widening_mul_u32x8(lhs_lo, rhs_hi);
            let hilo = self.widening_mul_u32x8(lhs_hi, rhs_lo);
            let hihi = self.widening_mul_u32x8(lhs_hi, rhs_hi);

            let l = lolo.0;

            let ml = self.wrapping_add_u32x8(lohi.0, hilo.0);
            let o0 = pulp::cast(self.cmp_lt_u32x8(ml, lohi.0));

            let ml = self.wrapping_add_u32x8(ml, lolo.1);
            let o1 = pulp::cast(self.cmp_lt_u32x8(ml, lolo.1));

            let mh = self.wrapping_sub_u32x8(lohi.1, o0);
            let mh = self.wrapping_sub_u32x8(mh, o1);
            let o0 =
                pulp::cast(self.and_m32x8(self.cmp_eq_u32x8(o0, max), self.cmp_eq_u32x8(mh, zero)));

            let mh = self.wrapping_add_u32x8(mh, hilo.1);
            let o1 = pulp::cast(self.cmp_lt_u32x8(mh, hilo.1));

            let mh = self.wrapping_add_u32x8(mh, hihi.0);
            let o2 = pulp::cast(self.cmp_lt_u32x8(mh, hihi.0));

            let h = hihi.1;
            let h = self.wrapping_sub_u32x8(h, o0);
            let h = self.wrapping_sub_u32x8(h, o1);
            let h = self.wrapping_sub_u32x8(h, o2);

            let (l, borrow) = (
                self.wrapping_sub_u32x8(l, h),
                pulp::cast(self.cmp_lt_u32x8(l, h)),
            );

            let (ml, borrow) = (
                self.wrapping_add_u32x8(ml, borrow),
                self.and_u32x8(borrow, pulp::cast(self.cmp_eq_u32x8(ml, zero))),
            );

            let overflow = self.and_u32x8(borrow, pulp::cast(self.cmp_eq_u32x8(l, max)));
            let l = self.wrapping_add_u32x8(l, self.and_u32x8(borrow, pl));
            let ml = self.wrapping_add_u32x8(ml, self.and_u32x8(borrow, ph));
            let ml = self.wrapping_sub_u32x8(ml, overflow);

            let borrow = pulp::cast(self.cmp_lt_u32x8(l, mh));
            let l = self.wrapping_sub_u32x8(l, mh);

            let ml = self.wrapping_add_u32x8(ml, borrow);
            let ml = self.wrapping_add_u32x8(ml, mh);

            let lo = l;
            let hi = ml;

            let less_than_hi = self.cmp_lt_u32x8(hi, mh);
            let eq_hi = self.cmp_eq_u32x8(hi, mh);
            let less_than_lo = self.cmp_lt_u32x8(lo, self.wrapping_sub_u32x8(zero, mh));

            let less_than =
                pulp::cast(self.or_m32x8(less_than_hi, self.and_m32x8(eq_hi, less_than_lo)));

            let (lo, borrow) = (
                self.wrapping_sub_u32x8(lo, self.and_u32x8(less_than, pl)),
                self.and_u32x8(less_than, pulp::cast(self.cmp_eq_u32x8(lo, zero))),
            );
            let hi = self.wrapping_add_u32x8(hi, borrow);
            let hi = self.wrapping_sub_u32x8(hi, self.and_u32x8(less_than, ph));

            let less_than_hi = self.cmp_lt_u32x8(hi, ph);
            let eq_hi = self.cmp_eq_u32x8(hi, ph);
            let less_than_lo = self.cmp_lt_u32x8(lo, pl);

            let less_than =
                pulp::cast(self.or_m32x8(less_than_hi, self.and_m32x8(eq_hi, less_than_lo)));

            let greater_eq = self.not_u32x8(less_than);

            let (lo, borrow) = (
                self.wrapping_sub_u32x8(lo, self.and_u32x8(greater_eq, pl)),
                self.and_u32x8(greater_eq, pulp::cast(self.cmp_eq_u32x8(lo, zero))),
            );
            let hi = self.wrapping_add_u32x8(hi, borrow);
            let hi = self.wrapping_sub_u32x8(hi, self.and_u32x8(greater_eq, ph));

            (lo, hi)
        }

        #[inline(always)]
        pub(crate) fn select_r96x8(self, mask: m32x8, if_true: R96x8, if_false: R96x8) -> R96x8 {
            R96x8 {
                x: self.select_i32x8(mask, if_true.x, if_false.x),
                y: self.select_i32x8(mask, if_true.y, if_false.y),
                z: self.select_i32x8(mask, if_true.z, if_false.z),
                w: self.select_i32x8(mask, if_true.w, if_false.w),
            }
        }

        #[inline(always)]
        pub(crate) fn interleave4_r96x8(self, values: [R96x8; 2]) -> [R96x8; 2] {
            let x = self.interleave4_u32x8([pulp::cast(values[0].x), pulp::cast(values[1].x)]);
            let y = self.interleave4_u32x8([pulp::cast(values[0].y), pulp::cast(values[1].y)]);
            let z = self.interleave4_u32x8([pulp::cast(values[0].z), pulp::cast(values[1].z)]);
            let w = self.interleave4_u32x8([pulp::cast(values[0].w), pulp::cast(values[1].w)]);

            [
                R96x8 {
                    x: pulp::cast(x[0]),
                    y: pulp::cast(y[0]),
                    z: pulp::cast(z[0]),
                    w: pulp::cast(w[0]),
                },
                R96x8 {
                    x: pulp::cast(x[1]),
                    y: pulp::cast(y[1]),
                    z: pulp::cast(z[1]),
                    w: pulp::cast(w[1]),
                },
            ]
        }

        #[inline(always)]
        pub(crate) fn interleave2_r96x8(self, values: [R96x8; 2]) -> [R96x8; 2] {
            let x = self.interleave2_u32x8([pulp::cast(values[0].x), pulp::cast(values[1].x)]);
            let y = self.interleave2_u32x8([pulp::cast(values[0].y), pulp::cast(values[1].y)]);
            let z = self.interleave2_u32x8([pulp::cast(values[0].z), pulp::cast(values[1].z)]);
            let w = self.interleave2_u32x8([pulp::cast(values[0].w), pulp::cast(values[1].w)]);

            [
                R96x8 {
                    x: pulp::cast(x[0]),
                    y: pulp::cast(y[0]),
                    z: pulp::cast(z[0]),
                    w: pulp::cast(w[0]),
                },
                R96x8 {
                    x: pulp::cast(x[1]),
                    y: pulp::cast(y[1]),
                    z: pulp::cast(z[1]),
                    w: pulp::cast(w[1]),
                },
            ]
        }

        #[inline(always)]
        pub(crate) fn interleave1_r96x8(self, values: [R96x8; 2]) -> [R96x8; 2] {
            let x = self.interleave1_u32x8([pulp::cast(values[0].x), pulp::cast(values[1].x)]);
            let y = self.interleave1_u32x8([pulp::cast(values[0].y), pulp::cast(values[1].y)]);
            let z = self.interleave1_u32x8([pulp::cast(values[0].z), pulp::cast(values[1].z)]);
            let w = self.interleave1_u32x8([pulp::cast(values[0].w), pulp::cast(values[1].w)]);

            [
                R96x8 {
                    x: pulp::cast(x[0]),
                    y: pulp::cast(y[0]),
                    z: pulp::cast(z[0]),
                    w: pulp::cast(w[0]),
                },
                R96x8 {
                    x: pulp::cast(x[1]),
                    y: pulp::cast(y[1]),
                    z: pulp::cast(z[1]),
                    w: pulp::cast(w[1]),
                },
            ]
        }

        #[inline(always)]
        pub fn ntt_32(self, p: &mut [i32x4; 32]) {
            let p: &mut [i32x4x2; 16] = bytemuck::cast_mut(p);

            let omega1 = mul_by_w8_new!(self, 1);
            let omega2 = mul_by_w8_new!(self, 2);
            let omega3 = mul_by_w8_new!(self, 3);

            // ntt_4x8
            {
                let (p0, p1, p2, p3) = split_4_mut(p);

                for (p0, p1, p2, p3) in izip!(p0, p1, p2, p3) {
                    [*p0, *p2] = [self.add_i32x4x2(*p0, *p2), self.sub_i32x4x2(*p0, *p2)];
                    [*p1, *p3] = [
                        self.add_i32x4x2(*p1, *p3),
                        omega2.apply(self.sub_i32x4x2(*p1, *p3)),
                    ];
                    [*p0, *p1] = [self.add_i32x4x2(*p0, *p1), self.sub_i32x4x2(*p0, *p1)];
                    [*p2, *p3] = [self.add_i32x4x2(*p2, *p3), self.sub_i32x4x2(*p2, *p3)];
                }
            }

            {
                macro_rules! mul_by_w8_inner_loop {
                    ($i: expr, $j: expr) => {{
                        const SHIFT: (i32, u32x8) = {
                            let mut pow = [0u32; 8];
                            let offset = 2 * $j;

                            pow[0] = offset * S * OMEGA_64_SHL_AMOUNT;
                            pow[4] = (offset + 1) * S * OMEGA_64_SHL_AMOUNT;

                            let floor = pow[0] / 24 * 24;
                            pow[0] -= floor;
                            pow[4] -= floor;

                            pow[1] = pow[0];
                            pow[2] = pow[0];
                            pow[3] = pow[0];
                            pow[5] = pow[4];
                            pow[6] = pow[4];
                            pow[7] = pow[4];

                            (floor as i32, u32x8_from_array(pow))
                        };

                        p[4 * $i + $j] = shl_const_new!(self, SHIFT.0).apply(p[4 * $i + $j]);
                        p[4 * $i + $j] =
                            self.shl_dyn_i32x4x2::<{ SHIFT.1 .4 / 24 }>(p[4 * $i + $j], SHIFT.1);
                    }};
                }

                macro_rules! mul_by_w8_outer_loop {
                    ($i: expr) => {{
                        const NBITS: u32 = 2;
                        const I: u32 = $i;
                        const I_REV: u32 = I.reverse_bits() >> (32 - NBITS);
                        const S: u32 = 2 * I_REV;

                        mul_by_w8_inner_loop!($i, 0);
                        mul_by_w8_inner_loop!($i, 1);
                        mul_by_w8_inner_loop!($i, 2);
                        mul_by_w8_inner_loop!($i, 3);
                    }};
                }

                // i = 0 => no-op
                mul_by_w8_outer_loop!(1);
                mul_by_w8_outer_loop!(2);
                mul_by_w8_outer_loop!(3);
            }

            // ntt_8
            let p: &mut [[[i32x4x2; 4]; 2]; 2] = bytemuck::cast_mut(p);

            for p in p.iter_mut() {
                let [mut x0, mut y0]: [i32x4x2; 2] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(p[0][0]), pulp::cast(p[1][0])]));
                let [mut x1, mut y1]: [i32x4x2; 2] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(p[0][1]), pulp::cast(p[1][1])]));
                let [mut x2, mut y2]: [i32x4x2; 2] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(p[0][2]), pulp::cast(p[1][2])]));
                let [mut x3, mut y3]: [i32x4x2; 2] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(p[0][3]), pulp::cast(p[1][3])]));

                [x0, y0] = [self.add_i32x4x2(x0, y0), self.sub_i32x4x2(x0, y0)];
                [x1, y1] = [self.add_i32x4x2(x1, y1), self.sub_i32x4x2(x1, y1)];
                [x2, y2] = [
                    self.add_i32x4x2(x2, y2),
                    omega2.apply(self.sub_i32x4x2(x2, y2)),
                ];
                [x3, y3] = [
                    self.add_i32x4x2(x3, y3),
                    omega2.apply(self.sub_i32x4x2(x3, y3)),
                ];

                [p[0][0], p[1][0]] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(x0), pulp::cast(y0)]));
                [p[0][1], p[1][1]] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(x1), pulp::cast(y1)]));
                [p[0][2], p[1][2]] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(x2), pulp::cast(y2)]));
                [p[0][3], p[1][3]] =
                    pulp::cast(self.interleave4_u32x8([pulp::cast(x3), pulp::cast(y3)]));
            }

            let p: &mut [[i32x4x2; 4]; 4] = bytemuck::cast_mut(p);

            const T: m32 = m32::new(true);
            const F: m32 = m32::new(false);

            for [p01, p23, p45, p67] in p {
                [*p01, *p23] = [self.add_i32x4x2(*p01, *p23), self.sub_i32x4x2(*p01, *p23)];
                *p23 = pulp::cast(self.select_i32x8(
                    m32x8(F, F, F, F, T, T, T, T),
                    pulp::cast(omega2.apply(*p23)),
                    pulp::cast(*p23),
                ));

                [*p45, *p67] = [self.add_i32x4x2(*p45, *p67), self.sub_i32x4x2(*p45, *p67)];

                *p45 = pulp::cast(self.select_i32x8(
                    m32x8(F, F, F, F, T, T, T, T),
                    pulp::cast(omega1.apply(*p45)),
                    pulp::cast(*p45),
                ));
                *p67 = pulp::cast(self.select_i32x8(
                    m32x8(F, F, F, F, T, T, T, T),
                    pulp::cast(omega3.apply(*p67)),
                    pulp::cast(*p67),
                ));
            }
        }

        #[inline(always)]
        pub fn ntt_32xn(self, p: &mut [i32x4], n: usize) {
            assert!(n.is_power_of_two());
            assert!(n >= 2);
            assert_eq!(p.len(), 32 * n);

            let p: &mut [i32x4x2] = bytemuck::cast_slice_mut(p);

            let omega1 = mul_by_w8_new!(self, 1);
            let omega2 = mul_by_w8_new!(self, 2);
            let omega3 = mul_by_w8_new!(self, 3);

            {
                let (p0, p1, p2, p3) = split_4_mut(p);

                for (p0, p1, p2, p3) in izip!(p0, p1, p2, p3) {
                    [*p0, *p2] = [self.sub_i32x4x2(*p0, *p2), self.sub_i32x4x2(*p0, *p2)];
                    [*p1, *p3] = [
                        self.sub_i32x4x2(*p1, *p3),
                        omega2.apply(self.sub_i32x4x2(*p1, *p3)),
                    ];

                    [*p0, *p1] = [self.sub_i32x4x2(*p0, *p1), self.sub_i32x4x2(*p0, *p1)];
                    [*p2, *p3] = [self.sub_i32x4x2(*p2, *p3), self.sub_i32x4x2(*p2, *p3)];
                }
            }

            {
                macro_rules! mul_by_w8_inner_loop {
                    ($iter: ident, $j: expr) => {
                        if let Some(p) = ($iter).next() {
                            for p in p {
                                *p = mul_by_w64_new!(self, $j * S).apply(*p);
                            }
                        }
                    };
                }

                macro_rules! mul_by_w8_outer_loop {
                    ($iter: ident, $i: expr) => {
                        if let Some(p) = ($iter).next() {
                            const NBITS: u32 = 2;
                            const I: u32 = $i;
                            const I_REV: u32 = I.reverse_bits() >> (32 - NBITS);
                            const S: u32 = 2 * I_REV;

                            let mut iter = p.chunks_exact_mut(n / 2);

                            // j = 0 => no-op
                            _ = iter.next();
                            mul_by_w8_inner_loop!(iter, 1);
                            mul_by_w8_inner_loop!(iter, 2);
                            mul_by_w8_inner_loop!(iter, 3);
                            mul_by_w8_inner_loop!(iter, 4);
                            mul_by_w8_inner_loop!(iter, 5);
                            mul_by_w8_inner_loop!(iter, 6);
                            mul_by_w8_inner_loop!(iter, 7);
                        }
                    };
                }

                let mut iter = p.chunks_exact_mut(4 * n);

                // i = 0 => no-op
                _ = iter.next();
                mul_by_w8_outer_loop!(iter, 1);
                mul_by_w8_outer_loop!(iter, 2);
                mul_by_w8_outer_loop!(iter, 3);
            }

            {
                let (p0, p1, p2, p3) = split_4_mut(p);

                for p in [p0, p1, p2, p3] {
                    let (p0, p1, p2, p3, p4, p5, p6, p7) = split_8_mut(p);

                    for (p0, p1, p2, p3, p4, p5, p6, p7) in izip!(p0, p1, p2, p3, p4, p5, p6, p7) {
                        [*p0, *p4] = [self.add_i32x4x2(*p0, *p4), self.sub_i32x4x2(*p0, *p4)];
                        [*p1, *p5] = [self.add_i32x4x2(*p1, *p5), self.sub_i32x4x2(*p1, *p5)];
                        [*p2, *p6] = [
                            self.add_i32x4x2(*p2, *p6),
                            omega2.apply(self.sub_i32x4x2(*p2, *p6)),
                        ];
                        [*p3, *p7] = [
                            self.add_i32x4x2(*p3, *p7),
                            omega2.apply(self.sub_i32x4x2(*p3, *p7)),
                        ];

                        [*p0, *p2] = [self.add_i32x4x2(*p0, *p2), self.sub_i32x4x2(*p0, *p2)];
                        [*p1, *p3] = [
                            self.add_i32x4x2(*p1, *p3),
                            omega2.apply(self.sub_i32x4x2(*p1, *p3)),
                        ];
                        [*p4, *p6] = [self.add_i32x4x2(*p4, *p6), self.sub_i32x4x2(*p4, *p6)];
                        [*p5, *p7] = [
                            omega1.apply(self.add_i32x4x2(*p5, *p7)),
                            omega3.apply(self.sub_i32x4x2(*p5, *p7)),
                        ];

                        [*p0, *p1] = [self.add_i32x4x2(*p0, *p1), self.sub_i32x4x2(*p0, *p1)];
                        [*p2, *p3] = [self.add_i32x4x2(*p2, *p3), self.sub_i32x4x2(*p2, *p3)];
                        [*p4, *p5] = [self.add_i32x4x2(*p4, *p5), self.sub_i32x4x2(*p4, *p5)];
                        [*p6, *p7] = [self.add_i32x4x2(*p6, *p7), self.sub_i32x4x2(*p6, *p7)];
                    }
                }
            }
        }
    }

    #[cfg(feature = "nightly")]
    impl V4 {
        const ZERO: i32x16 = i32x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        const MASK: i32x16 = i32x16(
            0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF,
            0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF,
        );

        #[inline(always)]
        pub fn u32x16_from_b16(self, mask: pulp::b16) -> u32x16 {
            let zero: u32x16 = pulp::cast(Self::ZERO);
            let one: u32x16 = pulp::cast(const { [1u32; 16] });

            self.select_u32x16(mask, one, zero)
        }

        #[inline(always)]
        pub fn from_zpx16(self, lo: u32x16, hi: u32x16) -> R96x16 {
            let mask_lo = pulp::cast(const { [0xFFFFFF_u32; 16] });
            let mask_hi = pulp::cast(const { [0xFFFF_u32; 16] });

            R96x16 {
                x: pulp::cast(self.and_u32x16(lo, mask_lo)),
                y: pulp::cast(self.or_u32x16(
                    self.shr_const_u32x16::<24>(lo),
                    self.and_u32x16(hi, mask_hi),
                )),
                z: pulp::cast(self.shr_const_u32x16::<16>(hi)),
                w: Self::ZERO,
            }
        }

        #[inline(always)]
        pub fn zpx16_from_r96x16(self, value: R96x16) -> (u32x16, u32x16) {
            let R96x16 { x, y, z, w } = value;

            let x: [i32x8; 2] = pulp::cast(x);
            let y: [i32x8; 2] = pulp::cast(y);
            let z: [i32x8; 2] = pulp::cast(z);
            let w: [i32x8; 2] = pulp::cast(w);

            let x = [
                self.convert_i32x8_to_i64x8(x[0]),
                self.convert_i32x8_to_i64x8(x[1]),
            ];
            let y = [
                self.shl_const_i64x8::<24>(self.convert_i32x8_to_i64x8(y[0])),
                self.shl_const_i64x8::<24>(self.convert_i32x8_to_i64x8(y[1])),
            ];

            let mask = pulp::cast(const { [0xFFFF_u64; 4] });
            let z0 = [
                self.shl_const_u64x8::<48>(self.and_u64x8(
                    self.convert_u32x8_to_u64x8(self.convert_i32x8_to_u32x8(z[0])),
                    mask,
                )),
                self.shl_const_u64x8::<48>(self.and_u64x8(
                    self.convert_u32x8_to_u64x8(self.convert_i32x8_to_u32x8(z[1])),
                    mask,
                )),
            ];

            let z1 = [
                self.convert_i32x8_to_i64x8(self.shr_const_i32x8::<16>(z[0])),
                self.convert_i32x8_to_i64x8(self.shr_const_i32x8::<16>(z[1])),
            ];

            let w0 = [
                self.shl_const_u64x8::<8>(
                    self.convert_u32x8_to_u64x8(self.convert_i32x8_to_u32x8(w[0])),
                ),
                self.shl_const_u64x8::<8>(
                    self.convert_u32x8_to_u64x8(self.convert_i32x8_to_u32x8(w[1])),
                ),
            ];

            let w1 = [
                self.convert_i32x8_to_i64x8(self.shr_const_i32x8::<24>(w[0])),
                self.convert_i32x8_to_i64x8(self.shr_const_i32x8::<24>(w[1])),
            ];

            let a = [
                self.wrapping_sub_i64x8(self.wrapping_add_i64x8(x[0], y[0]), w1[0]),
                self.wrapping_sub_i64x8(self.wrapping_add_i64x8(x[1], y[1]), w1[1]),
            ];

            let p: u64x8 = pulp::cast(const { [P; 4] });

            #[inline(always)]
            fn add_p_if_negative(simd: V4, a: i64x8) -> u64x8 {
                let p: i64x8 = pulp::cast(const { [P; 4] });

                pulp::cast(simd.avx._mm256_blendv_pd(
                    pulp::cast(a),
                    pulp::cast(simd.wrapping_add_i64x8(a, p)),
                    pulp::cast(a),
                ))
            }

            #[inline(always)]
            fn add(simd: V4, a: u64x8, b: u64x8) -> u64x8 {
                let p: u64x8 = pulp::cast(const { [P; 4] });
                let ab = simd.wrapping_add_u64x8(a, b);
                let overflow: u64x8 = pulp::cast(simd.cmp_lt_u64x8(ab, a));
                simd.wrapping_sub_u64x8(ab, simd.and_u64x8(overflow, p))
            }

            // we use the top bit (sign bit in 2's complement) of each 64-bit chunk as the mask bit
            let a = [add_p_if_negative(self, a[0]), add_p_if_negative(self, a[1])];

            let b = z0;

            let c = [
                add_p_if_negative(
                    self,
                    self.wrapping_sub_i64x8(self.shl_const_i64x8::<32>(z1[0]), z1[0]),
                ),
                add_p_if_negative(
                    self,
                    self.wrapping_sub_i64x8(self.shl_const_i64x8::<32>(z1[1]), z1[1]),
                ),
            ];

            let d = [
                self.wrapping_sub_u64x8(self.shl_const_u64x8::<32>(w0[0]), w0[0]),
                self.wrapping_sub_u64x8(self.shl_const_u64x8::<32>(w0[1]), w0[1]),
            ];

            let ab = [add(self, a[0], b[0]), add(self, a[1], b[1])];
            let cd = [add(self, c[0], d[0]), add(self, c[1], d[1])];

            let abcd = [add(self, ab[0], cd[0]), add(self, ab[1], cd[1])];
            let smaller_than_p: [u64x8; 2] = [
                pulp::cast(self.cmp_lt_u64x8(abcd[0], p)),
                pulp::cast(self.cmp_lt_u64x8(abcd[1], p)),
            ];

            let abcd = [
                self.wrapping_sub_u64x8(abcd[0], self.andnot_u64x8(smaller_than_p[0], p)),
                self.wrapping_sub_u64x8(abcd[1], self.andnot_u64x8(smaller_than_p[1], p)),
            ];

            self.u32x16x2_from_u64x8x2(abcd[0], abcd[1])
        }

        #[inline(always)]
        fn u32x16x2_from_u64x8x2(self, values_0: u64x8, values_1: u64x8) -> (u32x16, u32x16) {
            let a = pulp::cast(values_0);
            let b = pulp::cast(values_1);
            // a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, a, b, c, d, e, f]
            // b = [0', 1', 2', 3', 4', 5', 6', 7', 8', 9', a', b', c', d', e', f']
            //
            // we want
            // - [0, 2, 4, 6, 8, a, c, e, 0', 2', 4', 6', 8', a', c', e']
            // - [1, 3, 5, 7, 9, b, d, f, 1', 3', 5', 7', 9', b', d', f']

            let a_idx = pulp::cast(
                const {
                    u32x16(
                        0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, //
                        0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E,
                    )
                },
            );
            let b_idx = pulp::cast(
                const {
                    u32x16(
                        0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, //
                        0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F, //
                    )
                },
            );

            let (a, b) = (
                self.avx512f._mm512_permutex2var_epi32(a, a_idx, b),
                self.avx512f._mm512_permutex2var_epi32(a, b_idx, b),
            );

            (pulp::cast(a), pulp::cast(b))
        }

        #[inline(always)]
        pub fn from_u64x8x2(self, values_0: u64x8, values_1: u64x8) -> R96x16 {
            let (a, b) = self.u32x16x2_from_u64x8x2(values_0, values_1);
            self.from_zpx16(a, b)
        }

        #[inline(always)]
        #[track_caller]
        pub fn add_r96x16(self, lhs: R96x16, rhs: R96x16) -> R96x16 {
            R96x16 {
                x: self.wrapping_add_i32x16(lhs.x, rhs.x),
                y: self.wrapping_add_i32x16(lhs.y, rhs.y),
                z: self.wrapping_add_i32x16(lhs.z, rhs.z),
                w: self.wrapping_add_i32x16(lhs.w, rhs.w),
            }
        }

        #[inline(always)]
        #[track_caller]
        pub fn sub_r96x16(self, lhs: R96x16, rhs: R96x16) -> R96x16 {
            R96x16 {
                x: self.wrapping_sub_i32x16(lhs.x, rhs.x),
                y: self.wrapping_sub_i32x16(lhs.y, rhs.y),
                z: self.wrapping_sub_i32x16(lhs.z, rhs.z),
                w: self.wrapping_sub_i32x16(lhs.w, rhs.w),
            }
        }

        #[inline(always)]
        pub fn neg_r96x16(self, value: R96x16) -> R96x16 {
            R96x16 {
                x: self.wrapping_sub_i32x16(Self::ZERO, value.x),
                y: self.wrapping_sub_i32x16(Self::ZERO, value.y),
                z: self.wrapping_sub_i32x16(Self::ZERO, value.z),
                w: self.wrapping_sub_i32x16(Self::ZERO, value.w),
            }
        }

        #[inline(always)]
        pub fn propagate_carries_r96x16(self, value: R96x16) -> R96x16 {
            let lo = R96x16 {
                x: self.and_i32x16(value.x, Self::MASK),
                y: self.and_i32x16(value.y, Self::MASK),
                z: self.and_i32x16(value.z, Self::MASK),
                w: self.and_i32x16(value.w, Self::MASK),
            };
            let hi = R96x16 {
                x: self.shr_const_i32x16::<24>(self.wrapping_sub_i32x16(Self::ZERO, value.w)),
                y: self.shr_const_i32x16::<24>(value.x),
                z: self.shr_const_i32x16::<24>(value.y),
                w: self.shr_const_i32x16::<24>(value.z),
            };

            self.add_r96x16(lo, hi)
        }

        #[inline(always)]
        pub fn shl_const_r96x16<
            const AMOUNT: u32,
            const AMOUNT_MOD_24: u32,
            const TWENTY_FOUR_MINUS_AMOUNT_MOD_24: u32,
        >(
            self,
            value: R96x16,
        ) -> R96x16 {
            const {
                assert!(AMOUNT_MOD_24 == AMOUNT % 24);
                assert!(TWENTY_FOUR_MINUS_AMOUNT_MOD_24 == 24 - AMOUNT_MOD_24);
            };

            let mut value = value;

            {
                let R96x16 { x, y, z, w } = value;
                match const { (AMOUNT % 192) / 24 } {
                    0 => {}
                    1 => {
                        value.x = self.wrapping_sub_i32x16(Self::ZERO, w);
                        value.y = x;
                        value.z = y;
                        value.w = z;
                    }
                    2 => {
                        value.x = self.wrapping_sub_i32x16(Self::ZERO, z);
                        value.y = self.wrapping_sub_i32x16(Self::ZERO, w);
                        value.z = x;
                        value.w = y;
                    }
                    3 => {
                        value.x = self.wrapping_sub_i32x16(Self::ZERO, y);
                        value.y = self.wrapping_sub_i32x16(Self::ZERO, z);
                        value.z = self.wrapping_sub_i32x16(Self::ZERO, w);
                        value.w = x;
                    }
                    4 => {
                        value.x = self.wrapping_sub_i32x16(Self::ZERO, x);
                        value.y = self.wrapping_sub_i32x16(Self::ZERO, y);
                        value.z = self.wrapping_sub_i32x16(Self::ZERO, z);
                        value.w = self.wrapping_sub_i32x16(Self::ZERO, w);
                    }
                    5 => {
                        value.x = w;
                        value.y = self.wrapping_sub_i32x16(Self::ZERO, x);
                        value.z = self.wrapping_sub_i32x16(Self::ZERO, y);
                        value.w = self.wrapping_sub_i32x16(Self::ZERO, z);
                    }
                    6 => {
                        value.x = z;
                        value.y = w;
                        value.z = self.wrapping_sub_i32x16(Self::ZERO, x);
                        value.w = self.wrapping_sub_i32x16(Self::ZERO, y);
                    }
                    7 => {
                        value.x = y;
                        value.y = z;
                        value.z = w;
                        value.w = self.wrapping_sub_i32x16(Self::ZERO, x);
                    }
                    _ => unreachable!(),
                }
            }

            if const { AMOUNT_MOD_24 == 0 } {
                return value;
            }

            let mask: i32x16 =
                pulp::cast(const { [((1u32 << TWENTY_FOUR_MINUS_AMOUNT_MOD_24) - 1) as i32; 16] });

            value = R96x16 {
                x: self.wrapping_sub_i32x16(
                    self.shl_const_i32x16::<AMOUNT_MOD_24>(self.and_i32x16(value.x, mask)),
                    self.shr_const_i32x16::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.w),
                ),
                y: self.wrapping_add_i32x16(
                    self.shl_const_i32x16::<AMOUNT_MOD_24>(self.and_i32x16(value.y, mask)),
                    self.shr_const_i32x16::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.x),
                ),
                z: self.wrapping_add_i32x16(
                    self.shl_const_i32x16::<AMOUNT_MOD_24>(self.and_i32x16(value.z, mask)),
                    self.shr_const_i32x16::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.y),
                ),
                w: self.wrapping_add_i32x16(
                    self.shl_const_i32x16::<AMOUNT_MOD_24>(self.and_i32x16(value.w, mask)),
                    self.shr_const_i32x16::<TWENTY_FOUR_MINUS_AMOUNT_MOD_24>(value.z),
                ),
            };

            if const { AMOUNT_MOD_24 } > 16 {
                value = self.propagate_carries_r96x16(value)
            }
            value
        }

        #[inline(always)]
        pub fn shl_dyn_small_r96x16(self, value: R96x16, amount: u32x16) -> R96x16 {
            #[cfg(debug_assertions)]
            {
                let amount_: [u32; 16] = pulp::cast(amount);
                if max_u32s(&amount_) > 24 {
                    panic!("shift amount exceeds 24: {:?}", amount);
                }
            }

            let one: i32x16 = pulp::cast(const { [1i32; 16] });
            let twenty_four: i32x16 = pulp::cast(const { [24_i32; 16] });

            let twenty_four_minus_amount =
                self.wrapping_sub_i32x16(twenty_four, pulp::cast(amount));

            let mask: i32x16 = self.wrapping_sub_i32x16(
                self.shl_dyn_i32x16(one, pulp::cast(twenty_four_minus_amount)),
                one,
            );

            let value = R96x16 {
                x: self.wrapping_sub_i32x16(
                    self.shl_dyn_i32x16(self.and_i32x16(value.x, mask), amount),
                    self.shr_dyn_i32x16(value.w, twenty_four_minus_amount),
                ),
                y: self.wrapping_add_i32x16(
                    self.shl_dyn_i32x16(self.and_i32x16(value.y, mask), amount),
                    self.shr_dyn_i32x16(value.x, twenty_four_minus_amount),
                ),
                z: self.wrapping_add_i32x16(
                    self.shl_dyn_i32x16(self.and_i32x16(value.z, mask), amount),
                    self.shr_dyn_i32x16(value.y, twenty_four_minus_amount),
                ),
                w: self.wrapping_add_i32x16(
                    self.shl_dyn_i32x16(self.and_i32x16(value.w, mask), amount),
                    self.shr_dyn_i32x16(value.z, twenty_four_minus_amount),
                ),
            };

            self.propagate_carries_r96x16(value)
        }

        #[inline(always)]
        pub fn mul_zpx16(
            self,
            lhs_lo: u32x16,
            lhs_hi: u32x16,
            rhs_lo: u32x16,
            rhs_hi: u32x16,
        ) -> (u32x16, u32x16) {
            // pl = 1
            let pl: u32x16 = pulp::cast(const { [P as u32; 16] });
            // ph = u32::MAX
            let ph: u32x16 = pulp::cast(const { [(P >> 32) as u32; 16] });
            let zero: u32x16 = pulp::cast(Self::ZERO);
            let max = ph;

            let lolo = self.widening_mul_u32x16(lhs_lo, rhs_lo);
            let lohi = self.widening_mul_u32x16(lhs_lo, rhs_hi);
            let hilo = self.widening_mul_u32x16(lhs_hi, rhs_lo);
            let hihi = self.widening_mul_u32x16(lhs_hi, rhs_hi);

            let l = lolo.0;

            let ml = self.wrapping_add_u32x16(lohi.0, hilo.0);
            let o0 = self.u32x16_from_b16(self.cmp_lt_u32x16(ml, lohi.0));

            let ml = self.wrapping_add_u32x16(ml, lolo.1);
            let o1 = self.u32x16_from_b16(self.cmp_lt_u32x16(ml, lolo.1));

            let mh = self.wrapping_sub_u32x16(lohi.1, o0);
            let mh = self.wrapping_sub_u32x16(mh, o1);
            let o0 =
                self.u32x16_from_b16(self.cmp_eq_u32x16(o0, max) & self.cmp_eq_u32x16(mh, zero));

            let mh = self.wrapping_add_u32x16(mh, hilo.1);
            let o1 = self.u32x16_from_b16(self.cmp_lt_u32x16(mh, hilo.1));

            let mh = self.wrapping_add_u32x16(mh, hihi.0);
            let o2 = self.u32x16_from_b16(self.cmp_lt_u32x16(mh, hihi.0));

            let h = hihi.1;
            let h = self.wrapping_sub_u32x16(h, o0);
            let h = self.wrapping_sub_u32x16(h, o1);
            let h = self.wrapping_sub_u32x16(h, o2);

            let (l, borrow) = (
                self.wrapping_sub_u32x16(l, h),
                self.u32x16_from_b16(self.cmp_lt_u32x16(l, h)),
            );

            let (ml, borrow) = (
                self.wrapping_add_u32x16(ml, borrow),
                self.and_u32x16(borrow, self.u32x16_from_b16(self.cmp_eq_u32x16(ml, zero))),
            );

            let overflow =
                self.and_u32x16(borrow, self.u32x16_from_b16(self.cmp_eq_u32x16(l, max)));
            let l = self.wrapping_add_u32x16(l, self.and_u32x16(borrow, pl));
            let ml = self.wrapping_add_u32x16(ml, self.and_u32x16(borrow, ph));
            let ml = self.wrapping_sub_u32x16(ml, overflow);

            let borrow = self.u32x16_from_b16(self.cmp_lt_u32x16(l, mh));
            let l = self.wrapping_sub_u32x16(l, mh);

            let ml = self.wrapping_add_u32x16(ml, borrow);
            let ml = self.wrapping_add_u32x16(ml, mh);

            let lo = l;
            let hi = ml;

            let less_than_hi = self.cmp_lt_u32x16(hi, mh);
            let eq_hi = self.cmp_eq_u32x16(hi, mh);
            let less_than_lo = self.cmp_lt_u32x16(lo, self.wrapping_sub_u32x16(zero, mh));

            let less_than = self.u32x16_from_b16(less_than_hi | (eq_hi & less_than_lo));

            let (lo, borrow) = (
                self.wrapping_sub_u32x16(lo, self.and_u32x16(less_than, pl)),
                self.and_u32x16(
                    less_than,
                    self.u32x16_from_b16(self.cmp_eq_u32x16(lo, zero)),
                ),
            );
            let hi = self.wrapping_add_u32x16(hi, borrow);
            let hi = self.wrapping_sub_u32x16(hi, self.and_u32x16(less_than, ph));

            let less_than_hi = self.cmp_lt_u32x16(hi, ph);
            let eq_hi = self.cmp_eq_u32x16(hi, ph);
            let less_than_lo = self.cmp_lt_u32x16(lo, pl);

            let less_than = self.u32x16_from_b16(less_than_hi | (eq_hi & less_than_lo));

            let greater_eq = self.not_u32x16(less_than);

            let (lo, borrow) = (
                self.wrapping_sub_u32x16(lo, self.and_u32x16(greater_eq, pl)),
                self.and_u32x16(
                    greater_eq,
                    self.u32x16_from_b16(self.cmp_eq_u32x16(lo, zero)),
                ),
            );
            let hi = self.wrapping_add_u32x16(hi, borrow);
            let hi = self.wrapping_sub_u32x16(hi, self.and_u32x16(greater_eq, ph));

            (lo, hi)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::{
            prime64::{generic_solinas::PrimeModulusV3, Solinas},
            solinas::prototype::R96,
        };

        use super::*;
        use rand::prelude::*;

        #[test]
        fn test_shl() {
            let rng = &mut StdRng::seed_from_u64(0);
            let mut gen = || R96::rand(rng).to_zp();

            if let Some(simd) = V3::try_new() {
                let [x0, x1]: [u64x4; 2] = pulp::cast([(); 8].map(|()| gen()));
                let x = simd.r96x8_from_u64x4x2(x0, x1);

                let shl24 = shl_const!(simd, 24);
                let shl6 = shl_const!(simd, 6);
                let shl186 = shl_const!(simd, 186);

                let x24 = shl24(x);

                assert_eq!(x24.x, simd.wrapping_sub_i32x8(V3::ZERO, x.w));
                assert_eq!(x24.y, x.x);
                assert_eq!(x24.z, x.y);
                assert_eq!(x24.w, x.z);

                assert_eq!(shl186(shl6(x)), x);
            }
        }

        #[test]
        fn test_mul() {
            let rng = &mut StdRng::seed_from_u64(0);
            let mut gen = || R96::rand(rng).to_zp();

            if let Some(simd) = V3::try_new() {
                for _ in 0..10000 {
                    let [lhs_0, lhs_1]: [u64x4; 2] = pulp::cast([(); 8].map(|()| gen()));
                    let [rhs_0, rhs_1]: [u64x4; 2] = pulp::cast([(); 8].map(|()| gen()));

                    let (lhs_lo, lhs_hi) = simd.u32x8x2_from_u64x4x2(lhs_0, lhs_1);
                    let (rhs_lo, rhs_hi) = simd.u32x8x2_from_u64x4x2(rhs_0, rhs_1);
                    let (lo, hi) = simd.mul_zpx8(lhs_lo, lhs_hi, rhs_lo, rhs_hi);

                    let [lo_0, lo_1]: [pulp::u32x4; 2] = pulp::cast(lo);
                    let [hi_0, hi_1]: [pulp::u32x4; 2] = pulp::cast(hi);

                    let actual_0 = simd.or_u64x4(
                        simd.convert_u32x4_to_u64x4(lo_0),
                        simd.shl_const_u64x4::<32>(simd.convert_u32x4_to_u64x4(hi_0)),
                    );
                    let actual_1 = simd.or_u64x4(
                        simd.convert_u32x4_to_u64x4(lo_1),
                        simd.shl_const_u64x4::<32>(simd.convert_u32x4_to_u64x4(hi_1)),
                    );

                    let expected_0 = <Solinas as PrimeModulusV3>::mul((), simd, lhs_0, rhs_0);
                    let expected_1 = <Solinas as PrimeModulusV3>::mul((), simd, lhs_1, rhs_1);

                    assert_eq!(actual_0, expected_0);
                    assert_eq!(actual_1, expected_1);
                }
            }
        }
    }

    pub fn ntt_32(p: &mut [i32x4; 32]) {
        let simd = V3::try_new().unwrap();

        struct Impl<'a> {
            simd: V3,
            p: &'a mut [i32x4; 32],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self { simd, p } = self;
                simd.ntt_32(p);
            }
        }

        simd.vectorize(Impl { simd, p })
    }

    pub fn ntt_1024(p: &mut [i32x4], omega: &[u64]) {
        let simd = V3::try_new().unwrap();

        struct Impl<'a> {
            simd: V3,
            p: &'a mut [i32x4; 1024],
            omega: &'a [u64; 1024],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self { simd, p, omega } = self;

                simd.ntt_32xn(p, 32);

                {
                    let p = pulp::as_arrays_mut::<4, _>(p).0;
                    let omega: &[u64x4] = bytemuck::cast_slice(pulp::as_arrays::<4, _>(omega).0);

                    for (p, omega) in izip!(&mut *p, omega) {
                        let mut x = simd.interleaved_zpx4_from_i32x4x4(pulp::cast(*p));
                        x = <Solinas as PrimeModulusV3>::mul((), simd, x, *omega);
                        *p = pulp::cast(simd.i32x4x4_from_interleaved_zpx4(x));
                    }
                }

                for p in pulp::as_arrays_mut::<32, _>(p).0 {
                    simd.ntt_32(p);
                }
            }
        }

        let p = p.try_into().unwrap();
        let omega = omega.try_into().unwrap();

        simd.vectorize(Impl { simd, p, omega })
    }
}
