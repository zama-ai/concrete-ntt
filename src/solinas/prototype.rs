use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign},
};

mod util {
    #[inline]
    pub const fn u32_min(a: u32, b: u32) -> u32 {
        if a < b {
            a
        } else {
            b
        }
    }
}

/// 2^64 - 2^32 + 1
pub const P: u64 = 0xFFFF_FFFF_0000_0001;

// (2^24)^8 = 1 mod p
pub const OMEGA_8_SHL_AMOUNT: u32 = 24;
// (2^3)^64 = 1 mod p
pub const OMEGA_64_SHL_AMOUNT: u32 = 3;

/// Redundant repr
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct R96 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
    pub available_bits: u32,
}

impl fmt::Debug for R96 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_zp().fmt(f)
    }
}

impl PartialEq for R96 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_zp() == other.to_zp()
    }
}
impl Eq for R96 {}

impl R96 {
    pub const MAX_AVAILABLE_BITS: u32 = 7;

    pub const ZERO: Self = Self::from_zp(0);

    #[cfg(test)]
    pub fn rand(rng: &mut impl rand::RngCore) -> Self {
        use rand::distributions::Distribution;

        let dist = rand::distributions::Uniform::new(0, P);
        Self::from_zp(dist.sample(rng))
    }

    #[allow(dead_code)]
    pub const fn repr(self) -> impl Copy + fmt::Debug {
        /// Redundant repr
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug)]
        #[repr(C)]
        struct RR {
            x: i32,
            y: i32,
            z: i32,
            w: i32,
            available_bits: u32,
        }

        RR {
            x: self.x,
            y: self.y,
            z: self.z,
            w: self.w,
            available_bits: self.available_bits,
        }
    }

    #[inline]
    #[track_caller]
    pub const fn from_zp(value: u64) -> Self {
        assert!(value < P);
        Self {
            x: ((value >> 0) & 0xFFFFFF) as i32,
            y: ((value >> 24) & 0xFFFFFF) as i32,
            z: ((value >> 48) & 0xFFFFFF) as i32,
            w: 0,
            available_bits: Self::MAX_AVAILABLE_BITS,
        }
    }

    #[inline]
    pub const fn to_zp(self) -> u64 {
        // bits [0, 32)
        let x = self.x as i64;
        // bits [24, 56)
        let y = (self.y as i64) << 24;
        // bits [48, 64)
        let z0 = (self.z as u32 as u64 & 0xFFFF) << 48;
        // bits [64, 80)
        let z1 = self.z as i64 >> 16;
        // bits [72, 96)
        let w0 = (self.w as u32 as u64 & 0xFFFFFF) << 8;
        // bits [96, 104)
        let w1 = self.w as i64 >> 24;

        // -w1 because 2^96 = -1 mod p
        let a = x + y - w1;
        let a = if a >= 0 {
            a as u64
        } else {
            (a as u64).wrapping_add(P)
        };

        let b = z0;

        // 2^64 = 2^32 - 1 mod p
        let mut c = ((z1 << 32) - z1) as u64;
        if (c as i64) < 0 {
            c = c.wrapping_add(P)
        };

        let d = (w0 << 32) - w0;

        // result = a + b + c + d
        let mut ab = a.wrapping_add(b);
        if ab < a {
            ab = ab.wrapping_sub(P);
        }

        let mut cd = c.wrapping_add(d);
        if cd < c {
            cd = cd.wrapping_sub(P);
        }

        let mut abcd = ab.wrapping_add(cd);
        if abcd < ab {
            abcd = abcd.wrapping_sub(P);
        }
        if abcd >= P {
            abcd -= P
        }

        abcd
    }

    #[inline]
    #[track_caller]
    pub const fn add_const(self, rhs: Self) -> Self {
        assert!(self.available_bits != 0);
        assert!(rhs.available_bits != 0);

        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
            available_bits: util::u32_min(self.available_bits, rhs.available_bits) - 1,
        }
    }

    #[inline]
    #[track_caller]
    pub const fn neg_const(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
            available_bits: self.available_bits,
        }
    }

    #[inline]
    #[track_caller]
    pub const fn sub_const(self, rhs: Self) -> Self {
        assert!(self.available_bits != 0);
        assert!(rhs.available_bits != 0);

        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
            available_bits: util::u32_min(self.available_bits, rhs.available_bits) - 1,
        }
    }

    #[inline]
    pub const fn propagate_carries(self) -> Self {
        let lo = Self {
            x: self.x & 0xFFFFFF,
            y: self.y & 0xFFFFFF,
            z: self.z & 0xFFFFFF,
            w: self.w & 0xFFFFFF,
            available_bits: Self::MAX_AVAILABLE_BITS,
        };
        let hi = Self {
            x: -(self.w >> 24),
            y: self.x >> 24,
            z: self.y >> 24,
            w: self.z >> 24,
            available_bits: Self::MAX_AVAILABLE_BITS,
        };
        lo.add_const(hi)
    }

    #[inline]
    pub const fn mul_const(self, rhs: Self) -> Self {
        let a = self.to_zp();
        let b = rhs.to_zp();

        let mul = a as u128 * b as u128;

        let l = mul as u64;
        let m = (mul >> 64) as u32 as u64;
        let h = (mul >> 96) as u32 as u64;

        let m = (m << 32) - m;

        let mut lh = l.wrapping_sub(h);

        if l < h {
            lh = lh.wrapping_add(P);
        }

        let mut lmh = lh.wrapping_add(m);
        if lmh < m {
            lmh = lmh.wrapping_sub(P);
        }
        if lmh >= P {
            lmh -= P;
        }

        Self::from_zp(lmh)
    }

    #[inline]
    pub const fn shl_const(self, rhs: u32) -> Self {
        let rhs = rhs % 192;
        let rot = rhs / 24;
        let k = rhs % 24;

        let r = self;

        let r = match rot {
            0 => Self { ..r },
            1 => Self {
                x: -r.w,
                y: r.x,
                z: r.y,
                w: r.z,
                ..r
            },
            2 => Self {
                x: -r.z,
                y: -r.w,
                z: r.x,
                w: r.y,
                ..r
            },
            3 => Self {
                x: -r.y,
                y: -r.z,
                z: -r.w,
                w: r.x,
                ..r
            },
            4 => Self {
                x: -r.x,
                y: -r.y,
                z: -r.z,
                w: -r.w,
                ..r
            },
            5 => Self {
                x: r.w,
                y: -r.x,
                z: -r.y,
                w: -r.z,
                ..r
            },
            6 => Self {
                x: r.z,
                y: r.w,
                z: -r.x,
                w: -r.y,
                ..r
            },
            7 => Self {
                x: r.y,
                y: r.z,
                z: r.w,
                w: -r.x,
                ..r
            },
            _ => unreachable!(),
        };

        let mask = ((1u32 << (24 - k)) - 1) as i32;
        let r = Self {
            x: ((r.x & mask) << k) - (r.w >> (24 - k)),
            y: ((r.y & mask) << k) + (r.x >> (24 - k)),
            z: ((r.z & mask) << k) + (r.y >> (24 - k)),
            w: ((r.w & mask) << k) + (r.z >> (24 - k)),
            available_bits: Self::MAX_AVAILABLE_BITS - 1,
        };

        if k > 16 {
            r.propagate_carries()
        } else {
            r
        }
    }

    #[inline]
    pub const fn shr_const(self, rhs: u32) -> Self {
        let rhs = rhs % 192;
        self.shl_const(192 - rhs)
    }

    #[inline]
    pub const fn mul_by_w8(self, pow: i32) -> Self {
        self.shl_const((pow as u32 % 8) * OMEGA_8_SHL_AMOUNT)
    }

    #[inline]
    pub const fn mul_by_w64(self, pow: i32) -> Self {
        self.shl_const((pow as u32 % 64) * OMEGA_64_SHL_AMOUNT)
    }
}

impl Neg for R96 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.neg_const()
    }
}

impl Add for R96 {
    type Output = Self;

    #[inline]
    #[track_caller]
    fn add(self, rhs: Self) -> Self::Output {
        assert_ne!(self.available_bits, 0);
        assert_ne!(rhs.available_bits, 0);
        self.add_const(rhs)
    }
}
impl AddAssign for R96 {
    #[inline]
    #[track_caller]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for R96 {
    type Output = Self;

    #[inline]
    #[track_caller]
    fn sub(self, rhs: Self) -> Self::Output {
        assert_ne!(self.available_bits, 0);
        assert_ne!(rhs.available_bits, 0);
        self.sub_const(rhs)
    }
}
impl SubAssign for R96 {
    #[inline]
    #[track_caller]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for R96 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_const(rhs)
    }
}
impl MulAssign for R96 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Shl<u32> for R96 {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: u32) -> Self::Output {
        self.shl_const(rhs)
    }
}
impl ShlAssign<u32> for R96 {
    #[inline]
    fn shl_assign(&mut self, rhs: u32) {
        *self = (*self) << rhs;
    }
}

impl Shr<u32> for R96 {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: u32) -> Self::Output {
        self.shr_const(rhs)
    }
}
impl ShrAssign<u32> for R96 {
    #[inline]
    fn shr_assign(&mut self, rhs: u32) {
        *self = (*self) >> rhs;
    }
}

pub mod cyclic_4 {
    use super::R96;

    pub fn ntt(data: &mut [R96]) {
        assert_eq!(data.len(), 4);
        let p: &mut [R96; 4] = data.try_into().unwrap();

        *p = [
            //
            p[0] + p[2],
            p[1] + p[3],
            p[0] - p[2],
            (p[1] - p[3]).mul_by_w8(2),
        ];

        *p = [
            //
            p[0] + p[1],
            p[0] - p[1],
            p[2] + p[3],
            p[2] - p[3],
        ];
    }

    pub fn intt(data: &mut [R96]) {
        assert_eq!(data.len(), 4);
        let p: &mut [R96; 4] = data.try_into().unwrap();

        *p = [
            //
            p[0] + p[1],
            p[0] - p[1],
            p[2] + p[3],
            (p[2] - p[3]).mul_by_w8(6),
        ];

        *p = [
            //
            p[0] + p[2],
            p[1] + p[3],
            p[0] - p[2],
            p[1] - p[3],
        ];
    }
}

pub mod negacyclic_4 {
    use super::R96;

    pub fn ntt(data: &mut [R96]) {
        assert_eq!(data.len(), 4);
        let p: &mut [R96; 4] = data.try_into().unwrap();

        let t2 = p[2].mul_by_w8(2);
        let t3 = p[3].mul_by_w8(2);
        *p = [
            //
            p[0] + t2,
            (p[1] + t3).mul_by_w8(1),
            p[0] - t2,
            (p[1] - t3).mul_by_w8(3),
        ];

        *p = [
            //
            p[0] + p[1],
            p[0] - p[1],
            p[2] + p[3],
            p[2] - p[3],
        ];
    }

    pub fn intt(data: &mut [R96]) {
        assert_eq!(data.len(), 4);
        let p: &mut [R96; 4] = data.try_into().unwrap();

        *p = [
            p[0] + p[1],
            (p[0] - p[1]).mul_by_w8(7),
            p[2] + p[3],
            (p[2] - p[3]).mul_by_w8(5),
        ];

        *p = [
            p[0] + p[2],
            p[1] + p[3],
            (p[0] - p[2]).mul_by_w8(6),
            (p[1] - p[3]).mul_by_w8(6),
        ];
    }
}

pub mod cyclic_8 {
    use super::R96;

    pub fn ntt(data: &mut [R96]) {
        assert_eq!(data.len(), 8);
        let p: &mut [R96; 8] = data.try_into().unwrap();

        *p = [
            p[0] + p[4],
            p[1] + p[5],
            p[2] + p[6],
            p[3] + p[7],
            p[0] - p[4],
            p[1] - p[5],
            (p[2] - p[6]).mul_by_w8(2),
            (p[3] - p[7]).mul_by_w8(2),
        ];

        *p = [
            p[0] + p[2],
            p[1] + p[3],
            p[0] - p[2],
            (p[1] - p[3]).mul_by_w8(2),
            p[4] + p[6],
            (p[5] + p[7]).mul_by_w8(1),
            p[4] - p[6],
            (p[5] - p[7]).mul_by_w8(3),
        ];

        *p = [
            p[0] + p[1],
            p[0] - p[1],
            p[2] + p[3],
            p[2] - p[3],
            p[4] + p[5],
            p[4] - p[5],
            p[6] + p[7],
            p[6] - p[7],
        ];
    }

    pub fn intt(data: &mut [R96]) {
        assert_eq!(data.len(), 8);
        let p: &mut [R96; 8] = data.try_into().unwrap();

        *p = [
            p[0] + p[1],
            p[0] - p[1],
            p[2] + p[3],
            (p[2] - p[3]).mul_by_w8(6),
            p[4] + p[5],
            (p[4] - p[5]).mul_by_w8(7),
            p[6] + p[7],
            (p[6] - p[7]).mul_by_w8(5),
        ];

        *p = [
            p[0] + p[2],
            p[1] + p[3],
            p[0] - p[2],
            p[1] - p[3],
            p[4] + p[6],
            p[5] + p[7],
            (p[4] - p[6]).mul_by_w8(6),
            (p[5] - p[7]).mul_by_w8(6),
        ];

        *p = [
            p[0] + p[4],
            p[1] + p[5],
            p[2] + p[6],
            p[3] + p[7],
            p[0] - p[4],
            p[1] - p[5],
            p[2] - p[6],
            p[3] - p[7],
        ];
    }
}

pub mod cyclic_32 {
    use super::R96;

    pub fn ntt(data: &mut [R96]) {
        assert_eq!(data.len(), 32);

        for i in 0..8 {
            let data = &mut data[i..];
            let mut p = [R96::ZERO; 4];
            for j in 0..4 {
                p[j] = data[8 * j];
            }
            super::cyclic_4::ntt(&mut p);
            for j in 0..4 {
                data[8 * j] = p[j];
            }
        }

        for (i, data) in data.chunks_exact_mut(8).enumerate() {
            let i = i as u32;
            let i_rev = ((i & 1) << 1) | (i >> 1);
            let s = 2 * i_rev;

            for (j, x) in data.iter_mut().enumerate() {
                let j = j as u32;

                *x = (*x).mul_by_w64((s * j) as i32);
            }
        }

        for i in 0..4 {
            let data = &mut data[8 * i..];
            super::cyclic_8::ntt(&mut data[..8]);
        }
    }

    pub fn intt(data: &mut [R96]) {
        assert_eq!(data.len(), 32);

        for i in 0..4 {
            let data = &mut data[8 * i..];
            super::cyclic_8::intt(&mut data[..8]);
        }

        for (i, data) in data.chunks_exact_mut(8).enumerate() {
            let i = i as u32;
            let i_rev = ((i & 1) << 1) | (i >> 1);
            let s = 2 * i_rev;

            for (j, x) in data.iter_mut().enumerate() {
                let j = j as u32;

                *x = (*x).mul_by_w64(-((s * j) as i32));
            }
        }

        for i in 0..8 {
            let data = &mut data[i..];
            let mut p = [R96::ZERO; 4];
            for j in 0..4 {
                p[j] = data[8 * j];
            }
            super::cyclic_4::intt(&mut p);
            for j in 0..4 {
                data[8 * j] = p[j];
            }
        }
    }
}

pub mod negacyclic_32 {
    use super::R96;

    pub fn ntt(data: &mut [R96]) {
        assert_eq!(data.len(), 32);

        for i in 0..8 {
            let data = &mut data[i..];
            let mut p = [R96::ZERO; 4];
            for j in 0..4 {
                p[j] = data[8 * j];
            }
            super::negacyclic_4::ntt(&mut p);
            for j in 0..4 {
                data[8 * j] = p[j];
            }
        }

        for (i, data) in data.chunks_exact_mut(8).enumerate() {
            let i = i as u32;
            let i_rev = ((i & 1) << 1) | (i >> 1);
            let s = 2 * i_rev + 1;

            for (j, x) in data.iter_mut().enumerate() {
                let j = j as u32;

                *x = (*x).mul_by_w64((s * j) as i32);
            }
        }

        for i in 0..4 {
            let data = &mut data[8 * i..];
            super::cyclic_8::ntt(&mut data[..8]);
        }
    }

    pub fn intt(data: &mut [R96]) {
        assert_eq!(data.len(), 32);

        for i in 0..4 {
            let data = &mut data[8 * i..];
            super::cyclic_8::intt(&mut data[..8]);
        }

        for (i, data) in data.chunks_exact_mut(8).enumerate() {
            let i = i as u32;
            let i_rev = ((i & 1) << 1) | (i >> 1);
            let s = 2 * i_rev + 1;

            for (j, x) in data.iter_mut().enumerate() {
                let j = j as u32;

                *x = (*x).mul_by_w64(-((s * j) as i32));
            }
        }

        for i in 0..8 {
            let data = &mut data[i..];
            let mut p = [R96::ZERO; 4];
            for j in 0..4 {
                p[j] = data[8 * j];
            }
            super::negacyclic_4::intt(&mut p);
            for j in 0..4 {
                data[8 * j] = p[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::iter;
    use rand::prelude::*;

    extern crate alloc;
    type Box = alloc::boxed::Box<[R96]>;

    fn negacyclic_polymul(x: &[R96], y: &[R96]) -> Box {
        assert_eq!(x.len(), y.len());
        let n = x.len();

        let mut prod = vec![R96::ZERO; 2 * n];
        for i in 0..n {
            for j in 0..n {
                prod[i + j] = (prod[i + j] + x[i] * y[j]).propagate_carries();
            }
        }

        let (head, tail) = prod.split_at(n);
        iter::zip(head, tail).map(|(&x, &y)| x - y).collect()
    }

    fn cyclic_polymul(x: &[R96], y: &[R96]) -> Box {
        assert_eq!(x.len(), y.len());
        let n = x.len();

        let mut prod = vec![R96::ZERO; 2 * n];
        for i in 0..n {
            for j in 0..n {
                prod[i + j] = (prod[i + j] + x[i] * y[j]).propagate_carries();
            }
        }

        let (head, tail) = prod.split_at(n);
        iter::zip(head, tail).map(|(&x, &y)| x + y).collect()
    }

    #[test]
    fn test_shift() {
        let rng = &mut StdRng::seed_from_u64(0);

        for _ in 0..100 {
            let n = R96::rand(rng);

            let mut n_pow2 = n;
            for i in 0..200 {
                let target = n << i;
                assert_eq!(target, n_pow2);
                assert_eq!(target >> i, n);

                n_pow2 = (n_pow2 + n_pow2).propagate_carries();
            }
        }
    }

    #[test]
    fn test_ntt4() {
        let rng = &mut StdRng::seed_from_u64(0);

        let n = 4usize;

        let mut gen = || R96::rand(rng);
        let p = &mut (0..n).map(|_| gen()).collect::<Box>();
        let q = &mut (0..n).map(|_| gen()).collect::<Box>();

        let scale = R96::from_zp(n as u64);

        {
            let mut fwd = p.clone();
            cyclic_4::ntt(&mut fwd);

            let mut roundtrip = fwd.iter().map(|&x| x.propagate_carries()).collect::<Box>();
            cyclic_4::intt(&mut roundtrip);

            let scaled = p.iter().map(|&x| x * scale).collect::<Box>();

            assert_eq!(scaled, roundtrip);
        }

        {
            let mut p_fwd = p.clone();
            cyclic_4::ntt(&mut p_fwd);
            let mut q_fwd = q.clone();
            cyclic_4::ntt(&mut q_fwd);

            let prod = &mut *iter::zip(&p_fwd, &q_fwd)
                .map(|(&p, &q)| p * q)
                .collect::<Box>();
            cyclic_4::intt(prod);
            let prod = &*prod;

            let scaled = &*cyclic_polymul(&p, &q)
                .iter()
                .map(|&x| x * scale)
                .collect::<Box>();
            assert_eq!(scaled, prod);
        }
    }

    #[test]
    fn test_ng_ntt4() {
        let rng = &mut StdRng::seed_from_u64(0);

        let n = 4usize;

        let mut gen = || R96::rand(rng);
        let p = &mut (0..n).map(|_| gen()).collect::<Box>();
        let q = &mut (0..n).map(|_| gen()).collect::<Box>();
        let scale = R96::from_zp(n as u64);

        {
            let mut fwd = p.clone();
            negacyclic_4::ntt(&mut fwd);

            let mut roundtrip = fwd.iter().map(|&x| x.propagate_carries()).collect::<Box>();
            negacyclic_4::intt(&mut roundtrip);

            let scaled = p.iter().map(|&x| x * scale).collect::<Box>();
            assert_eq!(scaled, roundtrip);
        }

        {
            let mut p_fwd = p.clone();
            negacyclic_4::ntt(&mut p_fwd);
            let mut q_fwd = q.clone();
            negacyclic_4::ntt(&mut q_fwd);

            let prod = &mut *iter::zip(&p_fwd, &q_fwd)
                .map(|(&p, &q)| p * q)
                .collect::<Box>();
            negacyclic_4::intt(prod);
            let prod = &*prod;

            let scaled = &*negacyclic_polymul(&p, &q)
                .iter()
                .map(|&x| x * scale)
                .collect::<Box>();
            assert_eq!(scaled, prod);
        }
    }

    #[test]
    fn test_ntt8() {
        let rng = &mut StdRng::seed_from_u64(0);

        let n = 8usize;

        let mut gen = || R96::rand(rng);
        let p = &mut (0..n).map(|_| gen()).collect::<Box>();
        let q = &mut (0..n).map(|_| gen()).collect::<Box>();

        let scale = R96::from_zp(n as u64);

        {
            let mut fwd = p.clone();
            cyclic_8::ntt(&mut fwd);

            let mut roundtrip = fwd.iter().map(|&x| x.propagate_carries()).collect::<Box>();
            cyclic_8::intt(&mut roundtrip);

            let scaled = p.iter().map(|&x| x * scale).collect::<Box>();

            assert_eq!(scaled, roundtrip);
        }

        {
            let mut p_fwd = p.clone();
            cyclic_8::ntt(&mut p_fwd);
            let mut q_fwd = q.clone();
            cyclic_8::ntt(&mut q_fwd);

            let prod = &mut *iter::zip(&p_fwd, &q_fwd)
                .map(|(&p, &q)| p * q)
                .collect::<Box>();
            cyclic_8::intt(prod);
            let prod = &*prod;

            let scaled = &*cyclic_polymul(&p, &q)
                .iter()
                .map(|&x| x * scale)
                .collect::<Box>();
            assert_eq!(scaled, prod);
        }
    }

    #[test]
    fn test_ntt32() {
        let rng = &mut StdRng::seed_from_u64(0);

        let n = 32usize;

        let mut gen = || R96::rand(rng);
        let p = &mut (0..n).map(|_| gen()).collect::<Box>();
        let q = &mut (0..n).map(|_| gen()).collect::<Box>();
        let scale = R96::from_zp(n as u64);

        {
            let mut fwd = p.clone();
            cyclic_32::ntt(&mut fwd);

            let mut roundtrip = fwd.iter().map(|&x| x.propagate_carries()).collect::<Box>();
            cyclic_32::intt(&mut roundtrip);

            let scaled = p.iter().map(|&x| x * scale).collect::<Box>();
            assert_eq!(scaled, roundtrip);
        }

        {
            let mut p_fwd = p.clone();
            cyclic_32::ntt(&mut p_fwd);
            let mut q_fwd = q.clone();
            cyclic_32::ntt(&mut q_fwd);

            let prod = &mut *iter::zip(&p_fwd, &q_fwd)
                .map(|(&p, &q)| p * q)
                .collect::<Box>();
            cyclic_32::intt(prod);
            let prod = &*prod;

            let scaled = &*cyclic_polymul(&p, &q)
                .iter()
                .map(|&x| x * scale)
                .collect::<Box>();
            assert_eq!(scaled, prod);
        }
    }

    #[test]
    fn test_ng_ntt32() {
        let rng = &mut StdRng::seed_from_u64(0);

        let n = 32usize;

        let mut gen = || R96::rand(rng);
        let p = &mut (0..n).map(|_| gen()).collect::<Box>();
        let q = &mut (0..n).map(|_| gen()).collect::<Box>();
        let scale = R96::from_zp(n as u64);

        {
            let mut fwd = p.clone();
            negacyclic_32::ntt(&mut fwd);

            let mut roundtrip = fwd.iter().map(|&x| x.propagate_carries()).collect::<Box>();
            negacyclic_32::intt(&mut roundtrip);

            let scaled = p.iter().map(|&x| x * scale).collect::<Box>();
            assert_eq!(scaled, roundtrip);
        }

        {
            let mut p_fwd = p.clone();
            negacyclic_32::ntt(&mut p_fwd);
            let mut q_fwd = q.clone();
            negacyclic_32::ntt(&mut q_fwd);

            let prod = &mut *iter::zip(&p_fwd, &q_fwd)
                .map(|(&p, &q)| p * q)
                .collect::<Box>();
            negacyclic_32::intt(prod);
            let prod = &*prod;

            let scaled = &*negacyclic_polymul(&p, &q)
                .iter()
                .map(|&x| x * scale)
                .collect::<Box>();
            assert_eq!(scaled, prod);
        }
    }
}
