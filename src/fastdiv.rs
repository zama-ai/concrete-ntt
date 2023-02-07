use crate::u256;

#[inline(always)]
pub const fn mul64_u32(lowbits: u64, d: u32) -> u32 {
    ((lowbits as u128 * d as u128) >> 64) as u32
}

#[inline(always)]
pub const fn mul128_u64(lowbits: u128, d: u64) -> u64 {
    let mut bottom_half = (lowbits & 0xFFFFFFFFFFFFFFFF) * d as u128;
    bottom_half >>= 64;
    let top_half = (lowbits >> 64) * d as u128;
    let both_halves = bottom_half + top_half;
    (both_halves >> 64) as u64
}

#[inline(always)]
pub const fn mul256_u128(lowbits: u256, d: u128) -> u128 {
    lowbits.mul_u256_u128(d).1
}

#[inline(always)]
pub const fn mul256_u64(lowbits: u256, d: u64) -> u64 {
    lowbits.mul_u256_u64(d).1
}

#[derive(Copy, Clone, Debug)]
pub struct Div32 {
    pub double_reciprocal: u128,
    pub single_reciprocal: u64,
    pub divisor: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct Div64 {
    pub double_reciprocal: u256,
    pub single_reciprocal: u128,
    pub divisor: u64,
}

impl Div32 {
    pub const fn new(divisor: u32) -> Self {
        assert!(divisor > 1);
        let single_reciprocal = (u64::MAX / divisor as u64) + 1;
        let double_reciprocal = (u128::MAX / divisor as u128) + 1;

        Self {
            double_reciprocal,
            single_reciprocal,
            divisor,
        }
    }

    #[inline(always)]
    pub const fn div(n: u32, d: Self) -> u32 {
        mul64_u32(d.single_reciprocal, n)
    }

    #[inline(always)]
    pub const fn rem(n: u32, d: Self) -> u32 {
        let low_bits = d.single_reciprocal.wrapping_mul(n as u64);
        mul64_u32(low_bits, n)
    }

    #[inline(always)]
    pub const fn div_u64(n: u64, d: Self) -> u64 {
        mul128_u64(d.double_reciprocal, n)
    }

    #[inline(always)]
    pub const fn rem_u64(n: u64, d: Self) -> u32 {
        let low_bits = d.double_reciprocal.wrapping_mul(n as u128);
        mul128_u64(low_bits, d.divisor as u64) as u32
    }

    #[inline(always)]
    pub const fn divisor(&self) -> u32 {
        self.divisor
    }
}

impl Div64 {
    pub const fn new(divisor: u64) -> Self {
        assert!(divisor > 1);
        let single_reciprocal = ((u128::MAX) / divisor as u128) + 1;
        let double_reciprocal = u256::MAX
            .div_rem_u256_u64(divisor)
            .0
            .overflowing_add(u256 {
                x0: 1,
                x1: 0,
                x2: 0,
                x3: 0,
            })
            .0;

        Self {
            double_reciprocal,
            single_reciprocal,
            divisor,
        }
    }

    #[inline(always)]
    pub const fn div(n: u64, d: Self) -> u64 {
        mul128_u64(d.single_reciprocal, n)
    }

    #[inline(always)]
    pub const fn rem(n: u64, d: Self) -> u64 {
        let low_bits = d.single_reciprocal.wrapping_mul(n as u128);
        mul128_u64(low_bits, n)
    }

    #[inline(always)]
    pub const fn div_u128(n: u128, d: Self) -> u128 {
        mul256_u128(d.double_reciprocal, n)
    }

    #[inline(always)]
    pub const fn rem_u128(n: u128, d: Self) -> u64 {
        let low_bits = d.double_reciprocal.wrapping_mul_u256_u128(n);
        mul256_u64(low_bits, d.divisor)
    }

    #[inline(always)]
    pub const fn divisor(&self) -> u64 {
        self.divisor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random;

    #[test]
    fn test_div64() {
        for _ in 0..1000 {
            let divisor = loop {
                let d = random();
                if d != 0 {
                    break d;
                }
            };

            let div = Div64::new(divisor);
            let n = random();
            assert_eq!(Div64::div_u128(n, div) as u128, n / divisor as u128);
            assert_eq!(Div64::rem_u128(n, div) as u128, n % divisor as u128);
        }
    }

    #[test]
    fn test_div32() {
        for _ in 0..1000 {
            let divisor = loop {
                let d = random();
                if d != 0 {
                    break d;
                }
            };

            let div = Div32::new(divisor);
            let n = random();
            assert_eq!(Div32::div_u64(n, div) as u64, n / divisor as u64);
            assert_eq!(Div32::rem_u64(n, div) as u64, n % divisor as u64);
        }
    }
}
