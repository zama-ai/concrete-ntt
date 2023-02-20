Concrete-NTT is a pure Rust high performance number theoretic transform library that processes
vectors of sizes that are powers of two.

This library provides three kinds of NTT:
- The prime NTT computes the transform in a field `Z/pZ` with `p` prime, allowing for
arithmetic operations on the polynomial modulo `p`.
- The native NTT internally computes the transform of the first kind with several primes,
allowing the simulation of arithmetic modulo the product of those primes, and truncates the
result when the inverse transform is desired. The truncated result is guaranteed to be as if
the computations were performed with wrapping arithmetic, as long as the full integer result
would have be smaller than half the product of the primes, in absolute value. It is guaranteed
to be suitable for multiplying two polynomials with arbitrary coefficients, and returns the
result in wrapping arithmetic.
- The native binary NTT is similar to the native NTT, but is optimized for the case where one
of the operands of the multiplication has coefficients in `{0, 1}`.

# Features

- `std` (default): This enables runtime arch detection for accelerated SIMD instructions.
- `nightly`: This enables unstable Rust features to further speed up the NTT, by enabling
AVX512 instructions on CPUs that support them. This feature requires a nightly Rust
toolchain.

# Example

```rust
use concrete_ntt::prime32::Plan;

const N: usize = 32;
let p = 1062862849;
let plan = Plan::try_new(N, p).unwrap();

let data = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31,
];

let mut transformed_fwd = data;
plan.fwd(&mut transformed_fwd);

let mut transformed_inv = transformed_fwd;
plan.inv(&mut transformed_inv);

for (&actual, expected) in transformed_inv.iter().zip(data.iter().map(|x| x * N as u32)) {
    assert_eq!(expected, actual);
}
```
