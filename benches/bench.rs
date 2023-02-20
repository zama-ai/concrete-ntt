use std::ops::{Add, Mul, Sub};

use concrete_ntt::{prime::largest_prime_in_arithmetic_progression64, *};
use criterion::*;

trait Scalar: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn zero() -> Self;
}
impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Scalar for T {
    fn zero() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

fn slice_wrapping_add<T: Scalar>(output: &mut [T], lhs: &[T], rhs: &[T]) {
    output
        .iter_mut()
        .zip(lhs.iter().zip(rhs.iter()))
        .for_each(|(out, (&lhs, &rhs))| *out = lhs + rhs);
}
fn slice_wrapping_sub<T: Scalar>(output: &mut [T], lhs: &[T], rhs: &[T]) {
    output
        .iter_mut()
        .zip(lhs.iter().zip(rhs.iter()))
        .for_each(|(out, (&lhs, &rhs))| *out = lhs - rhs);
}
fn slice_wrapping_add_assign<T: Scalar>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(lhs, &rhs)| *lhs = *lhs + rhs);
}
fn slice_wrapping_sub_assign<T: Scalar>(lhs: &mut [T], rhs: &[T]) {
    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(lhs, &rhs)| *lhs = *lhs - rhs);
}

fn polynomial_karatsuba_wrapping_mul<T: Scalar>(output: &mut [T], p: &[T], q: &[T]) {
    // check same dimensions
    let poly_size = output.len();

    // allocate slices for the rec
    let mut a0 = vec![T::zero(); poly_size];
    let mut a1 = vec![T::zero(); poly_size];
    let mut a2 = vec![T::zero(); poly_size];
    let mut input_a2_p = vec![T::zero(); poly_size / 2];
    let mut input_a2_q = vec![T::zero(); poly_size / 2];

    // prepare for splitting
    let bottom = 0..(poly_size / 2);
    let top = (poly_size / 2)..poly_size;

    // induction
    induction_karatsuba(&mut a0, &p[bottom.clone()], &q[bottom.clone()]);
    induction_karatsuba(&mut a1, &p[top.clone()], &q[top.clone()]);
    slice_wrapping_add(&mut input_a2_p, &p[bottom.clone()], &p[top.clone()]);
    slice_wrapping_add(&mut input_a2_q, &q[bottom.clone()], &q[top.clone()]);
    induction_karatsuba(&mut a2, &input_a2_p, &input_a2_q);

    // rebuild the result
    slice_wrapping_sub(output, &a0, &a1);
    slice_wrapping_sub_assign(&mut output[bottom.clone()], &a2[top.clone()]);
    slice_wrapping_add_assign(&mut output[bottom.clone()], &a0[top.clone()]);
    slice_wrapping_add_assign(&mut output[bottom.clone()], &a1[top.clone()]);
    slice_wrapping_add_assign(&mut output[top.clone()], &a2[bottom.clone()]);
    slice_wrapping_sub_assign(&mut output[top.clone()], &a0[bottom.clone()]);
    slice_wrapping_sub_assign(&mut output[top], &a1[bottom]);
}

const KARATUSBA_STOP: usize = 64;
/// Compute the induction for the karatsuba algorithm.
fn induction_karatsuba<T: Scalar>(res: &mut [T], p: &[T], q: &[T]) {
    // stop the induction when polynomials have KARATUSBA_STOP elements
    if p.len() <= KARATUSBA_STOP {
        // schoolbook algorithm
        pulp::Arch::new().dispatch(
            #[inline(always)]
            || {
                for (lhs_degree, &lhs_elt) in p.iter().enumerate() {
                    let res = &mut res[lhs_degree..];
                    for (&rhs_elt, res) in q.iter().zip(res) {
                        *res = *res + lhs_elt * rhs_elt
                    }
                }
            },
        );
    } else {
        let poly_size = res.len();

        // allocate slices for the rec
        let mut a0 = vec![T::zero(); poly_size / 2];
        let mut a1 = vec![T::zero(); poly_size / 2];
        let mut a2 = vec![T::zero(); poly_size / 2];
        let mut input_a2_p = vec![T::zero(); poly_size / 4];
        let mut input_a2_q = vec![T::zero(); poly_size / 4];

        // prepare for splitting
        let bottom = 0..(poly_size / 4);
        let top = (poly_size / 4)..(poly_size / 2);

        // rec
        induction_karatsuba(&mut a0, &p[bottom.clone()], &q[bottom.clone()]);
        induction_karatsuba(&mut a1, &p[top.clone()], &q[top.clone()]);
        slice_wrapping_add(&mut input_a2_p, &p[bottom.clone()], &p[top.clone()]);
        slice_wrapping_add(&mut input_a2_q, &q[bottom], &q[top]);
        induction_karatsuba(&mut a2, &input_a2_p, &input_a2_q);

        // rebuild the result
        slice_wrapping_sub(&mut res[(poly_size / 4)..(3 * poly_size / 4)], &a2, &a0);
        slice_wrapping_sub_assign(&mut res[(poly_size / 4)..(3 * poly_size / 4)], &a1);
        slice_wrapping_add_assign(&mut res[0..(poly_size / 2)], &a0);
        slice_wrapping_add_assign(&mut res[(poly_size / 2)..poly_size], &a1);
    }
}

fn criterion_bench(c: &mut Criterion) {
    let ns = [1024, 16 * 1024, 32 * 1024];
    for n in ns {
        let mut data = vec![0; n];
        for p in [
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 29, 1 << 30).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 30, 1 << 31).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 31, 1 << 32).unwrap(),
        ] {
            let p = p as u32;
            let plan = prime32::Plan::try_new(n, p).unwrap();
            c.bench_function(&format!("fwd-32-{p}-{n}"), |b| {
                b.iter(|| plan.fwd(&mut data));
            });
            c.bench_function(&format!("inv-32-{p}-{n}"), |b| {
                b.iter(|| plan.inv(&mut data));
            });
        }
    }

    for n in ns {
        let mut data = vec![0; n];
        for p in [
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 49, 1 << 50).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 50, 1 << 51).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 61, 1 << 62).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63).unwrap(),
            prime64::Solinas::P,
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 63, u64::MAX).unwrap(),
        ] {
            let plan = prime64::Plan::try_new(n, p).unwrap();
            c.bench_function(&format!("fwd-64-{p}-{n}"), |b| {
                b.iter(|| plan.fwd(&mut data));
            });
            c.bench_function(&format!("inv-64-{p}-{n}"), |b| {
                b.iter(|| plan.inv(&mut data));
            });
        }
    }

    for n in ns {
        let mut prod = vec![0; n];
        let lhs = vec![0; n];
        let rhs = vec![0; n];

        let plan = native32::Plan32::try_new(n).unwrap();
        c.bench_function(&format!("native32-32-{n}"), |b| {
            b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
        });
        let plan = native_binary32::Plan32::try_new(n).unwrap();
        c.bench_function(&format!("nativebinary32-32-{n}"), |b| {
            b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
        });

        #[cfg(feature = "nightly")]
        {
            if let Some(plan) = native32::Plan52::try_new(n) {
                c.bench_function(&format!("native32-52-{n}"), |b| {
                    b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
                });
            }
            if let Some(plan) = native_binary32::Plan52::try_new(n) {
                c.bench_function(&format!("nativebinary32-52-{n}"), |b| {
                    b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
                });
            }
        }
        c.bench_function(&format!("native32-karatsuba-{n}"), |b| {
            b.iter(|| polynomial_karatsuba_wrapping_mul(&mut prod, &lhs, &rhs));
        });
    }

    for n in ns {
        let mut prod = vec![0; n];
        let lhs = vec![0; n];
        let rhs = vec![0; n];

        let plan = native64::Plan32::try_new(n).unwrap();
        c.bench_function(&format!("native64-32-{n}"), |b| {
            b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
        });
        let plan = native_binary64::Plan32::try_new(n).unwrap();
        c.bench_function(&format!("nativebinary64-32-{n}"), |b| {
            b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
        });

        #[cfg(feature = "nightly")]
        {
            if let Some(plan) = native64::Plan52::try_new(n) {
                c.bench_function(&format!("native64-52-{n}"), |b| {
                    b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
                });
            }
            if let Some(plan) = native_binary64::Plan52::try_new(n) {
                c.bench_function(&format!("nativebinary64-52-{n}"), |b| {
                    b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
                });
            }
        }
        c.bench_function(&format!("native64-karatsuba-{n}"), |b| {
            b.iter(|| polynomial_karatsuba_wrapping_mul(&mut prod, &lhs, &rhs));
        });
    }

    for n in ns {
        let mut prod = vec![0; n];
        let lhs = vec![0; n];
        let rhs = vec![0; n];

        let plan = native128::Plan32::try_new(n).unwrap();
        c.bench_function(&format!("native128-32-{n}"), |b| {
            b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
        });
        let plan = native_binary128::Plan32::try_new(n).unwrap();
        c.bench_function(&format!("nativebinary128-32-{n}"), |b| {
            b.iter(|| plan.negacyclic_polymul(&mut prod, &lhs, &rhs));
        });
        c.bench_function(&format!("native128-karatsuba-{n}"), |b| {
            b.iter(|| polynomial_karatsuba_wrapping_mul(&mut prod, &lhs, &rhs));
        });
    }
}

criterion_group!(benches, criterion_bench);
criterion_main!(benches);
