use concrete_ntt::{prime::largest_prime_in_arithmetic_progression64, *};
use criterion::*;

fn criterion_bench(c: &mut Criterion) {
    for n in [1024, 16 * 1024, 32 * 1024] {
        let mut data = vec![0; n];
        for p in [
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 29, 1 << 30).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 30, 1 << 31).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 31, 1 << 32).unwrap(),
        ] {
            let p = p as u32;
            let plan = _32::Plan::try_new(n, p).unwrap();
            c.bench_function(&format!("fwd-32-{p}-{n}"), |b| {
                b.iter(|| plan.fwd(&mut data));
            });
            c.bench_function(&format!("inv-32-{p}-{n}"), |b| {
                b.iter(|| plan.inv(&mut data));
            });
        }
    }

    for n in [1024, 16 * 1024, 32 * 1024] {
        let mut data = vec![0; n];
        for p in [
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 49, 1 << 50).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 50, 1 << 51).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 61, 1 << 62).unwrap(),
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 62, 1 << 63).unwrap(),
            _64::Solinas::P,
            largest_prime_in_arithmetic_progression64(1 << 16, 1, 1 << 63, u64::MAX).unwrap(),
        ] {
            let plan = _64::Plan::try_new(n, p).unwrap();
            c.bench_function(&format!("fwd-64-{p}-{n}"), |b| {
                b.iter(|| plan.fwd(&mut data));
            });
            c.bench_function(&format!("inv-64-{p}-{n}"), |b| {
                b.iter(|| plan.inv(&mut data));
            });
        }
    }
}

criterion_group!(benches, criterion_bench);
criterion_main!(benches);
