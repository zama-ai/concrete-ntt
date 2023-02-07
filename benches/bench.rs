use concrete_ntt::{fastdiv::Div64, *};
use core::iter::zip;
use criterion::*;
use rand::random;

fn criterion_bench(c: &mut Criterion) {
    let n = 1024;

    let mut data = vec![0; n];
    let mut roots = vec![0; n];
    let p = Solinas::P;
    for (x, y) in zip(&mut data, &mut roots) {
        *x = random::<u64>() % p;
        *y = random::<u64>() % p;
    }

    c.bench_function(&format!("fwd-{n}"), |b| {
        let p = Solinas::P;
        let p_div = Div64::new(p);
        b.iter(|| fwd_breadth_first_scalar(&mut data, p, p_div, &roots, 0, 0));
    });

    c.bench_function(&format!("fwd-solinas-{n}"), |b| {
        let p = Solinas;
        let p_div = ();
        b.iter(|| fwd_breadth_first_scalar(&mut data, p, p_div, &roots, 0, 0));
    });

    if let Some(simd) = Avx2::try_new() {
        c.bench_function(&format!("fwd-avx2-{n}"), |b| {
            let p = Solinas::P;
            let p_div = Div64::new(p);
            let u256 { x0, x1, x2, x3 } = p_div.double_reciprocal;
            b.iter(|| {
                fwd_breadth_first_avx2(simd, &mut data, p, (p, x0, x1, x2, x3), &roots, 0, 0)
            });
        });
        c.bench_function(&format!("fwd-solinas-avx2-{n}"), |b| {
            let p = Solinas;
            let p_div = ();
            b.iter(|| fwd_breadth_first_avx2(simd, &mut data, p, p_div, &roots, 0, 0));
        });
    }

    #[cfg(feature = "nightly")]
    if let Some(simd) = Avx512::try_new() {
        c.bench_function(&format!("fwd-avx512-{n}"), |b| {
            let p = Solinas::P;
            let p_div = Div64::new(p);
            let u256 { x0, x1, x2, x3 } = p_div.double_reciprocal;
            b.iter(|| {
                fwd_breadth_first_avx512(simd, &mut data, p, (p, x0, x1, x2, x3), &roots, 0, 0)
            });
        });
        c.bench_function(&format!("fwd-solinas-avx512-{n}"), |b| {
            let p = Solinas;
            let p_div = ();
            b.iter(|| fwd_breadth_first_avx512(simd, &mut data, p, p_div, &roots, 0, 0));
        });
    }
}

criterion_group!(benches, criterion_bench);
criterion_main!(benches);
