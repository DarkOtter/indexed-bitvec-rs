/*
   Copyright 2018 DarkOtter

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
extern crate indexed_bitvec;

#[macro_use]
extern crate criterion;
extern crate rand;

use criterion::Criterion;
use indexed_bitvec::*;
use rand::{Rng, SeedableRng, XorShiftRng};

fn rng() -> XorShiftRng {
    let seed = [42, 3497651, 341723721, 1829743298];
    XorShiftRng::from_seed(seed)
}

fn random_data(rng: &mut XorShiftRng, n_bits: u64) -> IndexedBits<Vec<u8>> {
    let n_bytes: usize = (n_bits / 8) as usize + 1;
    let data = {
        let mut data = vec![0u8; n_bytes];
        rng.fill_bytes(&mut data);
        Bits::from(data, n_bits).unwrap()
    };
    IndexedBits::build_index(data)
}

fn random_data_1gb(rng: &mut XorShiftRng) -> IndexedBits<Vec<u8>> {
    random_data(rng, 1000 * 1000 * 8 * 1000 + 1289747)
}

fn count_bits(c: &mut Criterion) {
    let data = random_data_1gb(&mut rng());
    c.bench_function("count_bits", move |b| {
        b.iter(|| {
            assert_eq!(
                data.count::<OneBits>() + data.count::<ZeroBits>(),
                data.bits().used_bits()
            )
        })
    });
}

fn rank_times_1000(c: &mut Criterion) {
    let mut rng = rng();
    let data = random_data_1gb(&mut rng);
    let n_bits = data.bits().used_bits();
    let indexes: Vec<_> = (0..1000).map(|_| rng.gen_range(0, n_bits)).collect();
    c.bench_function("rank_times_1000", move |b| {
        b.iter(|| for idx in indexes.iter().cloned() {
            let rank_ones = data.rank::<OneBits>(idx).unwrap();
            let rank_zeros = data.rank::<ZeroBits>(idx).unwrap();
            assert_eq!(rank_ones + rank_zeros, idx)
        })
    });
}

fn select_times_1000(c: &mut Criterion) {
    let mut rng = rng();
    let data = random_data_1gb(&mut rng);
    let count_ones = data.count::<OneBits>();
    let count_zeros = data.count::<ZeroBits>();
    let ones_indexes: Vec<_> = (0..1000).map(|_| rng.gen_range(0, count_ones)).collect();
    let zeros_indexes: Vec<_> = (0..1000).map(|_| rng.gen_range(0, count_zeros)).collect();
    let data_ones = data.clone();
    let data_zeros = data;
    c.bench_function("select_ones_times_1000", move |b| {
        b.iter(|| for idx in ones_indexes.iter().cloned() {
            let select_ones = data_ones.select::<OneBits>(idx).unwrap();
            assert!(select_ones >= idx)
        })
    });
    c.bench_function("select_zeros_times_1000", move |b| {
        b.iter(|| for idx in zeros_indexes.iter().cloned() {
            let select_zeros = data_zeros.select::<ZeroBits>(idx).unwrap();
            assert!(select_zeros >= idx)
        })
    });
}

fn build_sequential(c: &mut Criterion) {
    let mut rng = rng();
    let data = random_data_1gb(&mut rng);
    let data = data.decompose();
    c.bench_function("build_sequential", move |b| {
        let data = data.clone_ref();
        b.iter(|| IndexedBits::build_index(data))
    });
}

fn build_part_parallel(c: &mut Criterion) {
    let mut rng = rng();
    let data = random_data_1gb(&mut rng);
    let data = data.decompose();
    c.bench_function("build_part_parallel", move |b| {
        let data = data.clone_ref();
        b.iter(|| IndexedBits::build_index_partially_parallel(data))
    });
}

fn build_full_parallel(c: &mut Criterion) {
    let mut rng = rng();
    let data = random_data_1gb(&mut rng);
    let data = data.decompose();
    c.bench_function("build_full_parallel", move |b| {
        let data = data.clone_ref();
        b.iter(|| IndexedBits::build_index_fully_parallel(data))
    });
}

criterion_group!(
    benches,
    count_bits,
    rank_times_1000,
    select_times_1000,
    build_sequential,
    build_part_parallel,
    build_full_parallel
);
criterion_main!(benches);
