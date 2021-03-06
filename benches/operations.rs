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
extern crate rand_xorshift;

use criterion::Criterion;
use indexed_bitvec::*;
use rand::{Rng, RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;

fn rng() -> XorShiftRng {
    let seed = [
        42, 34, 97, 65, 1, 34, 172, 37, 21, 182, 97, 43, 2, 98, 12, 7,
    ];
    XorShiftRng::from_seed(seed)
}

fn random_data(rng: &mut XorShiftRng, n_bits: u64) -> IndexedBits<Vec<u8>> {
    let n_bytes: usize = (n_bits / 8) as usize + 1;
    let mut data = vec![0u8; n_bytes];
    rng.fill_bytes(&mut data);
    IndexedBits::build_from_bytes(data, n_bits).unwrap()
}

fn random_data_1gb(rng: &mut XorShiftRng) -> IndexedBits<Vec<u8>> {
    random_data(rng, 1000 * 1000 * 8 * 1000 + 1289747)
}

fn count_bits(c: &mut Criterion) {
    let data = random_data_1gb(&mut rng());
    c.bench_function("count_bits", move |b| {
        b.iter(|| assert_eq!(data.count_ones() + data.count_zeros(), data.bits().len()))
    });
}

fn rank_times_1000(c: &mut Criterion) {
    let mut rng = rng();
    let data = random_data_1gb(&mut rng);
    let n_bits = data.bits().len();
    let indexes: Vec<_> = (0..1000).map(|_| rng.gen_range(0, n_bits)).collect();
    c.bench_function("rank_times_1000", move |b| {
        b.iter(|| {
            for idx in indexes.iter().cloned() {
                let rank_ones = data.rank_ones(idx).unwrap();
                let rank_zeros = data.rank_zeros(idx).unwrap();
                assert_eq!(rank_ones + rank_zeros, idx)
            }
        })
    });
}

fn select_times_1000(c: &mut Criterion) {
    let mut rng = rng();
    let data = random_data_1gb(&mut rng);
    let count_ones = data.count_ones();
    let count_zeros = data.count_zeros();
    let ones_indexes: Vec<_> = (0..1000).map(|_| rng.gen_range(0, count_ones)).collect();
    let zeros_indexes: Vec<_> = (0..1000).map(|_| rng.gen_range(0, count_zeros)).collect();
    let data_ones = data.clone();
    let data_zeros = data;
    c.bench_function("select_ones_times_1000", move |b| {
        b.iter(|| {
            for idx in ones_indexes.iter().cloned() {
                let select_ones = data_ones.select_ones(idx).unwrap();
                assert!(select_ones >= idx)
            }
        })
    });
    c.bench_function("select_zeros_times_1000", move |b| {
        b.iter(|| {
            for idx in zeros_indexes.iter().cloned() {
                let select_zeros = data_zeros.select_zeros(idx).unwrap();
                assert!(select_zeros >= idx)
            }
        })
    });
}

criterion_group!(benches, count_bits, rank_times_1000, select_times_1000);
criterion_main!(benches);
