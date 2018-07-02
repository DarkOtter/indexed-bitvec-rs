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

fn random_data(n_bits: u64) -> IndexedBits<Vec<u8>> {
    use rand::{Rng, SeedableRng, XorShiftRng};
    let n_bytes: usize = (n_bits / 8) as usize + 1;
    let seed = [42, 3497651, 341723721, 1829743298];
    let mut rng = XorShiftRng::from_seed(seed);
    let data = {
        let mut data = vec![0u8; n_bytes];
        rng.fill_bytes(&mut data);
        Bits::from(data, n_bits).unwrap()
    };
    IndexedBits::build_index(data)
}

fn random_data_512mb() -> IndexedBits<Vec<u8>> {
    random_data(1000 * 1000 * 8 * 512)
}

fn count_bits(c: &mut Criterion) {
    let data = random_data_512mb();
    c.bench_function("build_sequential", move |b| {
        b.iter(|| {
            assert_eq!(
                data.count::<OneBits>() + data.count::<ZeroBits>(),
                data.bits().used_bits()
            )
        })
    });
}

criterion_group!(benches, count_bits);
criterion_main!(benches);
