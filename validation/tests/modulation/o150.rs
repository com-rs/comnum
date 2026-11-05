//! The pseudo-random test sequences of ITU-T Recommendation O.150 (05/96), *General
//! requirements for instrumentation for performance measurements on digital transmission
//! equipment*, Sec. 3 to 5 and Table 1, against `comnum_mapping::source::O150Sequence`.
//!
//! Every sequence is checked over a whole period, streamed from the generator
//! ([`O150Sequence::iter`]):
//!
//! - its period: the order of $x$ modulo the polynomial $x^{n} + x^{k} + 1$ of its clause
//!   (computed in $\mathrm{GF}(2)[x]$, independent of the generator) is $2^{n} - 1$, and the
//!   generated sequence repeats after $2^{n} - 1$ bits;
//! - its longest run of zeros over a period, read cyclically: Table 1 (8, 10, 15, 19, 14,
//!   23, 29, 31);
//! - the first bits of Sec. 5.1, "the first ONE of 9 consecutive ONEs";
//! - the claim of Sec. 3 that an inverted sequence contains $n - 1$ consecutive ONEs at
//!   most.
//!
//! The values are in the card `validation/cards/o150_sec5.yaml`. A period of $2^{31} - 1$
//! bits takes a few seconds in release: the test is a campaign.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::mapping::source::O150Sequence;
use comnum_validation::card::Card;
use serde_json::json;

/// $a b \bmod p$ in $\mathrm{GF}(2)[x]$, with $\deg p = n \le 31$ (bit $i$ is $x^{i}$).
fn mulmod(a: u64, b: u64, p: u64, n: u32) -> u64 {
    let (mut acc, mut a, mut b) = (0u64, a, b);
    while b != 0 {
        if b & 1 == 1 {
            acc ^= a;
        }
        b >>= 1;
        a <<= 1;
        if (a >> n) & 1 == 1 {
            a ^= p;
        }
    }
    acc
}

/// $x^{e} \bmod p$.
fn x_pow(e: u64, p: u64, n: u32) -> u64 {
    let (mut result, mut base, mut e) = (1u64, 2u64, e);
    while e != 0 {
        if e & 1 == 1 {
            result = mulmod(result, base, p, n);
        }
        base = mulmod(base, base, p, n);
        e >>= 1;
    }
    result
}

/// The distinct prime factors of `m` (trial division: $m < 2^{31}$).
fn prime_factors(mut m: u64) -> Vec<u64> {
    let mut f = Vec::new();
    let mut d = 2;
    while d * d <= m {
        if m % d == 0 {
            f.push(d);
            while m % d == 0 {
                m /= d;
            }
        }
        d += 1;
    }
    if m > 1 {
        f.push(m);
    }
    f
}

/// The order of $x$ modulo $x^{n} + x^{k} + 1$: the smallest divisor $d$ of $2^{n} - 1$
/// with $x^{d} = 1$ (reduced one prime factor at a time).
fn order_of_x(n: u32, k: u32) -> u64 {
    let p = (1u64 << n) | (1u64 << k) | 1;
    let mut order = (1u64 << n) - 1;
    assert_eq!(x_pow(order, p, n), 1, "x^(2^n - 1) = 1 for an irreducible polynomial");
    for q in prime_factors(order) {
        while order % q == 0 && x_pow(order / q, p, n) == 1 {
            order /= q;
        }
    }
    order
}

/// The longest runs of zeros and of ones of one period, read cyclically, and whether the
/// sequence repeats after the period (its first `n` bits).
fn runs(seq: O150Sequence) -> (u32, u32, bool) {
    let (period, n) = (seq.period(), seq.stages() as usize);
    let mut it = seq.iter();
    let head: Vec<u8> = it.by_ref().take(n).collect();
    let (mut longest, mut run, mut prev) = ([0u32; 2], 0u32, 2u8);
    let mut account = |b: u8| {
        run = if b == prev { run + 1 } else { 1 };
        prev = b;
        longest[b as usize] = longest[b as usize].max(run);
    };
    for &b in &head {
        account(b);
    }
    // the rest of the period, then the head again: the runs across the wrap-around
    for _ in n as u64..period {
        account(it.next().unwrap());
    }
    let again: Vec<u8> = it.take(n).collect();
    for &b in &again {
        account(b);
    }
    (longest[0], longest[1], again == head)
}

#[test]
fn the_first_bits_of_sec_5_1_are_nine_ones() {
    let b = O150Sequence::Prbs9.bits(10);
    assert_eq!(&b[..9], &[1; 9]);
    assert_eq!(b[9], 0);
}

#[test]
#[ignore = "validation against the standard: run with --release --ignored"]
fn the_sequences_follow_sec_5_and_table_1_of_o150() {
    let mut card = Card::load("o150_sec5").unwrap();
    for seq in O150Sequence::ALL {
        let name = seq.name();
        let order = order_of_x(seq.stages(), seq.feedback());
        let (zeros, ones, repeats) = runs(seq);
        println!("{name}: order of x {order}, longest zeros {zeros}, longest ones {ones}, repeats {repeats}");
        assert!(repeats, "{name} repeats after 2^n - 1 bits");
        card.check(&json!({"sequence": name, "quantity": "order_of_x"}), order as f64);
        card.check(&json!({"sequence": name, "quantity": "longest_zeros"}), f64::from(zeros));
        if seq.inverted() {
            card.check(&json!({"sequence": name, "quantity": "longest_ones"}), f64::from(ones));
        }
    }
    let first = O150Sequence::Prbs9.bits(16).iter().take_while(|&&b| b == 1).count();
    card.check(&json!({"sequence": "prbs9", "quantity": "leading_ones"}), first as f64);
    card.finish().unwrap();
}
