//! The throughput of the decoders of comnum on the configurations recorded from AFF3CT
//! (`validation/aff3ct/`), one thread: the decoder alone on LLRs drawn beforehand, and
//! the whole chain (bits, encoding, BPSK, noise, LLRs, decoding, error count), at one
//! $E_{b}/N_{0}$ given on the command line per configuration, in Mb/s of the bits AFF3CT
//! counts (information bits, or coded bits for the LDPC codes run with `--sim-coded`).
//!
//! ```text
//! cargo run --release -p comnum-validation --bin speed_fec -- <seconds> <name>=<ebn0_db> ...
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "a benchmark: a failure stops it")]

use comnum::core::rng::Rng;
use comnum::core::signal::hard_decisions;
use comnum::fec::bch::Bch;
use comnum::fec::block::BlockCode;
use comnum::fec::bounds::bpsk_noise_variance;
use comnum::fec::conv::ConvolutionalCode;
use comnum::fec::crc::Crc;
use comnum::fec::interleaving::Interleaver;
use comnum::fec::ldpc::{CheckRule, LdpcCode};
use comnum::fec::polar::PolarCode;
use comnum::fec::rs::ReedSolomon;
use comnum::fec::turbo::TurboCode;
use comnum::math::Jacobian;
use comnum::metrics::errors::bit_errors;
use comnum::systems::dvbs2::{CodeRate, Dvbs2Fec, FrameSize};
use comnum_validation::bpsk_awgn_llr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// An encoder of a block of bits.
type Encoder = Box<dyn Fn(&[u8]) -> Vec<u8>>;
/// A decoder of a block of LLRs into the bits compared.
type Decoder = Box<dyn Fn(&[f64]) -> Vec<u8>>;

/// A configuration: the code behind one closure per stage, and the bits AFF3CT counts.
struct Config {
    k: usize,
    n: usize,
    counted_bits: usize,
    encode: Encoder,
    /// Decode the LLRs of a block into the bits compared (information or coded bits).
    decode: Decoder,
    /// The bits the decision is compared with, from the information bits and the codeword.
    reference: fn(&[u8], &[u8]) -> Vec<u8>,
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("aff3ct")
}

fn info_bits(info: &[u8], _: &[u8]) -> Vec<u8> {
    info.to_vec()
}

fn coded_bits(_: &[u8], c: &[u8]) -> Vec<u8> {
    c.to_vec()
}

fn soft<C: BlockCode + 'static>(code: C) -> Config {
    let (k, n) = (code.k(), code.n());
    let c2 = std::rc::Rc::new(code);
    let c3 = c2.clone();
    Config {
        k,
        n,
        counted_bits: k,
        encode: Box::new(move |i| c2.encode_block(i).unwrap()),
        decode: Box::new(move |l| c3.decode_block(l).unwrap()),
        reference: info_bits,
    }
}

fn hard_bch(code: Bch) -> Config {
    let (k, n) = (code.k(), code.n());
    let c2 = std::rc::Rc::new(code);
    let c3 = c2.clone();
    Config {
        k,
        n,
        counted_bits: k,
        encode: Box::new(move |i| c2.encode(i).unwrap()),
        decode: Box::new(move |l| c3.decode(&hard_decisions(l)).unwrap().0),
        reference: info_bits,
    }
}

fn ldpc(mut code: LdpcCode, rule: CheckRule, layered: bool, iterations: usize) -> Config {
    code.decoder.rule = rule;
    code.decoder.layered = layered;
    code.decoder.max_iterations = iterations;
    let (k, n) = (code.k(), code.n());
    let c2 = std::rc::Rc::new(code);
    let c3 = c2.clone();
    Config {
        k,
        n,
        counted_bits: n,
        encode: Box::new(move |i| c2.encode_block(i).unwrap()),
        decode: Box::new(move |l| hard_decisions(&c3.decode_llr(l).unwrap().0)),
        reference: coded_bits,
    }
}

fn matrix(file: &str) -> LdpcCode {
    LdpcCode::from_alist_file(dir().join("matrices").join(file)).unwrap()
}

fn config(name: &str) -> Config {
    let rs = |m, poly, t, n| soft(ReedSolomon::with_field(m, poly, 1, 1, t, n).unwrap());
    match name {
        "rs_n15_k11_t2" => rs(4, 0x13, 2, 15),
        "rs_n63_k55_t4" => rs(6, 0x43, 4, 63),
        "rs_n255_k239_t8" => rs(8, 0x171, 8, 255),
        "bch_n63_k45_t3" => hard_bch(Bch::new(6, 0x43, 3, 63).unwrap()),
        "bch_n127_k92_t5" => hard_bch(Bch::new(7, 0x83, 5, 127).unwrap()),
        "bch_n255_k223_t4" => hard_bch(Bch::new(8, 0x171, 4, 255).unwrap()),
        "conv_k7_k64_viterbi" => soft(ConvolutionalCode::k7(64).unwrap()),
        "ldpc_peg_n576_k288_flooding_spa_i50" => ldpc(matrix("peg_n576_k288.alist"), CheckRule::SumProduct, false, 50),
        "ldpc_peg_n576_k288_layered_nms075_i20" => {
            ldpc(matrix("peg_n576_k288.alist"), CheckRule::NormalizedMinSum(0.75), true, 20)
        }
        "ldpc_peg_n576_k288_flooding_oms015_i20" => {
            ldpc(matrix("peg_n576_k288.alist"), CheckRule::OffsetMinSum(0.15), false, 20)
        }
        "ldpc_peg_n648_k540_layered_spa_i20" => ldpc(matrix("peg_n648_k540.alist"), CheckRule::SumProduct, true, 20),
        "ldpc_dvbs2_short_r1_2_layered_spa_i50" => {
            let code = Dvbs2Fec::new(FrameSize::Short, CodeRate::R1_2).unwrap().ldpc().clone();
            ldpc(code, CheckRule::SumProduct, true, 50)
        }
        "polar_n256_k128_sc" => soft(PolarCode::new(256, 128, 2.0).unwrap()),
        "polar_n1024_k512_sc" => soft(PolarCode::new(1024, 512, 2.0).unwrap()),
        "polar_n256_k128_cascl_l8_crc11" => {
            soft(PolarCode::new(256, 128, 2.0).unwrap().with_crc(Crc::crc11()).unwrap().with_list(8).unwrap())
        }
        "turbo_lte_k40_logmap_i8" => soft(TurboCode::lte(40).unwrap()),
        "turbo_lte_k1024_maxlogmap_i6" => {
            let mut code = TurboCode::with_interleaver(Interleaver::qpp(1024, 31, 64).unwrap()).unwrap();
            code.decoder.iterations = 6;
            code.decoder.jacobian = Jacobian::MaxLog;
            soft(code)
        }
        other => panic!("unknown configuration {other}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seconds: f64 = args[0].parse().unwrap();
    let budget = Duration::from_secs_f64(seconds);
    println!("{{");
    for (i, spec) in args[1..].iter().enumerate() {
        let (name, ebn0) = spec.split_once('=').unwrap();
        let ebn0: f64 = ebn0.parse().unwrap();
        let cfg = config(name);
        let s2 = bpsk_noise_variance(ebn0, cfg.k as f64 / cfg.n as f64);
        let mut rng = Rng::new(42);
        // the decoder alone: a pool of frames of LLRs drawn beforehand
        let pool: Vec<(Vec<f64>, Vec<u8>)> = (0..256)
            .map(|_| {
                let info: Vec<u8> = (0..cfg.k).map(|_| rng.bit()).collect();
                let c = (cfg.encode)(&info);
                (bpsk_awgn_llr(&c, s2, &mut rng), (cfg.reference)(&info, &c))
            })
            .collect();
        for (l, _) in pool.iter().take(16) {
            std::hint::black_box((cfg.decode)(l));
        }
        let start = Instant::now();
        let mut frames = 0u64;
        let mut errors = 0u64;
        while start.elapsed() < budget {
            let (l, r) = &pool[frames as usize % pool.len()];
            errors += bit_errors(&(cfg.decode)(l), r).unwrap() as u64;
            frames += 1;
        }
        let dec_mbps = frames as f64 * cfg.counted_bits as f64 / start.elapsed().as_secs_f64() / 1e6;
        // the whole chain
        let start = Instant::now();
        let mut chain_frames = 0u64;
        let mut chain_errors = 0u64;
        while start.elapsed() < budget {
            let info: Vec<u8> = (0..cfg.k).map(|_| rng.bit()).collect();
            let c = (cfg.encode)(&info);
            let decided = (cfg.decode)(&bpsk_awgn_llr(&c, s2, &mut rng));
            chain_errors += bit_errors(&decided, &(cfg.reference)(&info, &c)).unwrap() as u64;
            chain_frames += 1;
        }
        let chain_mbps = chain_frames as f64 * cfg.counted_bits as f64 / start.elapsed().as_secs_f64() / 1e6;
        let ber = chain_errors as f64 / (chain_frames as f64 * cfg.counted_bits as f64);
        let comma = if i + 1 < args.len() - 1 { "," } else { "" };
        println!(
            " \"{name}\": {{\"ebn0_db\": {ebn0}, \"decoder_mbps\": {dec_mbps:.4}, \"chain_mbps\": {chain_mbps:.4}, \"frames\": {frames}, \"ber\": {ber:.3e}, \"counted_bits\": {}}}{comma}",
            cfg.counted_bits
        );
        let _ = errors;
    }
    println!("}}");
}
