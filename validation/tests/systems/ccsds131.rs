//! The codes and synchronisation of CCSDS 131.0-B-5, *TM Synchronization and Channel
//! Coding*, Recommended Standard (Blue Book), Issue 5, September 2023, against the values
//! it prints (`comnum_fec::ccsds`, `comnum_systems::ccsds`).
//!
//! The standard specifies formats and gives no error-rate figure, so every comparison is
//! exact:
//!
//! - Table F-1 (Annex F, the 256 elements of GF(2^8) of F(x) = x^8 + x^7 + x^2 + x + 1 in
//!   the conventional and the dual basis): every row against alpha^i of the field and its
//!   image by the matrix T of Sec. 4.3.9; the examples 1 and 2 of Annex F;
//! - Annex G: the 17 (E = 16) and 9 (E = 8) distinct coefficients of g(x);
//! - Sec. 4.3.8: the check bits (1280) and the transmitted codeblock (10 200 bits) of
//!   E = 16, I = 5;
//! - Table 3-1: the output sequences of the punctured convolutional codes;
//! - Tables 6-2 and 7-5: the codeword lengths of the turbo and AR4JA codes; Sec. 7.3.5:
//!   the 8160 bits of the rate-7/8 LDPC code;
//! - the parity checks of Table 7-1 and Sec. 7.4.2 on codewords of the encoders (Table
//!   C-1, the generator of Sec. 7.4.3), for the nine AR4JA codes;
//! - Sec. 10.4.3 NOTE 2: the first 40 bits of both pseudo-random sequences; Figures 9-1
//!   to 9-6: the markers.
//!
//! The values are in the card `validation/cards/ccsds131_exact.yaml`. The AR4JA codes of
//! k = 16384 invert matrices of up to 8192 x 8192 bits: the test is a campaign.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::core::rng::Rng;
use comnum::fec::block::BlockCode;
use comnum::fec::ccsds::{Ar4jaRate, CcsdsLdpc, CcsdsRs, CcsdsTurbo, TurboRate, from_dual_basis, to_dual_basis};
use comnum::math::gf2m::Field;
use comnum::systems::ccsds::{ASM, ASM_64, ASM_EMBEDDED, ASM_TURBO_1_3, ConvolutionalRate, Randomizer};
use comnum_validation::card::Card;
use serde_json::json;

/// Table F-1: the power of alpha (`None` for zero), the conventional bits u7 ... u0 and
/// the dual-basis bits z0 ... z7, verbatim.
const TABLE_F1: [(Option<usize>, u8, u8); 256] = [
    (None, 0b0000_0000, 0b0000_0000),
    (Some(0), 0b0000_0001, 0b0111_1011),
    (Some(1), 0b0000_0010, 0b1010_1111),
    (Some(2), 0b0000_0100, 0b1001_1001),
    (Some(3), 0b0000_1000, 0b1111_1010),
    (Some(4), 0b0001_0000, 0b1000_0110),
    (Some(5), 0b0010_0000, 0b1110_1100),
    (Some(6), 0b0100_0000, 0b1110_1111),
    (Some(7), 0b1000_0000, 0b1000_1101),
    (Some(8), 0b1000_0111, 0b1100_0000),
    (Some(9), 0b1000_1001, 0b0000_1100),
    (Some(10), 0b1001_0101, 0b1110_1001),
    (Some(11), 0b1010_1101, 0b0111_1001),
    (Some(12), 0b1101_1101, 0b1111_1100),
    (Some(13), 0b0011_1101, 0b0111_0010),
    (Some(14), 0b0111_1010, 0b1101_0000),
    (Some(15), 0b1111_0100, 0b1001_0001),
    (Some(16), 0b0110_1111, 0b1011_0100),
    (Some(17), 0b1101_1110, 0b0010_1000),
    (Some(18), 0b0011_1011, 0b0100_0100),
    (Some(19), 0b0111_0110, 0b1011_0011),
    (Some(20), 0b1110_1100, 0b1110_1101),
    (Some(21), 0b0101_1111, 0b1101_1110),
    (Some(22), 0b1011_1110, 0b0010_1011),
    (Some(23), 0b1111_1011, 0b0010_0110),
    (Some(24), 0b0111_0001, 0b1111_1110),
    (Some(25), 0b1110_0010, 0b0010_0001),
    (Some(26), 0b0100_0011, 0b0011_1011),
    (Some(27), 0b1000_0110, 0b1011_1011),
    (Some(28), 0b1000_1011, 0b1010_0011),
    (Some(29), 0b1001_0001, 0b0111_0000),
    (Some(30), 0b1010_0101, 0b1000_0011),
    (Some(31), 0b1100_1101, 0b0111_1010),
    (Some(32), 0b0001_1101, 0b1001_1110),
    (Some(33), 0b0011_1010, 0b0011_1111),
    (Some(34), 0b0111_0100, 0b0001_1100),
    (Some(35), 0b1110_1000, 0b0111_0100),
    (Some(36), 0b0101_0111, 0b0010_0100),
    (Some(37), 0b1010_1110, 0b1010_1101),
    (Some(38), 0b1101_1011, 0b1100_1010),
    (Some(39), 0b0011_0001, 0b0001_0001),
    (Some(40), 0b0110_0010, 0b1010_1100),
    (Some(41), 0b1100_0100, 0b1111_1011),
    (Some(42), 0b0000_1111, 0b1011_0111),
    (Some(43), 0b0001_1110, 0b0100_1010),
    (Some(44), 0b0011_1100, 0b0000_1001),
    (Some(45), 0b0111_1000, 0b0111_1111),
    (Some(46), 0b1111_0000, 0b0000_1000),
    (Some(47), 0b0110_0111, 0b0100_1110),
    (Some(48), 0b1100_1110, 0b1010_1110),
    (Some(49), 0b0001_1011, 0b1010_1000),
    (Some(50), 0b0011_0110, 0b0101_1100),
    (Some(51), 0b0110_1100, 0b0110_0000),
    (Some(52), 0b1101_1000, 0b0001_1110),
    (Some(53), 0b0011_0111, 0b0010_0111),
    (Some(54), 0b0110_1110, 0b1100_1111),
    (Some(55), 0b1101_1100, 0b1000_0111),
    (Some(56), 0b0011_1111, 0b1101_1101),
    (Some(57), 0b0111_1110, 0b0100_1001),
    (Some(58), 0b1111_1100, 0b0110_1011),
    (Some(59), 0b0111_1111, 0b0011_0010),
    (Some(60), 0b1111_1110, 0b1100_0100),
    (Some(61), 0b0111_1011, 0b1010_1011),
    (Some(62), 0b1111_0110, 0b0011_1110),
    (Some(63), 0b0110_1011, 0b0010_1101),
    (Some(64), 0b1101_0110, 0b1101_0010),
    (Some(65), 0b0010_1011, 0b1100_0010),
    (Some(66), 0b0101_0110, 0b0101_1111),
    (Some(67), 0b1010_1100, 0b0000_0010),
    (Some(68), 0b1101_1111, 0b0101_0011),
    (Some(69), 0b0011_1001, 0b1110_1011),
    (Some(70), 0b0111_0010, 0b0010_1010),
    (Some(71), 0b1110_0100, 0b0001_0111),
    (Some(72), 0b0100_1111, 0b0101_1000),
    (Some(73), 0b1001_1110, 0b1100_0111),
    (Some(74), 0b1011_1011, 0b1100_1001),
    (Some(75), 0b1111_0001, 0b0111_0011),
    (Some(76), 0b0110_0101, 0b1110_0001),
    (Some(77), 0b1100_1010, 0b0011_0111),
    (Some(78), 0b0001_0011, 0b0101_0010),
    (Some(79), 0b0010_0110, 0b1101_1010),
    (Some(80), 0b0100_1100, 0b1000_1100),
    (Some(81), 0b1001_1000, 0b1111_0001),
    (Some(82), 0b1011_0111, 0b1010_1010),
    (Some(83), 0b1110_1001, 0b0000_1111),
    (Some(84), 0b0101_0101, 0b1000_1011),
    (Some(85), 0b1010_1010, 0b0011_0100),
    (Some(86), 0b1101_0011, 0b0011_0000),
    (Some(87), 0b0010_0001, 0b1001_0111),
    (Some(88), 0b0100_0010, 0b0100_0000),
    (Some(89), 0b1000_0100, 0b0001_0100),
    (Some(90), 0b1000_1111, 0b0011_1010),
    (Some(91), 0b1001_1001, 0b1000_1010),
    (Some(92), 0b1011_0101, 0b0000_0101),
    (Some(93), 0b1110_1101, 0b1001_0110),
    (Some(94), 0b0101_1101, 0b0111_0001),
    (Some(95), 0b1011_1010, 0b1011_0010),
    (Some(96), 0b1111_0011, 0b1101_1100),
    (Some(97), 0b0110_0001, 0b0111_1000),
    (Some(98), 0b1100_0010, 0b1100_1101),
    (Some(99), 0b0000_0011, 0b1101_0100),
    (Some(100), 0b0000_0110, 0b0011_0110),
    (Some(101), 0b0000_1100, 0b0110_0011),
    (Some(102), 0b0001_1000, 0b0111_1100),
    (Some(103), 0b0011_0000, 0b0110_1010),
    (Some(104), 0b0110_0000, 0b0000_0011),
    (Some(105), 0b1100_0000, 0b0110_0010),
    (Some(106), 0b0000_0111, 0b0100_1101),
    (Some(107), 0b0000_1110, 0b1100_1100),
    (Some(108), 0b0001_1100, 0b1110_0101),
    (Some(109), 0b0011_1000, 0b1001_0000),
    (Some(110), 0b0111_0000, 0b1000_0101),
    (Some(111), 0b1110_0000, 0b1000_1110),
    (Some(112), 0b0100_0111, 0b1010_0010),
    (Some(113), 0b1000_1110, 0b0100_0001),
    (Some(114), 0b1001_1011, 0b0010_0101),
    (Some(115), 0b1011_0001, 0b1001_1100),
    (Some(116), 0b1110_0101, 0b0110_1100),
    (Some(117), 0b0100_1101, 0b1111_0111),
    (Some(118), 0b1001_1010, 0b0101_1110),
    (Some(119), 0b1011_0011, 0b0011_0011),
    (Some(120), 0b1110_0001, 0b1111_0101),
    (Some(121), 0b0100_0101, 0b0000_1101),
    (Some(122), 0b1000_1010, 0b1101_1000),
    (Some(123), 0b1001_0011, 0b1101_1111),
    (Some(124), 0b1010_0001, 0b0001_1010),
    (Some(125), 0b1100_0101, 0b1000_0000),
    (Some(126), 0b0000_1101, 0b0001_1000),
    (Some(127), 0b0001_1010, 0b1101_0011),
    (Some(128), 0b0011_0100, 0b1111_0011),
    (Some(129), 0b0110_1000, 0b1111_1001),
    (Some(130), 0b1101_0000, 0b1110_0100),
    (Some(131), 0b0010_0111, 0b1010_0001),
    (Some(132), 0b0100_1110, 0b0010_0011),
    (Some(133), 0b1001_1100, 0b0110_1000),
    (Some(134), 0b1011_1111, 0b0101_0000),
    (Some(135), 0b1111_1001, 0b1000_1001),
    (Some(136), 0b0111_0101, 0b0110_0111),
    (Some(137), 0b1110_1010, 0b1101_1011),
    (Some(138), 0b0101_0011, 0b1011_1101),
    (Some(139), 0b1010_0110, 0b0101_0111),
    (Some(140), 0b1100_1011, 0b0100_1100),
    (Some(141), 0b0001_0001, 0b1111_1101),
    (Some(142), 0b0010_0010, 0b0100_0011),
    (Some(143), 0b0100_0100, 0b0111_0110),
    (Some(144), 0b1000_1000, 0b0111_0111),
    (Some(145), 0b1001_0111, 0b0100_0110),
    (Some(146), 0b1010_1001, 0b1110_0000),
    (Some(147), 0b1101_0101, 0b0000_0110),
    (Some(148), 0b0010_1101, 0b1111_0100),
    (Some(149), 0b0101_1010, 0b0011_1100),
    (Some(150), 0b1011_0100, 0b0111_1110),
    (Some(151), 0b1110_1111, 0b0011_1001),
    (Some(152), 0b0101_1001, 0b1110_1000),
    (Some(153), 0b1011_0010, 0b0100_1000),
    (Some(154), 0b1110_0011, 0b0101_1010),
    (Some(155), 0b0100_0001, 0b1001_0100),
    (Some(156), 0b1000_0010, 0b0010_0010),
    (Some(157), 0b1000_0011, 0b0101_1001),
    (Some(158), 0b1000_0001, 0b1111_0110),
    (Some(159), 0b1000_0101, 0b0110_1111),
    (Some(160), 0b1000_1101, 0b1001_0101),
    (Some(161), 0b1001_1101, 0b0001_0011),
    (Some(162), 0b1011_1101, 0b1111_1111),
    (Some(163), 0b1111_1101, 0b0001_0000),
    (Some(164), 0b0111_1101, 0b1001_1101),
    (Some(165), 0b1111_1010, 0b0101_1101),
    (Some(166), 0b0111_0011, 0b0101_0001),
    (Some(167), 0b1110_0110, 0b1011_1000),
    (Some(168), 0b0100_1011, 0b1100_0001),
    (Some(169), 0b1001_0110, 0b0011_1101),
    (Some(170), 0b1010_1011, 0b0100_1111),
    (Some(171), 0b1101_0001, 0b1001_1111),
    (Some(172), 0b0010_0101, 0b0000_1110),
    (Some(173), 0b0100_1010, 0b1011_1010),
    (Some(174), 0b1001_0100, 0b1001_0010),
    (Some(175), 0b1010_1111, 0b1101_0110),
    (Some(176), 0b1101_1001, 0b0110_0101),
    (Some(177), 0b0011_0101, 0b1000_1000),
    (Some(178), 0b0110_1010, 0b0101_0110),
    (Some(179), 0b1101_0100, 0b0111_1101),
    (Some(180), 0b0010_1111, 0b0101_1011),
    (Some(181), 0b0101_1110, 0b1010_0101),
    (Some(182), 0b1011_1100, 0b1000_0100),
    (Some(183), 0b1111_1111, 0b1011_1111),
    (Some(184), 0b0111_1001, 0b0000_0100),
    (Some(185), 0b1111_0010, 0b1010_0111),
    (Some(186), 0b0110_0011, 0b1101_0111),
    (Some(187), 0b1100_0110, 0b0101_0100),
    (Some(188), 0b0000_1011, 0b0010_1110),
    (Some(189), 0b0001_0110, 0b1011_0000),
    (Some(190), 0b0010_1100, 0b1000_1111),
    (Some(191), 0b0101_1000, 0b1001_0011),
    (Some(192), 0b1011_0000, 0b1110_0111),
    (Some(193), 0b1110_0111, 0b1100_0011),
    (Some(194), 0b0100_1001, 0b0110_1110),
    (Some(195), 0b1001_0010, 0b1010_0100),
    (Some(196), 0b1010_0011, 0b1011_0101),
    (Some(197), 0b1100_0001, 0b0001_1001),
    (Some(198), 0b0000_0101, 0b1110_0010),
    (Some(199), 0b0000_1010, 0b0101_0101),
    (Some(200), 0b0001_0100, 0b0001_1111),
    (Some(201), 0b0010_1000, 0b0001_0110),
    (Some(202), 0b0101_0000, 0b0110_1001),
    (Some(203), 0b1010_0000, 0b0110_0001),
    (Some(204), 0b1100_0111, 0b0010_1111),
    (Some(205), 0b0000_1001, 0b1000_0001),
    (Some(206), 0b0001_0010, 0b0010_1001),
    (Some(207), 0b0010_0100, 0b0111_0101),
    (Some(208), 0b0100_1000, 0b0001_0101),
    (Some(209), 0b1001_0000, 0b0000_1011),
    (Some(210), 0b1010_0111, 0b0010_1100),
    (Some(211), 0b1100_1001, 0b1110_0011),
    (Some(212), 0b0001_0101, 0b0110_0100),
    (Some(213), 0b0010_1010, 0b1011_1001),
    (Some(214), 0b0101_0100, 0b1111_0000),
    (Some(215), 0b1010_1000, 0b1001_1011),
    (Some(216), 0b1101_0111, 0b1010_1001),
    (Some(217), 0b0010_1001, 0b0110_1101),
    (Some(218), 0b0101_0010, 0b1100_0110),
    (Some(219), 0b1010_0100, 0b1111_1000),
    (Some(220), 0b1100_1111, 0b1101_0101),
    (Some(221), 0b0001_1001, 0b0000_0111),
    (Some(222), 0b0011_0010, 0b1100_0101),
    (Some(223), 0b0110_0100, 0b1001_1010),
    (Some(224), 0b1100_1000, 0b1001_1000),
    (Some(225), 0b0001_0111, 0b1100_1011),
    (Some(226), 0b0010_1110, 0b0010_0000),
    (Some(227), 0b0101_1100, 0b0000_1010),
    (Some(228), 0b1011_1000, 0b0001_1101),
    (Some(229), 0b1111_0111, 0b0100_0101),
    (Some(230), 0b0110_1001, 0b1000_0010),
    (Some(231), 0b1101_0010, 0b0100_1011),
    (Some(232), 0b0010_0011, 0b0011_1000),
    (Some(233), 0b0100_0110, 0b1101_1001),
    (Some(234), 0b1000_1100, 0b1110_1110),
    (Some(235), 0b1001_1111, 0b1011_1100),
    (Some(236), 0b1011_1001, 0b0110_0110),
    (Some(237), 0b1111_0101, 0b1110_1010),
    (Some(238), 0b0110_1101, 0b0001_1011),
    (Some(239), 0b1101_1010, 0b1011_0001),
    (Some(240), 0b0011_0011, 0b1011_1110),
    (Some(241), 0b0110_0110, 0b0011_0101),
    (Some(242), 0b1100_1100, 0b0000_0001),
    (Some(243), 0b0001_1111, 0b0011_0001),
    (Some(244), 0b0011_1110, 0b1010_0110),
    (Some(245), 0b0111_1100, 0b1110_0110),
    (Some(246), 0b1111_1000, 0b1111_0010),
    (Some(247), 0b0111_0111, 0b1100_1000),
    (Some(248), 0b1110_1110, 0b0100_0010),
    (Some(249), 0b0101_1011, 0b0100_0111),
    (Some(250), 0b1011_0110, 0b1101_0001),
    (Some(251), 0b1110_1011, 0b1010_0000),
    (Some(252), 0b0101_0001, 0b0001_0010),
    (Some(253), 0b1010_0010, 0b1100_1110),
    (Some(254), 0b1100_0011, 0b1011_0110),
];

fn bits(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| u8::from(c == '1')).collect()
}

fn hex_bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |k| (b >> k) & 1)).collect()
}

#[test]
#[ignore = "validation against the standard: run with --release --ignored"]
fn the_codes_follow_the_tables_of_ccsds_131_0_b_5() {
    let mut card = Card::load("ccsds131_exact").unwrap();
    let field = Field::new(8, 0x187).unwrap();

    // Table F-1 and Annex F
    let rows = TABLE_F1
        .iter()
        .filter(|&&(p, u, z)| {
            let alpha = p.map_or(0, |e| field.pow(e) as u8);
            alpha == u && to_dual_basis(u) == z && from_dual_basis(z) == u
        })
        .count();
    card.check(&json!({"item": "table_f1_rows"}), rows as f64);
    let examples = usize::from(from_dual_basis(0b1011_1001) == 0b0010_1010)
        + usize::from(to_dual_basis(0b0101_1001) == 0b1110_1000);
    card.check(&json!({"item": "annex_f_examples"}), examples as f64);

    // Annex G: the exponents of G_0 ... G_E
    for (e, exps) in [
        (16, &[0, 249, 59, 66, 4, 43, 126, 251, 97, 30, 3, 213, 50, 66, 170, 5, 24][..]),
        (8, &[0, 30, 230, 49, 235, 129, 81, 76, 173][..]),
    ] {
        let g = CcsdsRs::new(e, 1, 0).unwrap().code().generator().to_vec();
        let ok =
            exps.iter().enumerate().filter(|&(i, &x)| u32::from(g[i]) == field.pow(x) && g[i] == g[2 * e - i]).count();
        card.check(&json!({"item": "annex_g_coefficients", "e": e}), ok as f64);
    }

    // Sec. 4.3.8: E = 16, I = 5
    let rs = CcsdsRs::new(16, 5, 0).unwrap();
    card.check(&json!({"item": "rs_check_bits_e16_i5"}), (rs.n() - rs.k()) as f64);
    card.check(&json!({"item": "rs_codeblock_bits_e16_i5"}), rs.n() as f64);

    // Table 3-1: the output sequences, labels C1(t), C2(t)
    let order = |rate: ConvolutionalRate| -> String {
        let p = rate.puncturer().unwrap();
        let steps = p.period();
        let names: Vec<String> = (1..=steps).flat_map(|t| [format!("C1({t})"), format!("C2({t})")]).collect();
        p.selection(2 * steps).unwrap().positions().iter().map(|&i| names[i].clone()).collect::<Vec<_>>().join(" ")
    };
    for (rate, printed) in [
        (ConvolutionalRate::R2_3, "C1(1) C2(1) C2(2)"),
        (ConvolutionalRate::R3_4, "C1(1) C2(1) C2(2) C1(3)"),
        (ConvolutionalRate::R5_6, "C1(1) C2(1) C2(2) C1(3) C2(4) C1(5)"),
        (ConvolutionalRate::R7_8, "C1(1) C2(1) C2(2) C2(3) C2(4) C1(5) C2(6) C1(7)"),
    ] {
        card.check(
            &json!({"item": "table_3_1_sequence", "rate": rate.name()}),
            f64::from(u8::from(order(rate) == printed)),
        );
    }

    // Table 6-2
    for k in [1784, 3568, 7136, 8920] {
        for rate in [TurboRate::R1_2, TurboRate::R1_3, TurboRate::R1_4, TurboRate::R1_6] {
            let code = CcsdsTurbo::new(k, rate).unwrap();
            card.check(&json!({"item": "table_6_2_n", "k": k, "rate": rate.name()}), code.n() as f64);
        }
    }

    // Sec. 7.3: the rate-7/8 code, its codewords against the checks of Table 7-1
    let mut rng = Rng::new(131);
    let c2 = CcsdsLdpc::c2().unwrap();
    card.check(&json!({"item": "ldpc_c2_n"}), c2.n() as f64);
    card.check(&json!({"item": "ldpc_codewords", "code": "7/8"}), codeword_fraction(&c2, &mut rng));

    // Table 7-5 and the AR4JA codewords
    for k in [1024, 4096, 16384] {
        for rate in [Ar4jaRate::R1_2, Ar4jaRate::R2_3, Ar4jaRate::R4_5] {
            let code = CcsdsLdpc::ar4ja(k, rate).unwrap();
            card.check(&json!({"item": "table_7_5_n", "k": k, "rate": rate.name()}), code.n() as f64);
            card.check(
                &json!({"item": "ldpc_codewords", "code": format!("ar4ja {k} {}", rate.name())}),
                codeword_fraction(&code, &mut rng),
            );
        }
    }

    // Sec. 10.4.3 NOTE 2 and Figures 9-1 to 9-6
    let long = bits("0001 1100 0111 0001 1011 1001 0001 1011 1010 1001");
    let short = bits("1111 1111 0100 1000 0000 1110 1100 0000 1001 1010");
    let same = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x == y).count() as f64;
    card.check(
        &json!({"item": "randomizer_first_bits", "sequence": "long"}),
        same(&Randomizer::Long.sequence(40), &long),
    );
    card.check(
        &json!({"item": "randomizer_first_bits", "sequence": "short"}),
        same(&Randomizer::Short.sequence(40), &short),
    );
    for (name, marker, figure) in [
        ("1ACFFC1D", &ASM[..], "0001 1010 1100 1111 1111 1100 0001 1101"),
        ("352EF853", &ASM_EMBEDDED[..], "0011 0101 0010 1110 1111 1000 0101 0011"),
        (
            "034776C7272895B0",
            &ASM_64[..],
            "0000 0011 0100 0111 0111 0110 1100 0111 0010 0111 0010 1000 1001 0101 1011 0000",
        ),
        (
            "25D5C0CE8990F6C9461BF79C",
            &ASM_TURBO_1_3[..],
            "0010 0101 1101 0101 1100 0000 1100 1110 1000 1001 1001 0000 1111 0110 1100 1001 0100 0110 0001 1011 1111 0111 1001 1100",
        ),
    ] {
        card.check(
            &json!({"item": "marker_bits", "marker": name}),
            f64::from(u8::from(hex_bits(marker) == bits(figure))),
        );
    }
    card.finish().unwrap();
}

/// The fraction of 10 random codewords of the encoder that satisfy every check of the
/// full parity-check matrix.
fn codeword_fraction(code: &CcsdsLdpc, rng: &mut Rng) -> f64 {
    let ok = (0..10)
        .filter(|_| {
            let info: Vec<u8> = (0..code.k()).map(|_| rng.bit()).collect();
            code.ldpc().is_codeword(&code.full_codeword(&info).unwrap())
        })
        .count();
    ok as f64 / 10.0
}
