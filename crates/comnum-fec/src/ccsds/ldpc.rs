//! The LDPC codes of CCSDS 131.0-B-5 Sec. 7: the (8160, 7136) code of near-Earth missions
//! and the AR4JA codes of deep-space missions, [`CcsdsLdpc`].

use crate::block::BlockCode;
use crate::block::SoftDecode;
use crate::block::container_traits;
use crate::block::expect_len;
use crate::ldpc::LdpcCode;
use comnum_core::Error;
use comnum_core::Result;
use comnum_core::ensure;
use comnum_core::signal::KNOWN_BIT_LLR;
use comnum_core::signal::hard_decisions;
use comnum_math::gf2;

/// Table 7-1: the two positions of the ones in the first row of every circulant $A_{i,j}$
/// of the (8176, 7156) code, $i = 1, 2$, $j = 1, \dots, 16$.
const LDPC_C2_A: [[[u16; 2]; 16]; 2] = [
    [
        [0, 176],
        [12, 239],
        [0, 352],
        [24, 431],
        [0, 392],
        [151, 409],
        [0, 351],
        [9, 359],
        [0, 307],
        [53, 329],
        [0, 207],
        [18, 281],
        [0, 399],
        [202, 457],
        [0, 247],
        [36, 261],
    ],
    [
        [99, 471],
        [130, 473],
        [198, 435],
        [260, 478],
        [215, 420],
        [282, 481],
        [48, 396],
        [193, 445],
        [273, 430],
        [302, 451],
        [96, 379],
        [191, 386],
        [244, 467],
        [364, 470],
        [51, 382],
        [192, 414],
    ],
];

/// Table C-1: the first rows of the circulants $B_{i,j}$ of the generator of the
/// (8176, 7154) subcode, $i = 1, \dots, 14$, $j = 1, 2$: 128 hexadecimal digits each, the
/// first one octal (3 bits), 511 bits, position 0 first.
const LDPC_C2_B: [[&str; 2]; 14] = [
    [
        "55BF56CC55283DFEEFEA8C8CFF04E1EBD9067710988E25048D67525426939E2068D2DC6FCD2F822BEB6BD96C8A76F4932AAE9BC53AD20A2A9C86BB461E43759C",
        "6855AE08698A50AA3051768793DC238544AF3FE987391021AAF6383A6503409C3CE971A80B3ECE12363EE809A01D91204F1811123EAB867D3E40E8C652585D28",
    ],
    [
        "62B21CF0AEE0649FA67B7D0EA6551C1CD194CA77501E0FCF8C85867B9CF679C18BCF7939E10F8550661848A4E0A9E9EDB7DAB9EDABA18C168C8E28AACDDEAB1E",
        "64B71F486AD57125660C4512247B229F0017BA649C6C11148FB00B70808286F1A9790748D296A593FA4FD2C6D7AAF7750F0C71B31AEE5B400C7F5D73AAF00710",
    ],
    [
        "681A8E51420BD8294ECE13E491D618083FFBBA830DB5FAF330209877D801F92B5E07117C57E75F6F0D873B3E520F21EAFD78C1612C6228111A369D5790F5929A",
        "04DF1DD77F1C20C1FB570D7DD7A1219EAECEA4B2877282651B0FFE713DF338A63263BC0E324A87E2DC1AD64C9F10AAA585ED6905946EE167A73CF04AD2AF9218",
    ],
    [
        "35951FEE6F20C902296C9488003345E6C5526C5519230454C556B8A04FC0DC642D682D94B4594B5197037DF15B5817B26F16D0A3302C09383412822F6D2B234E",
        "7681CF7F278380E28F1262B22F40BF3405BFB92311A8A34D084C086464777431DBFDDD2E82A2E6742BAD6533B51B2BDEE0377E9F6E63DCA0B0F1DF97E73D5CD8",
    ],
    [
        "188157AE41830744BAE0ADA6295E08B79A44081E111F69BBE7831D07BEEBF76232E065F752D4F218D39B6C5BF20AE5B8FF172A7F1F680E6BF5AAC3C4343736C2",
        "5D80A6007C175B5C0DD88A442440E2C29C6A136BBCE0D95A58A83B48CA0E7474E9476C92E33D164BFF943A61CE1031DFF441B0B175209B498394F4794644392E",
    ],
    [
        "60CD1F1C282A1612657E8C7C1420332CA245C0756F78744C807966C3E1326438878BD2CCC83388415A612705AB192B3512EEF0D95248F7B73E5B0F412BF76DB4",
        "434B697B98C9F3E48502C8DBD891D0A0386996146DEBEF11D4B833033E05EDC28F808F25E8F314135E6675B7608B66F7FF3392308242930025DDC4BB65CD7B6E",
    ],
    [
        "766855125CFDC804DAF8DBE3660E8686420230ED4E049DF11D82E357C54FE256EA01F5681D95544C7A1E32B7C30A8E6CF5D0869E754FFDE6AEFA6D7BE8F1B148",
        "222975D325A487FE560A6D146311578D9C5501D28BC0A1FB48C9BDA173E869133A3AA9506C42AE9F466E85611FC5F8F74E439638D66D2F00C682987A96D8887C",
    ],
    [
        "14B5F98E8D55FC8E9B4EE453C6963E052147A857AC1E08675D99A308E7269FAC5600D7B155DE8CB1BAC786F45B46B523073692DE745FDF10724DDA38FD093B1C",
        "1B71AFFB8117BCF8B5D002A99FEEA49503C0359B056963FE5271140E626F6F8FCE9F29B37047F9CA89EBCE760405C6277F329065DF21AB3B779AB3E8C8955400",
    ],
    [
        "0008B4E899E5F7E692BDCE69CE3FAD997183CFAEB2785D0C3D9CAE510316D4BD65A2A06CBA7F4E4C4A80839ACA81012343648EEA8DBBA2464A68E115AB3F4034",
        "5B7FE6808A10EA42FEF0ED9B41920F82023085C106FBBC1F56B567A14257021BC5FDA60CBA05B08FAD6DC3B0410295884C7CCDE0E56347D649DE6DDCEEB0C95E",
    ],
    [
        "5E9B2B33EF82D0E64AA2226D6A0ADCD179D5932EE1CF401B336449D0FF775754CA56650716E61A43F963D59865C7F017F53830514306649822CAA72C152F6EB2",
        "2CD8140C8A37DE0D0261259F63AA2A420A8F81FECB661DBA5C62DF6C817B4A61D2BC1F068A50DFD0EA8FE1BD387601062E2276A4987A19A70B460C54F215E184",
    ],
    [
        "06F1FF249192F2EAF063488E267EEE994E7760995C4FA6FFA0E4241825A7F5B65C74FB16AC4C891BC008D33AD4FF97523EE5BD14126916E0502FF2F8E4A07FC2",
        "65287840D00243278F41CE1156D1868F24E02F91D3A1886ACE906CE741662B40B4EFDFB90F76C1ADD884D920AFA8B3427EEB84A759FA02E00635743F50B942F0",
    ],
    [
        "4109DA2A24E41B1F375645229981D4B7E88C36A12DAB64E91C764CC43CCEC188EC8C5855C8FF488BB91003602BEF43DBEC4A621048906A2CDC5DBD4103431DB8",
        "2185E3BC7076BA51AAD6B199C8C60BCD70E8245B874927136E6D8DD527DF0693DC10A1C8E51B5BE93FF7538FA138B335738F4315361ABF8C73BF40593AE22BE4",
    ],
    [
        "228845775A262505B47288E065B23B4A6D78AFBDDB2356B392C692EF56A35AB4AA27767DE72F058C6484457C95A8CCDD0EF225ABA56B7657B7F0E947DC17F972",
        "2630C6F79878E50CF5ABD353A6ED80BEACC7169179EA57435E44411BC7D566136DFA983019F3443DE8E4C60940BC4E31DCEAD514D755AF95A622585D69572692",
    ],
    [
        "7273E8342918E097B1C1F5FEF32A150AEF5E11184782B5BD5A1D8071E94578B0AC722D7BF49E8C78D391294371FFBA7B88FABF8CC03A62B940CE60D669DFB7B6",
        "087EA12042793307045B283D7305E93D8F74725034E77D25D3FF043ADC5F8B5B186DB70A968A816835EFB575952EAE7EA4E76DF0D5F097590E1A2A978025573E",
    ],
];

/// $\theta_{k}$ of Tables 7-3 and 7-4, $k = 1, \dots, 26$.
const AR4JA_THETA: [u8; 26] = [3, 0, 1, 2, 2, 3, 0, 1, 0, 1, 2, 0, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 3];

/// $\phi_{k}(j, M)$ of Tables 7-3 ($j = 0, 1$) and 7-4 ($j = 2, 3$), for $M = 2^{7}, \dots,
/// 2^{13}$.
const AR4JA_PHI: [[[u16; 7]; 4]; 26] = [
    [[1, 59, 16, 160, 108, 226, 1148], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
    [
        [22, 18, 103, 241, 126, 618, 2032],
        [27, 32, 53, 182, 375, 767, 1822],
        [12, 46, 8, 35, 219, 254, 318],
        [13, 44, 35, 162, 312, 285, 1189],
    ],
    [
        [0, 52, 105, 185, 238, 404, 249],
        [30, 21, 74, 249, 436, 227, 203],
        [30, 45, 119, 167, 16, 790, 494],
        [19, 51, 97, 7, 503, 554, 458],
    ],
    [
        [26, 23, 0, 251, 481, 32, 1807],
        [28, 36, 45, 65, 350, 247, 882],
        [18, 27, 89, 214, 263, 642, 1467],
        [14, 12, 112, 31, 388, 809, 460],
    ],
    [
        [0, 11, 50, 209, 96, 912, 485],
        [7, 30, 47, 70, 260, 284, 1989],
        [10, 48, 31, 84, 415, 248, 757],
        [15, 15, 64, 164, 48, 185, 1039],
    ],
    [
        [10, 7, 29, 103, 28, 950, 1044],
        [1, 29, 0, 141, 84, 370, 957],
        [16, 37, 122, 206, 403, 899, 1085],
        [20, 12, 93, 11, 7, 49, 1000],
    ],
    [
        [5, 22, 115, 90, 59, 534, 717],
        [8, 44, 59, 237, 318, 482, 1705],
        [13, 41, 1, 122, 184, 328, 1630],
        [17, 4, 99, 237, 185, 101, 1265],
    ],
    [
        [18, 25, 30, 184, 225, 63, 873],
        [20, 29, 102, 77, 382, 273, 1083],
        [9, 13, 69, 67, 279, 518, 64],
        [4, 7, 94, 125, 328, 82, 1223],
    ],
    [
        [3, 27, 92, 248, 323, 971, 364],
        [26, 39, 25, 55, 169, 886, 1072],
        [7, 9, 92, 147, 198, 477, 689],
        [4, 2, 103, 133, 254, 898, 874],
    ],
    [
        [22, 30, 78, 12, 28, 304, 1926],
        [24, 14, 3, 12, 213, 634, 354],
        [15, 49, 47, 54, 307, 404, 1300],
        [11, 30, 91, 99, 202, 627, 1292],
    ],
    [
        [3, 43, 70, 111, 386, 409, 1241],
        [4, 22, 88, 227, 67, 762, 1942],
        [16, 36, 11, 23, 432, 698, 148],
        [17, 53, 3, 105, 285, 154, 1491],
    ],
    [
        [8, 14, 66, 66, 305, 708, 1769],
        [12, 15, 65, 42, 313, 184, 446],
        [18, 10, 31, 93, 240, 160, 777],
        [20, 23, 6, 17, 11, 65, 631],
    ],
    [
        [25, 46, 39, 173, 34, 719, 532],
        [23, 48, 62, 52, 242, 696, 1456],
        [4, 11, 19, 20, 454, 497, 1431],
        [8, 29, 39, 97, 168, 81, 464],
    ],
    [
        [25, 62, 84, 42, 510, 176, 768],
        [15, 55, 68, 243, 188, 413, 1940],
        [23, 18, 66, 197, 294, 100, 659],
        [22, 37, 113, 91, 127, 823, 461],
    ],
    [
        [2, 44, 79, 157, 147, 743, 1138],
        [15, 39, 91, 179, 1, 854, 1660],
        [5, 54, 49, 46, 479, 518, 352],
        [19, 42, 92, 211, 8, 50, 844],
    ],
    [
        [27, 12, 70, 174, 199, 759, 965],
        [22, 11, 70, 250, 306, 544, 1661],
        [3, 40, 81, 162, 289, 92, 1177],
        [15, 48, 119, 128, 437, 413, 392],
    ],
    [
        [7, 38, 29, 104, 347, 674, 141],
        [31, 1, 115, 247, 397, 864, 587],
        [29, 27, 96, 101, 373, 464, 836],
        [5, 4, 74, 82, 475, 462, 922],
    ],
    [
        [7, 47, 32, 144, 391, 958, 1527],
        [3, 50, 31, 164, 80, 82, 708],
        [11, 35, 38, 76, 104, 592, 1572],
        [21, 10, 73, 115, 85, 175, 256],
    ],
    [
        [15, 1, 45, 43, 165, 984, 505],
        [29, 40, 121, 17, 33, 1009, 1466],
        [4, 25, 83, 78, 141, 198, 348],
        [17, 18, 116, 248, 419, 715, 1986],
    ],
    [
        [10, 52, 113, 181, 414, 11, 1312],
        [21, 62, 45, 31, 7, 437, 433],
        [8, 46, 42, 253, 270, 856, 1040],
        [9, 56, 31, 62, 459, 537, 19],
    ],
    [
        [4, 61, 86, 250, 97, 413, 1840],
        [2, 27, 56, 149, 447, 36, 1345],
        [2, 24, 58, 124, 439, 235, 779],
        [20, 9, 127, 26, 468, 722, 266],
    ],
    [
        [19, 10, 1, 202, 158, 925, 709],
        [5, 38, 54, 105, 336, 562, 867],
        [11, 33, 24, 143, 333, 134, 476],
        [18, 11, 98, 140, 209, 37, 471],
    ],
    [
        [7, 55, 42, 68, 86, 687, 1427],
        [11, 40, 108, 183, 424, 816, 1551],
        [11, 18, 25, 63, 399, 542, 191],
        [31, 23, 23, 121, 311, 488, 1166],
    ],
    [
        [9, 7, 118, 177, 168, 752, 989],
        [26, 15, 14, 153, 134, 452, 2041],
        [3, 37, 92, 41, 14, 545, 1393],
        [13, 8, 38, 12, 211, 179, 1300],
    ],
    [
        [26, 12, 33, 170, 506, 867, 1925],
        [9, 11, 30, 177, 152, 290, 1383],
        [15, 35, 38, 214, 277, 777, 1752],
        [2, 7, 18, 41, 510, 430, 1033],
    ],
    [
        [17, 2, 126, 89, 489, 323, 270],
        [17, 18, 116, 19, 492, 778, 1790],
        [13, 21, 120, 70, 412, 483, 1627],
        [18, 24, 62, 249, 320, 264, 1606],
    ],
];

/// The size of a circulant of the (8176, 7156) code.
const CIRCULANT: usize = 511;

/// The code rates of the AR4JA codes (Sec. 7.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ar4jaRate {
    /// 1/2, $K = 2$.
    R1_2,
    /// 2/3, $K = 4$.
    R2_3,
    /// 4/5, $K = 8$.
    R4_5,
}

impl Ar4jaRate {
    /// The number $K$ of information block columns of $H$.
    pub fn blocks(self) -> usize {
        match self {
            Ar4jaRate::R1_2 => 2,
            Ar4jaRate::R2_3 => 4,
            Ar4jaRate::R4_5 => 8,
        }
    }

    /// The name of the rate, `1/2`, `2/3` or `4/5`.
    pub fn name(self) -> &'static str {
        match self {
            Ar4jaRate::R1_2 => "1/2",
            Ar4jaRate::R2_3 => "2/3",
            Ar4jaRate::R4_5 => "4/5",
        }
    }
}

impl std::str::FromStr for Ar4jaRate {
    type Err = Error;

    /// `1/2`, `2/3` or `4/5`.
    fn from_str(s: &str) -> Result<Ar4jaRate> {
        match s.trim() {
            "1/2" => Ok(Ar4jaRate::R1_2),
            "2/3" => Ok(Ar4jaRate::R2_3),
            "4/5" => Ok(Ar4jaRate::R4_5),
            _ => Err(Error::Config(format!("the AR4JA rate must be 1/2, 2/3 or 4/5, got {s:?}"))),
        }
    }
}

/// The permutation $\pi_{k}(i)$ of $\Pi_{k}$, $k = 1, \dots, 26$, for the submatrix size $M$
/// ($2^{7}$ to $2^{13}$), Sec. 7.4.2.4.
pub(super) fn ar4ja_pi(k: usize, i: usize, m: usize) -> usize {
    let col = m.trailing_zeros() as usize - 7;
    let q = m / 4;
    let j = 4 * i / m;
    q * ((usize::from(AR4JA_THETA[k - 1]) + j) % 4) + (usize::from(AR4JA_PHI[k - 1][j][col]) + i) % q
}

/// A block of $H$ of an AR4JA code: the sum of an identity (if `identity`) and of the
/// permutations $\Pi_{k}$ of `pis`.
#[derive(Clone, Copy)]
struct Block {
    identity: bool,
    pis: &'static [usize],
}

const ZERO: Block = Block { identity: false, pis: &[] };

const IDENTITY: Block = Block { identity: true, pis: &[] };

const fn pis(pis: &'static [usize]) -> Block {
    Block { identity: false, pis }
}

/// The block rows of $H$ of an AR4JA code, $3 \times (K + 3)$ blocks (Secs. 7.4.2.2,
/// 7.4.2.3).
fn ar4ja_blocks(rate: Ar4jaRate) -> Vec<[Block; 3]> {
    let h12 = [
        [ZERO, IDENTITY, IDENTITY],
        [ZERO, IDENTITY, pis(&[5, 6])],
        [IDENTITY, ZERO, ZERO],
        [ZERO, IDENTITY, pis(&[7, 8])],
        [Block { identity: true, pis: &[1] }, pis(&[2, 3, 4]), IDENTITY],
    ];
    let p23 = [[ZERO, pis(&[9, 10, 11]), IDENTITY], [ZERO, IDENTITY, pis(&[12, 13, 14])]];
    let p45 = [
        [ZERO, pis(&[21, 22, 23]), IDENTITY],
        [ZERO, IDENTITY, pis(&[24, 25, 26])],
        [ZERO, pis(&[15, 16, 17]), IDENTITY],
        [ZERO, IDENTITY, pis(&[18, 19, 20])],
    ];
    // block columns, each the three blocks of its block rows
    let mut cols: Vec<[Block; 3]> = Vec::new();
    if rate == Ar4jaRate::R4_5 {
        cols.extend(p45);
    }
    if rate != Ar4jaRate::R1_2 {
        cols.extend(p23);
    }
    cols.extend(h12);
    cols
}

/// The LDPC codes of CCSDS (see the module): the decoder of [`LdpcCode`] on the full
/// parity-check matrix, the encoder of the standard, the virtual zeros and the punctured
/// bits.
#[derive(Clone, Debug, PartialEq)]
pub struct CcsdsLdpc {
    /// The decoder: the code of the full parity-check matrix.
    code: LdpcCode,
    kind: LdpcKind,
}

#[derive(Clone, Debug, PartialEq)]
enum LdpcKind {
    /// The (8160, 7136) code: the 28 generator circulants of Table C-1 (511 bits each).
    C2 { b: Vec<Vec<u8>> },
    /// An AR4JA code: $M$, $K$, the permutations of the blocks and the inverse of
    /// $C = I + (\Pi_{7} + \Pi_{8})(\Pi_{2} + \Pi_{3} + \Pi_{4})$.
    Ar4ja { m: usize, blocks: usize, perms: Vec<Vec<usize>>, c_inv: gf2::Matrix },
}

impl CcsdsLdpc {
    /// The (8160, 7136) code of rate 223/255 (Sec. 7.3).
    ///
    /// # Errors
    /// Never: the tables are those of the standard.
    pub fn c2() -> Result<CcsdsLdpc> {
        let mut rows = Vec::with_capacity(2 * CIRCULANT);
        for a in &LDPC_C2_A {
            for r in 0..CIRCULANT {
                rows.push(
                    a.iter()
                        .enumerate()
                        .flat_map(|(j, pos)| pos.iter().map(move |&p| CIRCULANT * j + (usize::from(p) + r) % CIRCULANT))
                        .collect(),
                );
            }
        }
        let code = LdpcCode::without_encoder(16 * CIRCULANT, rows, (0..14 * CIRCULANT).collect())?;
        let b = LDPC_C2_B.iter().flatten().map(|h| circulant_row(h)).collect::<Result<_>>()?;
        Ok(CcsdsLdpc { code, kind: LdpcKind::C2 { b } })
    }

    /// The AR4JA code of `k` information bits (1024, 4096 or 16384) and rate `rate`
    /// (Sec. 7.4). Building the code of $k = 16384$ and rate 1/2 inverts an $8192 \times
    /// 8192$ binary matrix (seconds in release).
    ///
    /// # Errors
    /// [`Error::Config`] for another $k$.
    pub fn ar4ja(k: usize, rate: Ar4jaRate) -> Result<CcsdsLdpc> {
        ensure(matches!(k, 1024 | 4096 | 16384), || format!("the AR4JA codes have k = 1024, 4096 or 16384, got {k}"))?;
        let kb = rate.blocks();
        let m = k / kb;
        let perms: Vec<Vec<usize>> = (1..=26).map(|p| (0..m).map(|i| ar4ja_pi(p, i, m)).collect()).collect();
        let cols = ar4ja_blocks(rate);
        let n_full = cols.len() * m;
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); 3 * m];
        for (c, col) in cols.iter().enumerate() {
            for (r, block) in col.iter().enumerate() {
                for i in 0..m {
                    let row = &mut rows[r * m + i];
                    if block.identity {
                        toggle(row, c * m + i);
                    }
                    for &p in block.pis {
                        toggle(row, c * m + perms[p - 1][i]);
                    }
                }
            }
        }
        for row in &mut rows {
            row.sort_unstable();
        }
        // C = I + (P7 + P8)(P2 + P3 + P4): its row i lists i and pi_a(pi_b(i))
        let c_rows: Vec<Vec<usize>> = (0..m)
            .map(|i| {
                let mut row = vec![i];
                for &b in &[7, 8] {
                    let j = perms[b - 1][i];
                    row.extend([2, 3, 4].iter().map(|&a| perms[a - 1][j]));
                }
                row
            })
            .collect();
        let c_inv = gf2::Matrix::from_rows(m, &c_rows)
            .inverse()
            .ok_or_else(|| Error::Numerical("the AR4JA matrix C is singular".into()))?;
        let code = LdpcCode::without_encoder(n_full, rows, (0..kb * m).collect())?;
        Ok(CcsdsLdpc { code, kind: LdpcKind::Ar4ja { m, blocks: kb, perms, c_inv } })
    }

    /// The code of the full parity-check matrix (its decoder settings are public fields).
    pub fn ldpc(&self) -> &LdpcCode {
        &self.code
    }

    /// The code of the full parity-check matrix, to change its decoder settings.
    pub fn ldpc_mut(&mut self) -> &mut LdpcCode {
        &mut self.code
    }

    /// The codeword of the full parity-check matrix ($n$ = 8176 or $(K + 3) M$ bits) of
    /// `info`: the 18 virtual zeros of the rate-7/8 code, the information, the parity
    /// bits, the punctured ones of the AR4JA codes included; [`LdpcCode::is_codeword`] of
    /// [`CcsdsLdpc::ldpc`] checks it.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `info` has $k$ bits.
    pub fn full_codeword(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k(), "bits")?;
        Ok(self.full_codeword_unchecked(info))
    }

    fn full_codeword_unchecked(&self, info: &[u8]) -> Vec<u8> {
        match &self.kind {
            LdpcKind::C2 { b } => {
                let mut c = vec![0u8; 18];
                c.extend_from_slice(info);
                let mut parity = vec![0u8; 2 * CIRCULANT];
                for i in 0..14 {
                    for r in 0..CIRCULANT {
                        if c[i * CIRCULANT + r] == 1 {
                            for j in 0..2 {
                                let row = &b[2 * i + j];
                                for (t, p) in parity[j * CIRCULANT..(j + 1) * CIRCULANT].iter_mut().enumerate() {
                                    *p ^= row[(t + CIRCULANT - r) % CIRCULANT];
                                }
                            }
                        }
                    }
                }
                c.extend(parity);
                c
            }
            LdpcKind::Ar4ja { m, blocks, perms, c_inv } => {
                let (m, kb) = (*m, *blocks);
                // s = Q u: the syndrome of the information block columns
                let mut s = vec![0u8; 3 * m];
                for (r, row) in self.code.rows().iter().enumerate() {
                    s[r] = row.iter().filter(|&&v| v < kb * m).fold(0, |a, &v| a ^ info[v]);
                }
                let pi = |p: usize, v: &[u8]| -> Vec<u8> { (0..m).map(|i| v[perms[p - 1][i]]).collect() };
                let xor = |a: &mut [u8], b: &[u8]| a.iter_mut().zip(b).for_each(|(x, y)| *x ^= y);
                let (s0, s1, s2) = (&s[..m], &s[m..2 * m], &s[2 * m..]);
                // x4 = C^-1 (s2 + (P7 + P8) s1)
                let mut y = s2.to_vec();
                xor(&mut y, &pi(7, s1));
                xor(&mut y, &pi(8, s1));
                let x4 = c_inv.mul_vec(&y);
                // x3 = s1 + (P2 + P3 + P4) x4, x2 = s0 + (I + P1) x4
                let mut x3 = s1.to_vec();
                for p in [2, 3, 4] {
                    xor(&mut x3, &pi(p, &x4));
                }
                let mut x2 = s0.to_vec();
                xor(&mut x2, &x4);
                xor(&mut x2, &pi(1, &x4));
                let mut c = info.to_vec();
                c.extend(x2);
                c.extend(x3);
                c.extend(x4);
                c
            }
        }
    }

    /// The channel LLRs of the full codeword from the received ones: the virtual zeros
    /// known, the punctured bits at zero.
    fn full_llr(&self, llr: &[f64]) -> Vec<f64> {
        match &self.kind {
            LdpcKind::C2 { .. } => {
                let mut full = vec![KNOWN_BIT_LLR; 18];
                full.extend_from_slice(&llr[..16 * CIRCULANT - 18]);
                full
            }
            LdpcKind::Ar4ja { m, .. } => {
                let mut full = llr.to_vec();
                full.resize(llr.len() + m, 0.0);
                full
            }
        }
    }

    /// The information positions in the full codeword.
    fn info_range(&self) -> std::ops::Range<usize> {
        match &self.kind {
            LdpcKind::C2 { .. } => 18..18 + 7136,
            LdpcKind::Ar4ja { m, blocks, .. } => 0..m * blocks,
        }
    }
}

/// Toggle `v` in a row of $H$ (two equal terms of a block cancel).
fn toggle(row: &mut Vec<usize>, v: usize) {
    match row.iter().position(|&x| x == v) {
        Some(p) => {
            row.swap_remove(p);
        }
        None => row.push(v),
    }
}

/// The 511 bits of a first row of Table C-1: the 128 hexadecimal digits, the first one
/// read as 3 bits.
fn circulant_row(hex: &str) -> Result<Vec<u8>> {
    let mut bits = Vec::with_capacity(512);
    for (i, ch) in hex.chars().enumerate() {
        let d = ch.to_digit(16).ok_or_else(|| Error::Config(format!("not a hexadecimal digit: {ch:?}")))?;
        let width = if i == 0 { 3 } else { 4 };
        bits.extend((0..width).rev().map(|k| ((d >> k) & 1) as u8));
    }
    ensure(bits.len() == CIRCULANT, || format!("a circulant row of {} bits", bits.len()))?;
    Ok(bits)
}

impl BlockCode for CcsdsLdpc {
    fn k(&self) -> usize {
        self.info_range().len()
    }

    /// The transmitted bits: 8160, or $(K + 2) M$ (Table 7-5).
    fn n(&self) -> usize {
        match &self.kind {
            LdpcKind::C2 { .. } => 8160,
            LdpcKind::Ar4ja { m, blocks, .. } => (blocks + 2) * m,
        }
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k(), "bits")?;
        let mut c = self.full_codeword_unchecked(info);
        match &self.kind {
            LdpcKind::C2 { .. } => {
                c.drain(..18);
                c.extend([0, 0]);
            }
            LdpcKind::Ar4ja { m, .. } => c.truncate(c.len() - m),
        }
        Ok(c)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        Ok(hard_decisions(&self.app(llr, None)?))
    }
}

impl SoftDecode for CcsdsLdpc {
    /// The a posteriori LLRs of the information bits by belief propagation on the full
    /// parity-check matrix; the a priori LLRs add to the channel LLRs of the information
    /// bits.
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        expect_len(out.len(), self.k(), "a posteriori LLRs")?;
        let mut full = self.full_llr(llr);
        let range = self.info_range();
        if let Some(a) = apriori {
            expect_len(a.len(), self.k(), "a priori LLRs")?;
            for (l, a) in full[range.clone()].iter_mut().zip(a) {
                *l += a;
            }
        }
        let (app, iterations) = self.code.decode_llr(&full)?;
        out.copy_from_slice(&app[range]);
        Ok(iterations)
    }
}

container_traits!(CcsdsLdpc);
