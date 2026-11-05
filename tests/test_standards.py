"""The functions of the standards from Python: ITU-T O.150, DVB-S, DVB-T, CCSDS 131.0-B, LTE."""

import numpy as np
import pytest

import comnum


@pytest.mark.parametrize(
    ("name", "stages", "longest_zeros"),
    [("prbs9", 9, 8), ("prbs11", 11, 10), ("prbs15", 15, 15), ("prbs20z", 20, 14)],
)
def test_o150_sequences_have_the_period_and_runs_of_table_1(name, stages, longest_zeros):
    period = 2**stages - 1
    b = comnum.o150_sequence(name, 2 * period)
    assert b.dtype == np.uint8
    np.testing.assert_array_equal(b[:period], b[period:])
    # the longest run of zeros of a period, read cyclically
    runs = "".join(map(str, np.concatenate([b[:period], b[:stages]]))).split("1")
    assert max(len(r) for r in runs) == longest_zeros


def test_o150_prbs9_begins_with_the_first_of_nine_ones():
    assert comnum.o150_sequence("prbs9", 10).tolist() == [1] * 9 + [0]


def test_o150_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="prbs7"):
        comnum.o150_sequence("prbs7", 10)


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4", "5/6", "7/8"])
def test_dvbs_round_trips_a_burst(rate):
    link = comnum.Dvbs(rate)
    packets = comnum.dvb_packets(9, comnum.Rng(3))
    x = link.modulate(packets)
    assert len(x) == link.symbols_len(9)
    np.testing.assert_allclose(np.abs(x), 1.0)
    out, corrected, failed = link.decode(x, 0.05)
    np.testing.assert_array_equal(out, packets)
    assert (corrected, failed) == (0, 0)


def test_dvbs_rates_of_table_d1():
    # Annex D, Table D.1: 25.776 Mbaud, useful bit rate per inner code rate (Mbit/s)
    for rate, ru in zip(
        ["1/2", "2/3", "3/4", "5/6", "7/8"], [23.754, 31.672, 35.631, 39.590, 41.570], strict=True
    ):
        assert abs(comnum.Dvbs(rate).useful_bit_rate(25.776e6) / 1e6 - ru) < 6e-4


def test_dvbs_constellation_is_figure_5():
    p = np.asarray(comnum.Dvbs().constellation.points)
    # labels 00, 01, 10, 11 (I bit first): quadrants 1, 4, 2, 3
    np.testing.assert_allclose(p * np.sqrt(2), [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])


def test_dvbs_rejects_a_packet_without_sync_byte():
    packets = comnum.dvb_packets(2, comnum.Rng(1))
    packets[188] = 0
    with pytest.raises(ValueError, match="sync"):
        comnum.Dvbs().modulate(packets)
    with pytest.raises(ValueError, match="255"):
        comnum.Dvbs().modulate(np.full(188, 300))


@pytest.mark.parametrize(
    ("constellation", "rate", "packets"),
    [("QPSK", "1/2", 252), ("16-QAM", "2/3", 672), ("64-QAM", "7/8", 1323)],
)
def test_dvbt_packets_per_superframe_of_table_16(constellation, rate, packets):
    assert comnum.Dvbt("2K", "1/4", constellation, rate).packets_per_superframe == packets
    assert comnum.Dvbt("8K", "1/4", constellation, rate).packets_per_superframe == 4 * packets


def test_dvbt_useful_bit_rates_of_table_17():
    # Table 17 (8 MHz): 64-QAM 7/8 for the guard intervals 1/4, 1/8, 1/16, 1/32 (Mbit/s)
    for guard, ru in zip(["1/4", "1/8", "1/16", "1/32"], [26.13, 29.03, 30.74, 31.67], strict=True):
        assert (
            abs(comnum.Dvbt("8K", guard, "64-QAM", "7/8").useful_bit_rate(8e6) / 1e6 - ru) < 0.005
        )


def test_dvbt_round_trips_through_the_ofdm_signal_and_signals_its_parameters():
    link = comnum.Dvbt("2K", "1/32", "64-QAM", "3/4")
    packets = comnum.dvb_packets(400, comnum.Rng(4))
    x = link.modulate(packets)
    assert len(x) == link.symbols_len(400) * (2048 + 64)
    cells = link.demodulate(x)
    out, corrected, failed = link.decode(cells, 400, 1e-4)
    np.testing.assert_array_equal(out, packets)
    assert (corrected, failed) == (0, 0)
    tps = link.decode_tps(cells[:, :68])
    assert (tps["mode"], tps["guard"], tps["constellation"], tps["rate_hp"]) == (
        "2K",
        "1/32",
        "64-QAM",
        "3/4",
    )
    assert tps["hierarchy"] == 0 and tps["rate_lp"] is None


def test_dvbt_decodes_over_the_portable_channel_with_known_gains():
    link = comnum.Dvbt("2K", "1/4", "QPSK", "1/2")
    packets = comnum.dvb_packets(40, comnum.Rng(5))
    cells = link.cells(comnum.Dvbs().outer(packets))
    h = comnum.dvbt_channel_gains("P1", link.carrier_frequencies())
    assert abs(np.mean(np.abs(h) ** 2) - 1) < 0.2
    noise_var = link.carrier_noise_var(12.0)
    rng = np.random.default_rng(5)
    w = np.sqrt(noise_var / 2) * (rng.normal(size=cells.shape) + 1j * rng.normal(size=cells.shape))
    out, _, failed = link.decode(cells * h[:, None] + w, 40, noise_var, channel=h)
    np.testing.assert_array_equal(out, packets)
    assert failed == 0
    with pytest.raises(ValueError, match="profile"):
        comnum.dvbt_channel_gains("P2", [0.0])


def test_dvbt_constellation_is_figure_9():
    p = np.asarray(comnum.Dvbt("2K", "1/4", "16-QAM", "1/2").constellation.points) * np.sqrt(10)
    # labels 1000, 1010, 0010, 0000 on the row Im = 3 (Figure 9a)
    np.testing.assert_allclose(
        p[[0b1000, 0b1010, 0b0010, 0b0000]], [-3 + 3j, -1 + 3j, 1 + 3j, 3 + 3j]
    )


# CCSDS 131.0-B-5


def test_ccsds_rs_corrects_e_octets_per_codeword_of_an_interleaved_codeblock():
    rs = comnum.CcsdsRs(8, 4, fill=10)
    assert (rs.frame_len, rs.codeblock_len) == ((255 - 16 - 10) * 4, (255 - 10) * 4)
    frame = np.random.default_rng(1).integers(0, 256, rs.frame_len)
    block = rs.encode_bytes(frame)
    np.testing.assert_array_equal(block[: rs.frame_len], frame)
    block[100:132] ^= 0x01  # 8 octets in each of the 4 codewords
    decoded, corrected = rs.decode_bytes(block)
    np.testing.assert_array_equal(decoded, frame)
    assert corrected == 32
    block[200:204] ^= 0x01  # a ninth error in each codeword
    _, corrected = rs.decode_bytes(block)
    assert corrected is None
    bits = comnum.bits(rs.k, comnum.Rng(1))
    np.testing.assert_array_equal(rs.decode(1.0 - 2.0 * rs.encode(bits)), bits)
    with pytest.raises(ValueError, match="depth"):
        comnum.CcsdsRs(16, 6)


@pytest.mark.parametrize(("rate", "k", "n"), [("1/2", 1024, 2048), ("4/5", 1024, 1280)])
def test_ccsds_ldpc_codewords_satisfy_the_full_parity_checks(rate, k, n):
    code = comnum.CcsdsLdpc(rate, k)
    assert (code.k, code.n, code.rate) == (k, n, rate)
    b = comnum.bits(k, comnum.Rng(7))
    c = code.encode(b)
    np.testing.assert_array_equal(c[:k], b)
    full = code.full_codeword(b)
    np.testing.assert_array_equal(full[:n], c)
    assert code.ldpc.is_codeword(full)
    app = code.app(3.0 - 6.0 * c)
    np.testing.assert_array_equal((app < 0).astype(np.uint8), b)
    with pytest.raises(ValueError, match="k"):
        comnum.CcsdsLdpc("1/2")
    with pytest.raises(ValueError):
        comnum.CcsdsLdpc("1/2", 2048)


def test_ccsds_ldpc_rate_7_8_has_the_virtual_zeros_and_two_appended_zeros():
    code = comnum.CcsdsLdpc("7/8")
    assert (code.k, code.n) == (7136, 8160)
    b = comnum.bits(7136, comnum.Rng(8))
    c = code.encode(b)
    np.testing.assert_array_equal(c[-2:], [0, 0])
    full = code.full_codeword(b)
    assert len(full) == 8176 and not full[:18].any()
    assert code.ldpc.is_codeword(full)


@pytest.mark.parametrize(("rate", "n"), [("1/2", 3576), ("1/4", 7152), ("1/6", 10728)])
def test_ccsds_turbo_lengths_are_table_6_2_and_decode_cleanly(rate, n):
    code = comnum.CcsdsTurbo(1784, rate, iterations=4)
    assert (code.k, code.n, code.rate) == (1784, n, rate)
    perm = code.permutation
    np.testing.assert_array_equal(np.sort(perm), np.arange(1784))
    b = comnum.bits(1784, comnum.Rng(9))
    np.testing.assert_array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b)
    with pytest.raises(ValueError):
        comnum.CcsdsTurbo(1000)


def test_ccsds_convolutional_code_inverts_g2_at_rate_one_half_only():
    half = comnum.CcsdsConvolutional("1/2")
    impulse = half.encode([1, 0, 0, 0, 0, 0, 0])
    # G1 = 171 (1111001) and G2 = 133 (1011011) inverted (Sec. 3.3)
    np.testing.assert_array_equal(impulse[0::2], [1, 1, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(impulse[1::2], [0, 1, 0, 0, 1, 0, 0])
    bits = comnum.bits(64, comnum.Rng(10))
    np.testing.assert_array_equal(half.decode(1.0 - 2.0 * half.encode(bits), 64), bits)
    seven = comnum.CcsdsConvolutional("7/8")
    c = seven.encode(bits[:56])
    assert len(c) == 64
    np.testing.assert_array_equal(seven.decode(1.0 - 2.0 * c, 56), bits[:56])
    with pytest.raises(ValueError):
        comnum.CcsdsConvolutional("4/5")


def test_ccsds_randomizers_and_markers_are_those_of_sections_9_and_10():
    long = comnum.ccsds_randomizer("long", 131071 + 17)
    np.testing.assert_array_equal(long[:17], long[131071:])
    short = comnum.ccsds_randomizer("short", 255 * 2)
    np.testing.assert_array_equal(short[:255], short[255:])
    assert comnum.ccsds_marker("turbo_1_2").tobytes().hex() == "034776c7272895b0"
    assert comnum.ccsds_marker("embedded").tobytes().hex() == "352ef853"
    assert [len(comnum.ccsds_marker(f"turbo_1_{r}")) for r in (3, 4, 6)] == [12, 16, 24]
    block = comnum.bits(80, comnum.Rng(11))
    cadu = comnum.ccsds_cadu(comnum.ccsds_marker("asm"), block, "long")
    np.testing.assert_array_equal(cadu[32:] ^ comnum.ccsds_randomizer("long", 80), block)
    np.testing.assert_array_equal(comnum.ccsds_cadu(comnum.ccsds_marker("asm"), block)[32:], block)
    with pytest.raises(ValueError, match="marker"):
        comnum.ccsds_marker("asm_2")
    with pytest.raises(ValueError, match="randomizer"):
        comnum.ccsds_randomizer("medium", 8)


def test_lte_cell_search_and_mib_after_the_front_end():
    """A 5 MHz cell of two ports through the receiver of a measured capture."""
    rng = comnum.Rng(36211)
    gen = np.random.default_rng(36211)
    fs = 7.68e6
    frames = [comnum.lte_frame(287, sfn=s, n_rb=25, ports=2, rng=rng) for s in range(668, 673)]
    h = np.array([1.0, 0.6 * np.exp(1j * 0.7)])
    x = np.concatenate([h @ f for f in frames])
    x = np.concatenate([np.zeros(12_345), x])  # a delay of 12345 samples
    n = np.arange(len(x))
    x = x * np.exp(2j * np.pi * 13.4e3 * n / fs)  # the oscillator of the receiver 13.4 kHz off
    sigma = np.sqrt(np.mean(np.abs(x) ** 2) / 10**0.5)  # 5 dB
    x = x + sigma * (gen.standard_normal(len(x)) + 1j * gen.standard_normal(len(x))) / np.sqrt(2)
    y = comnum.select_channel(x, fs, 0.0, 1.92e6, 0.6e6)
    assert len(y) == -(-len(x) // 4)
    cell = comnum.lte_cell_search(y)[0]
    assert cell["n_cell_id"] == 287
    assert abs(cell["cfo_hz"] - 13.4e3) < 50
    assert abs(cell["frame_start"] - 12_345 / 4) <= 1
    decoded = comnum.lte_decode_mib(y, cell)
    assert len(decoded) >= 4
    for f in decoded:
        assert (f["ports"], f["n_rb"], f["phich_extended"], f["phich_ng"]) == (2, 25, False, 1.0)
        assert f["sfn"] == 668 + round((f["start"] - cell["frame_start"]) / 19200)
        assert f["spare"] == [0] * 10


def test_lte_pss_is_the_punctured_zadoff_chu_sequence():
    for n_id2, u in enumerate((25, 29, 34)):
        np.testing.assert_allclose(
            comnum.lte_pss(n_id2), np.delete(comnum.zadoff_chu(u, 63), 31), atol=1e-12
        )
    with pytest.raises(ValueError):
        comnum.zadoff_chu(21, 63)  # not prime to 63
    with pytest.raises(ValueError):
        comnum.lte_frame(0, n_rb=7)
    with pytest.raises(ValueError):
        comnum.lte_cell_search(np.zeros(100, complex))
    with pytest.raises(ValueError):
        comnum.lte_decode_mib(np.zeros(100, complex), {"n_cell_id": 0})
    with pytest.raises(ValueError):
        comnum.select_channel(np.ones(10), 1e6, 0.0, 1e6, 6e5)


def test_fm_station_with_rds_through_the_front_end():
    """A stereo station with its RDS through FM, a capture at 2.4 MS/s and the receiver."""
    fs = 228e3
    n = int(1.2 * fs)
    t = np.arange(n) / fs
    left, right = 0.6 * np.sin(2 * np.pi * 1e3 * t), 0.3 * np.sin(2 * np.pi * 2.5e3 * t)
    ps = b"PYTHON 1"
    groups = [
        comnum.rds_group_bits([0xF20C, s, 0, ps[2 * s] << 8 | ps[2 * s + 1]]) for s in range(4)
    ]
    groups.append(
        comnum.rds_group_bits([0xF20C, 1 << 11, 0xF20C, ord("O") << 8 | ord("K")], version_b=True)
    )
    d = comnum.rds_differential_encode(np.concatenate(groups))
    m = comnum.fm_multiplex(left, right, fs, rds_bits=d)
    assert np.max(np.abs(m)) < 75e3
    x = comnum.resample_rational(comnum.fm_modulate(m, fs, 1.0), fs, 2.4e6, 110e3)
    x = x * np.exp(2j * np.pi * 300e3 * np.arange(len(x)) / 2.4e6)
    y = comnum.select_channel(x, 2.4e6, 300e3, fs, 110e3)
    f = comnum.fm_discriminator(y, fs)
    r = comnum.rds_receive(f, fs)
    assert (r["pi"], r["ps"], r["group_types"]["0A"] > 0) == (0xF20C, "PYTHON 1", True)
    assert r["valid_blocks"] >= 0.9 * r["blocks"] > 0  # the edges of the filters may cost a group
    assert all(len(g) == 4 for g in r["groups"])
    a = comnum.fm_stereo_decode(f, fs)
    assert a["stereo"] and a["fs"] == 48e3 and abs(a["pilot_hz"] - 6750) < 5
    rms = np.sqrt(np.mean(a["right"][12000:36000] ** 2))
    assert abs(rms / (0.3 / np.sqrt(2)) - 1) < 0.05


def test_ais_packets_through_the_front_end():
    """Three AIS position reports through GMSK, a capture at 2 MS/s tuned 25 kHz below
    the channel at 12 dB, the front end and the receiver; the sentences and the fields back."""
    fs = comnum.ais_front_end_rate()
    reports = [
        comnum.ais_position_report(
            200000000 + k,
            lat_deg=46.12,
            lon_deg=-9.88,
            sog_knots=8.4,
            cog_deg=271.5,
            heading_deg=270,
            timestamp=42,
        )
        for k in range(3)
    ]
    x = comnum.ais_transmit(reports, fs)
    assert abs(np.abs(x[len(x) // 2]) - 1.0) < 1e-9 or np.abs(x[len(x) // 2]) == 0.0
    up = comnum.resample_rational(x, fs, 2e6, 7e3)
    rng = np.random.default_rng(1371)
    # 12 dB in the 25 kHz channel: every packet of 40 draws decoded (89 % at 10 dB)
    sigma = np.sqrt(2e6 / 25e3 / 10**1.2)
    cap = up * np.exp(2j * np.pi * 25e3 * np.arange(len(up)) / 2e6)
    cap = cap + sigma * (
        rng.standard_normal(len(cap)) + 1j * rng.standard_normal(len(cap))
    ) / np.sqrt(2)
    y = comnum.select_channel(cap, 2e6, 25e3, fs, 12.5e3)
    packets = comnum.ais_receive(y, fs)
    mmsi = [p["message"]["mmsi"] for p in packets if "message" in p]
    assert mmsi == [200000000, 200000001, 200000002]
    p = packets[0]["message"]["position"]
    assert (round(p["lat_deg"], 4), round(p["lon_deg"], 4), p["sog_knots"], p["heading_deg"]) == (
        46.12,
        -9.88,
        8.4,
        270,
    )
    assert packets[0]["aivdm"][0].startswith("!AIVDM,1,1,,A,")
    assert np.array_equal(comnum.ais_messages(packets[0]["aivdm"])[0]["bits"], reports[0])
    with pytest.raises(ValueError):
        comnum.ais_receive(y, 50e3)


def test_fm_emphasis_and_discriminator():
    x = np.sin(0.01 * np.arange(1000))
    np.testing.assert_allclose(
        comnum.fm_deemphasis(comnum.fm_preemphasis(x, 228e3), 228e3), x, atol=1e-9
    )
    f = comnum.fm_discriminator(comnum.fm_modulate(x, 228e3, 75e3), 228e3)
    np.testing.assert_allclose(f[1:], 75e3 * x[1:], atol=1e-6)
    with pytest.raises(ValueError):
        comnum.fm_deemphasis(x, 228e3, tau_s=0.0)
    with pytest.raises(ValueError):
        comnum.rds_receive(np.zeros(100), 228e3)
    with pytest.raises(ValueError):
        comnum.fm_multiplex(x, x[1:], 228e3)
    with pytest.raises(ValueError):
        comnum.rds_differential_encode([0, 2])
