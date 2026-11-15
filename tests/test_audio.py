"""A sound card as a channel: text as bits, HDLC frames, noncoherent FSK, the audio
carrier and the WAV files, and the whole chain of a text message through a simulated
room."""

import numpy as np
import pytest

import comnum


def test_text_and_bits_are_utf8_most_significant_bit_first():
    assert comnum.text_to_bits("A").tolist() == [0, 1, 0, 0, 0, 0, 0, 1]
    text = "Bonjour, ça va ? ✓"
    bits = comnum.text_to_bits(text)
    assert bits.dtype == np.uint8 and bits.size == 8 * len(text.encode())
    assert comnum.bits_to_text(bits) == text
    assert comnum.bits_to_text(np.concatenate([bits, [1, 0, 1]])) == text
    # a bit in error gives a wrong character, not an exception
    bits[0] ^= 1
    assert comnum.bits_to_text(bits).endswith("onjour, ça va ? ✓")
    with pytest.raises(ValueError, match="bits must be 0 or 1"):
        comnum.bits_to_text([0, 2])


def test_hdlc_frames_are_found_in_a_stream_and_checked():
    frame = comnum.hdlc_frame("hello")
    assert frame[:8].tolist() == [0, 1, 1, 1, 1, 1, 1, 0]
    assert comnum.hdlc_frames(frame) == [(b"hello", True)]
    assert comnum.hdlc_frames(comnum.hdlc_frame(b"\xff\x7e\x00")) == [(b"\xff\x7e\x00", True)]
    # in the middle of other bits, and with a bit in error
    rng = comnum.Rng(1)
    stream = np.concatenate([comnum.bits(40, rng), frame, comnum.bits(40, rng)])
    assert (b"hello", True) in comnum.hdlc_frames(stream)
    stream[40 + 12] ^= 1
    assert all(not ok for payload, ok in comnum.hdlc_frames(stream) if payload != b"hello")
    assert (b"hello", True) not in comnum.hdlc_frames(stream)


def test_fsk_round_trip_timing_and_error_rate():
    fsk = comnum.Fsk(4, 16, spacing=2)
    assert (fsk.m, fsk.sps, fsk.spacing) == (4, 16, 2)
    rng = comnum.Rng(2)
    symbols = rng.integers(4, 50_000)
    x = fsk.modulate(symbols)
    np.testing.assert_allclose(np.abs(x[:2000]), 1.0, atol=1e-12)
    np.testing.assert_array_equal(fsk.demodulate(x), symbols)
    # an unknown phase and a late start
    late = np.concatenate([np.zeros(5, complex), x[:4000] * np.exp(1j)])
    assert fsk.timing(late) in (5, 6)
    np.testing.assert_array_equal(fsk.demodulate(late[5:]), symbols[:250])
    # Eb/N0 = 7 dB: the closed form of four orthogonal signals
    noisy = x + rng.complex_noise(x.size, 16 / (2 * comnum.db2lin(7.0)))
    wrong = fsk.demodulate(noisy) ^ symbols
    ber = np.mean((wrong >> 1) + (wrong & 1)) / 2
    assert ber == pytest.approx(comnum.theory_ber_fsk_noncoherent(4, 7.0), rel=0.1)
    assert comnum.theory_ber_fsk_noncoherent(2, 10.0) == pytest.approx(0.5 * np.exp(-5.0))
    with pytest.raises(ValueError, match="samples per symbol"):
        comnum.Fsk(4, 7, spacing=2)


def test_a_baseband_signal_goes_through_the_carrier_and_a_wav_file(tmp_path):
    fs, carrier = 4800, 9000.0
    x = 0.6 * np.exp(2j * np.pi * 300 * np.arange(960) / fs)
    s = comnum.audio_upconvert(x, fs, carrier)
    assert s.size == 9600 and s.dtype == np.float64
    path = tmp_path / "tone.wav"
    comnum.write_wav(path, s)
    samples, rate = comnum.read_wav(path)
    assert rate == 48_000 and samples.shape == (9600,)
    np.testing.assert_allclose(samples, s, atol=1e-4)  # 16 bits
    y = comnum.audio_downconvert(samples, carrier, fs, rate)
    np.testing.assert_allclose(y[100:860], x[100:860], atol=2e-3)
    with pytest.raises(ValueError, match="does not fit"):
        comnum.audio_upconvert(x, fs, 1000.0)
    with pytest.raises(OSError, match="missing"):
        comnum.read_wav(tmp_path / "missing.wav")
    (tmp_path / "bad.wav").write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="WAV"):
        comnum.read_wav(tmp_path / "bad.wav")


def test_a_text_message_crosses_a_simulated_room(tmp_path):
    message = "Bonjour ! Ce message est passé par le haut-parleur. ✓"
    fsk = comnum.Fsk(4, 24, spacing=2)
    fs, carrier = 2400, 6000.0
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    bits = np.concatenate([np.tile(flag, 4), comnum.hdlc_frame(message), flag])
    bits = np.concatenate([bits, np.zeros(bits.size % 2, dtype=np.uint8)])
    audio = 0.5 * comnum.audio_upconvert(fsk.modulate(2 * bits[0::2] + bits[1::2]), fs, carrier)
    comnum.write_wav(tmp_path / "message.wav", audio)
    played, rate = comnum.read_wav(tmp_path / "message.wav")
    rng = np.random.default_rng(3)
    for ppm, delay, echo in [(60, 31_234, 192), (-90, 5_001, 700), (20, 120_000, 1500)]:
        # the clock of the second card, a delay, an echo at half the amplitude, noise
        skew = 1 + ppm * 1e-6
        t = np.arange(int(played.size * skew)) / skew
        recording = np.concatenate(
            [np.zeros(delay), np.interp(t, np.arange(played.size), played), np.zeros(9000)]
        )
        recording[echo:] += 0.5 * recording[:-echo]
        recording += 0.05 * rng.standard_normal(recording.size)
        y = comnum.audio_downconvert(recording, carrier + 3.0, fs, rate)  # 3 Hz off
        decided = fsk.demodulate(y[fsk.timing(y) :])
        received = np.stack([decided >> 1, decided & 1], axis=1).ravel()
        assert (message.encode(), True) in comnum.hdlc_frames(received)
