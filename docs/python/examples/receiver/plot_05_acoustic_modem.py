"""
A text message through a loudspeaker and a microphone
=====================================================

A sound card is a radio that everybody owns: a loudspeaker for the transmitter, a
microphone for the receiver, the air of the room for the channel. This tutorial sends
a text from one computer to another through it, with the blocks of the library, and
receives it with a few lines of Python: the chain of a real link, where nothing is
aligned and nothing is known in advance.

**The transmitter**

1. *Text to bits.* :func:`comnum.text_to_bits` gives the bits of the UTF-8 bytes. Here
   the bytes go into an HDLC frame instead (:func:`comnum.hdlc_frame`): a flag
   ``01111110`` on each side, the payload with a 16-bit check sequence, and a 0
   stuffed after five consecutive 1 so that the flag cannot appear inside. The
   receiver will find the message by its flags, wherever it starts, and know from the
   check sequence whether it is intact.
2. *Bits to a waveform.* Two bits choose one of four tones, held for a symbol:
   :class:`comnum.Fsk`, orthogonal 4-FSK. A noncoherent receiver compares the energies
   of the four tones and needs neither the phase nor the exact frequency of the
   carrier, which a pair of sound cards will not give.
3. *Baseband to sound.* :func:`comnum.audio_upconvert` resamples the complex baseband
   signal :math:`x` to the 48 kHz of the card and plays it on an audio carrier,
   :math:`s[n] = \\operatorname{Re}\\bigl(x[n]\\,e^{j2\\pi f_c n/f_a}\\bigr)`;
   :func:`comnum.write_wav` writes the file to play.

**The channel** adds what a cable would not: an unknown delay, the echoes of the room,
the noise, and two clocks that differ by tens of parts per million, so that the
receiver sees the carrier *and* the symbol rate slightly off.

**The receiver**

1. :func:`comnum.read_wav` and :func:`comnum.audio_downconvert` bring the band of the
   carrier back to baseband.
2. ``Fsk.timing`` finds where the symbols start, ``Fsk.demodulate`` decides them.
3. :func:`comnum.hdlc_frames` finds the frames in the bits and checks them.

The room is simulated below so that the page builds anywhere; the last section says
how to do it with two computers.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, SERIES

AUDIO_RATE = 48_000  # the rate of the sound card (Hz)
CARRIER_HZ = 6_000.0  # the audio carrier
BAUD = 100  # symbols per second
SPS = 24  # samples per symbol at baseband: 2.4 kHz
FS = BAUD * SPS

fsk = comnum.Fsk(4, SPS, spacing=2)  # four tones 200 Hz apart

# %%
# The transmitter
# ---------------
# A few flags first let the receiver settle; then the frame.
message = "Bonjour ! Ce message est passé par le haut-parleur. ✓"
print(f"{len(message)} characters, {comnum.text_to_bits(message).size} bits of UTF-8")
frame = comnum.hdlc_frame(message)
flag = [0, 1, 1, 1, 1, 1, 1, 0]
bits = np.concatenate([np.tile(flag, 4), frame, flag])
bits = np.concatenate([bits, np.zeros(bits.size % 2, dtype=np.uint8)])
symbols = 2 * bits[0::2] + bits[1::2]  # two bits per tone
print(f"{frame.size} bits in the frame, {symbols.size} symbols, {symbols.size / BAUD:.1f} s")

baseband = fsk.modulate(symbols)
audio = 0.5 * comnum.audio_upconvert(baseband, FS, CARRIER_HZ, AUDIO_RATE)
audio = np.concatenate([np.zeros(AUDIO_RATE // 4), audio, np.zeros(AUDIO_RATE // 4)])

folder = Path(tempfile.mkdtemp())
comnum.write_wav(folder / "message.wav", audio, AUDIO_RATE)
print(f"message.wav: {audio.size / AUDIO_RATE:.2f} s at {AUDIO_RATE} Hz")

# %%
# The four tones sit at 6.0, 6.2, 6.4 and 6.6 kHz, one per symbol of 10 ms.
fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5))
start = AUDIO_RATE // 4
t_ms = 1e3 * np.arange(3 * AUDIO_RATE // BAUD) / AUDIO_RATE
top.plot(t_ms, audio[start : start + t_ms.size], color=SERIES[0], linewidth=0.6)
top.set(xlabel="time (ms)", ylabel="amplitude", title="The first three symbols on the card")
signal = audio[start : start + AUDIO_RATE]  # the first second, without the silence
bottom.specgram(signal, NFFT=480, Fs=AUDIO_RATE, noverlap=240, cmap="Greys")
bottom.set(
    ylim=(5500, 7100),
    xlim=(0.0, 0.8),
    xlabel="time (s)",
    ylabel="frequency (Hz)",
    title="Spectrogram: one tone per symbol",
)
fig.tight_layout()

# %%
# The room
# --------
# What the second computer records: silence first, the signal with an echo 4 ms later
# (1.4 m of extra path) at half the amplitude, noise, and a card whose clock runs 60
# parts per million fast.
rng = comnum.Rng(7)
skew = 1 + 60e-6
t = np.arange(int(audio.size * skew)) / skew
played = np.interp(t, np.arange(audio.size), audio)
recording = np.concatenate([np.zeros(31_234), played, np.zeros(10_000)])
recording[192:] += 0.5 * recording[:-192]
recording += 0.05 * rng.gaussian(recording.size)
comnum.write_wav(folder / "recording.wav", recording, AUDIO_RATE)

# %%
# The receiver
# ------------
# Six lines: the file, the band of the carrier at baseband, the start of the symbols,
# the decisions, the bits, the frames.
samples, rate = comnum.read_wav(folder / "recording.wav")
y = comnum.audio_downconvert(samples, CARRIER_HZ, FS, rate)
offset = fsk.timing(y)
decided = fsk.demodulate(y[offset:])
received_bits = np.stack([decided >> 1, decided & 1], axis=1).ravel()
for payload, intact in comnum.hdlc_frames(received_bits):
    if intact:
        print(f"received: {payload.decode()!r}")
assert any(ok and p.decode() == message for p, ok in comnum.hdlc_frames(received_bits))

# %%
# What the receiver saw: the energies of the four tones, symbol after symbol. Before
# the signal, noise; then the flags, a repeating pattern; then the message. The
# decision is the darkest tone of every column. The two upper tones arrive weaker than
# the two lower ones: the echo adds to the direct path at 6.0 kHz and nearly cancels it
# at 6.4 kHz, the frequency selectivity of a room in one picture.
energies = fsk.energies(y[offset:]).reshape(-1, 4).T
first = int(np.argmax(energies.max(axis=0) > 0.25 * energies.max())) - 5
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.imshow(
    energies[:, first : first + 70] / energies.max(),
    aspect="auto",
    origin="lower",
    cmap="Greys",
    extent=(0, 70, -0.5, 3.5),
)
ax.set(xlabel="symbol", ylabel="tone", yticks=range(4), title="Energy of the four tones")
fig.tight_layout()

# %%
# How much noise can it take?
# ---------------------------
# Over white noise the bit error rate of the noncoherent receiver is that of four
# orthogonal signals (:func:`comnum.theory_ber_fsk_noncoherent`; Proakis & Salehi, Sec.
# 4.5). The same modem at baseband, with the noise of a given :math:`E_b/N_0`: the
# simulated points fall on the curve. A message of 500 bits needs a bit error rate
# below :math:`10^{-3}` or so to arrive intact, about 8 dB.
ebn0_db = np.arange(0.0, 11.0, 2.0)
ebn0_fine = np.linspace(0, 11, 111)
generator = comnum.Rng(11)
simulated = []
for e in ebn0_db:
    sent = generator.integers(4, 100_000)
    got = fsk.demodulate(comnum.awgn_ebn0(fsk.modulate(sent), e, 2, generator, sps=SPS))
    wrong = sent ^ got
    simulated.append(np.mean((wrong >> 1) + (wrong & 1)) / 2)

fig, ax = plt.subplots()
ax.semilogy(
    ebn0_fine,
    comnum.theory_ber_fsk_noncoherent(4, ebn0_fine),
    color=GREY,
    label="theory, noncoherent 4-FSK",
)
ax.plot(ebn0_db, simulated, "o", color=SERIES[0], markeredgecolor=BACKGROUND, label="simulated")
ax.set(ylim=(1e-5, 0.5), xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate")
ax.legend()
fig.tight_layout()

expected = comnum.theory_ber_fsk_noncoherent(4, ebn0_db)
keep = expected * 200_000 >= 400
gap = np.max(np.abs(np.array(simulated)[keep] / expected[keep] - 1))
print(f"simulated within {100 * gap:.0f} % of the theory")

# %%
# With two computers
# ------------------
# Write ``message.wav`` where you can play it (``comnum.write_wav("message.wav", audio)``),
# start a recording at 48 kHz on the second computer (Audacity, or the voice recorder of
# a telephone exported as WAV), play the file, stop, and run the six lines of the
# receiver on the recording. Things worth trying with a class:
#
# * move the microphone away, or talk during the transmission: the check sequence says
#   when the frame is lost;
# * raise ``BAUD``: symbols shorter than the echoes of the room overlap, and the link
#   breaks well before the noise does it;
# * change ``CARRIER_HZ``: small loudspeakers pass little below 300 Hz, and many
#   microphones little above 10 kHz;
# * look at ``comnum.psd(samples, fs=rate)`` of a recording: the four tones, and
#   everything else the room contains.
#
# A message of a few seconds survives the clock difference of two cards without
# tracking it: 100 parts per million drift by a tenth of a symbol in 10 s at 100 baud.
# Longer messages, or a coherent modulation such as QPSK, need the timing and carrier
# recovery of :doc:`plot_03_sync_mimo_ofdm` and of the user guide. The playground
# exports the signal of any scope as a WAV file on a carrier (the **WAV** button of a
# scope tab), to be recorded and processed the same way: its tutorial
# ``13_text_fsk`` is this chain, with the blocks ``text_source`` and ``fsk_tx``.
