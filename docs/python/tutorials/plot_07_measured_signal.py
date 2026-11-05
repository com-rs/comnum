"""
A measured signal
=================

A capture of a software radio goes through the same blocks as a simulation. This
tutorial takes two: the LTE downlink of a cell (its identity and the master
information block of every frame), and an FM station with its stereo multiplex and
its radio data system. Both are simulated here, so that the page builds anywhere; the
same calls read the recordings of a HackRF One at the end of the page.

The front end is :func:`comnum.select_channel`: it removes the DC offset of the
converter, shifts the channel of interest to 0 Hz and resamples it by the exact
rational ratio of the two rates under a low-pass filter whose window rejects the
neighbouring channels by 80 dB.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND

# %%
# **An LTE cell.** Twenty frames of a 5 MHz cell of two antenna ports at 7.68 MS/s
# (:func:`comnum.lte_frame` builds the PSS, the SSS, the cell-specific reference
# signals and the PBCH of 3GPP TS 36.211), through a flat two-port channel, an
# oscillator 13 kHz off and 5 dB of SNR: the kind of signal a HackRF records.
rng = comnum.Rng(287)
fs = 7.68e6
frames = [comnum.lte_frame(287, sfn=s, n_rb=25, ports=2, rng=rng) for s in range(668, 688)]
h = np.array([1.0, 0.6 * np.exp(0.7j)])
x = np.concatenate([h @ f for f in frames])
n = np.arange(len(x))
x = x * np.exp(2j * np.pi * 13.4e3 * n / fs)
gen = np.random.default_rng(1)
sigma = np.sqrt(np.mean(np.abs(x) ** 2) / 10**0.5)
x = x + sigma * (gen.standard_normal(len(x)) + 1j * gen.standard_normal(len(x))) / np.sqrt(2)

# %%
# The six central resource blocks of an LTE carrier carry the synchronisation signals
# and the broadcast channel in every bandwidth, so the receiver reads them at
# 1.92 MS/s. The cell search tries the three PSS over a grid of frequency offsets,
# refines the offset on the cyclic prefixes, and reads the SSS to get the identity
# and the start of the frames; the MIB is then decoded in every frame, its timing
# tracked on the PSS.
y = comnum.select_channel(x, fs, 0.0, 1.92e6, 0.6e6)
cell = comnum.lte_cell_search(y)[0]
mib = comnum.lte_decode_mib(y, cell)
print(
    f"cell {cell['n_cell_id']}, offset {cell['cfo_hz']:+.0f} Hz, PSS metric {cell['pss_metric']:.2f}"
)
print(f"MIB decoded in {len(mib)} frames; the first: {mib[0]}")

# %%
# The correlation of the capture with the PSS, after the offset is removed: one peak
# every 5 ms, the signature of the cell.
yy = y * np.exp(-2j * np.pi * cell["cfo_hz"] * np.arange(len(y)) / 1.92e6)
pss = comnum.lte_pss(cell["n_cell_id"] % 3)
replica = np.fft.ifft(np.concatenate([[0], pss[31:], np.zeros(65), pss[:31]]))
seg = yy[: int(0.0205 * 1.92e6)]
c = np.abs(np.correlate(seg, replica, "valid")) ** 2
e = np.convolve(np.abs(seg) ** 2, np.ones(128), "valid") * np.vdot(replica, replica).real
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(np.arange(len(c)) / 1.92e3, c / e, lw=1)
ax.set(xlabel="time (ms)", ylabel="normalised PSS correlation", ylim=(0, 1))
ax.set_title(f"cell {cell['n_cell_id']}: the PSS of slots 0 and 10", fontsize=10)
fig.tight_layout()

# %%
# **An FM station.** The stereophonic multiplex of ITU-R BS.450-4 (the pilot at
# 19 kHz, the difference channel around 38 kHz) with the RDS of NRSC-4 / EN 50067 on
# its 57 kHz subcarrier: the name of the station in groups of type 0A, a radiotext in
# groups 2A. The multiplex frequency-modulates a carrier with 75 kHz of maximum
# deviation, which a capture at 2.4 MS/s receives 300 kHz above its tuning frequency.
fs_mpx = 228e3
t = np.arange(int(1.2 * fs_mpx)) / fs_mpx
left, right = 0.6 * np.sin(2 * np.pi * 1e3 * t), 0.3 * np.sin(2 * np.pi * 2.5e3 * t)
ps, text = b"TUTORIAL", b"Hello, comnum!  "
groups = [comnum.rds_group_bits([0xF20C, s, 0, ps[2 * s] << 8 | ps[2 * s + 1]]) for s in range(4)]
for s in range(4):
    c4 = text[4 * s : 4 * s + 4]
    groups.append(
        comnum.rds_group_bits([0xF20C, (2 << 12) | s, c4[0] << 8 | c4[1], c4[2] << 8 | c4[3]])
    )
d = comnum.rds_differential_encode(np.concatenate(groups))
mpx = comnum.fm_multiplex(left, right, fs_mpx, rds_bits=d)
capture = comnum.resample_rational(comnum.fm_modulate(mpx, fs_mpx, 1.0), fs_mpx, 2.4e6, 110e3)
capture = capture * np.exp(2j * np.pi * 300e3 * np.arange(len(capture)) / 2.4e6)

# %%
# The receiver: the front end to 228 kHz, the polar discriminator
# (:func:`comnum.fm_discriminator` gives the multiplex in hertz of deviation), then
# the stereo decoder and the RDS receiver on the same multiplex.
y = comnum.select_channel(capture, 2.4e6, 300e3, fs_mpx, 110e3)
m = comnum.fm_discriminator(y, fs_mpx)
audio = comnum.fm_stereo_decode(m, fs_mpx)
rds = comnum.rds_receive(m, fs_mpx)
print(f"stereo: {audio['stereo']}, pilot {audio['pilot_hz']:.0f} Hz of deviation")
print(
    f"PI {rds['pi']:#06x}, PS {rds['ps']!r}, radiotext {rds['radiotext']!r}, {rds['valid_blocks']} of {rds['blocks']} blocks"
)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
f = np.fft.rfftfreq(len(m), 1 / fs_mpx)
spectrum = np.abs(np.fft.rfft((m - m.mean()) * np.hanning(len(m)))) ** 2
axes[0].semilogy(f / 1e3, spectrum / spectrum.max(), lw=0.6)
axes[0].set(xlabel="frequency (kHz)", ylabel="multiplex spectrum", xlim=(0, 100), ylim=(1e-9, 2))
for pos, name in ((19, "pilot"), (38, "L − R"), (57, "RDS")):
    axes[0].annotate(name, (pos, 1.0), (pos, 1.2), ha="center", fontsize=8)
ta = np.arange(2400) / audio["fs"] * 1e3
axes[1].plot(ta, audio["left"][24000:26400], lw=0.8, label="left, 1 kHz")
axes[1].plot(ta, audio["right"][24000:26400], lw=0.8, label="right, 2.5 kHz")
axes[1].set(xlabel="time (ms)", ylabel="audio", xlim=(0, 4))
axes[1].legend(loc="upper right", fontsize=8, frameon=False, facecolor=BACKGROUND)
fig.tight_layout()

# %%
# **Ships: the AIS.** The automatic identification system (ITU-R M.1371-5) sends
# 256-bit packets of GMSK at 9600 bit/s on two VHF channels, 161.975 and 162.025 MHz.
# :func:`comnum.ais_transmit` builds the packets of three position reports (the
# training sequence, the HDLC frame, NRZI), here captured at 2 MS/s 25 kHz below the
# channel at 10 dB of SNR; the receiver is :func:`comnum.select_channel` to 153.6 kHz
# (16 samples per bit) and :func:`comnum.ais_receive`: the polar discriminator, a bit
# clock from the zero crossings, the frames whose check sequence is correct, and the
# fields of the messages, with their ``!AIVDM`` sentence.
fs_ais = comnum.ais_front_end_rate()
reports = [
    comnum.ais_position_report(
        200000000 + k,
        lat_deg=46.1234,
        lon_deg=-9.8765,
        sog_knots=8.4,
        cog_deg=271.5,
        heading_deg=270,
        timestamp=42,
    )
    for k in range(3)
]
packets = comnum.resample_rational(comnum.ais_transmit(reports, fs_ais), fs_ais, 2e6, 7e3)
n = np.arange(len(packets))
capture = packets * np.exp(2j * np.pi * 25e3 * n / 2e6)
capture = capture + np.sqrt(2e6 / 25e3 / 10 / 2) * (
    gen.standard_normal(len(n)) + 1j * gen.standard_normal(len(n))
)
y = comnum.select_channel(capture, 2e6, 25e3, fs_ais, 12.5e3)
for p in comnum.ais_receive(y, fs_ais):
    m = p["message"]
    print(
        f"{p['start_bit'] / 9600:6.3f} s  {p['aivdm'][0]}  MMSI {m['mmsi']}, "
        f"{m['position']['lat_deg']:.4f} N {-m['position']['lon_deg']:.4f} W, {m['position']['sog_knots']} kn"
    )

fig, ax = plt.subplots(figsize=(9, 2.8))
f_inst = comnum.fm_discriminator(y, fs_ais)
t_ais = np.arange(len(f_inst)) / fs_ais * 1e3
ax.plot(t_ais, f_inst / 1e3, lw=0.5)
ax.set(xlabel="time (ms)", ylabel="instantaneous frequency (kHz)", xlim=(30, 65), ylim=(-6, 6))
ax.annotate("training sequence", (36, 4.2), fontsize=8)
ax.annotate("flag, message, frame check sequence, flag", (41, -5.2), fontsize=8)
fig.tight_layout()

# %%
# **A real recording.** The same calls on the files of a software radio. A HackRF One
# writes raw 8-bit samples, which become a SigMF recording with a metadata file
# (``hackrf_transfer -r capture.sigmf-data -f 775000000 -s 15360000``, and
# ``capture.sigmf-meta`` with ``core:datatype`` ``ci8``, ``core:sample_rate`` and the
# ``core:frequency`` of its capture):
#
# .. code-block:: python
#
#    x, fs, meta = comnum.read_sigmf("capture", channel=0)
#    y = comnum.select_channel(x, fs, 770.5e6 - 775e6, 1.92e6, 0.6e6)
#    cell = comnum.lte_cell_search(y, max_cfo_hz=25e3)[0]
#    frames = comnum.lte_decode_mib(y, cell)
#
# On one second of a 20 MHz cell of band 28 recorded indoors with the telescopic
# antenna, the search gives the identity and an offset of −13.06 kHz (the oscillator of
# the HackRF 17 ppm fast), and the MIB of the 100 frames decodes with its CRC. The FM
# receiver needs a carrier about 10 dB above the noise in its 200 kHz, the threshold
# of FM; the user guide (:doc:`recordings <../user_guide/recordings>`) gives the
# commands and what the captures showed.
