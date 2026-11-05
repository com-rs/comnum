Recordings
==========

Measured captures enter a link as recordings. :func:`comnum.read_sigmf` reads a
`SigMF <https://sigmf.org>`_ recording (the metadata ``.sigmf-meta`` and the samples
``.sigmf-data``) and returns the samples, the sampling rate and the metadata;
:func:`comnum.write_sigmf` writes one, in any of the SigMF sample formats:

.. code-block:: python

   import numpy as np
   import comnum

   x = np.exp(2j * np.pi * 0.01 * np.arange(1024))
   comnum.write_sigmf("capture", x, 1e6, "ci16_le", {"core:description": "a tone"})
   y, fs, meta = comnum.read_sigmf("capture")

In a specification, the block ``capture`` replaces the transmitter and the channel by
a recording (SigMF, raw IQ files or NPY arrays) and aligns the reference sequence on
it, so that the receiver DSP and the metrics run unchanged on measured data.

A software radio records a band wider than the channel of interest.
:func:`comnum.select_channel` is the front end of its receiver: it removes the DC
offset of the converter, shifts the channel to 0 Hz and resamples it by the exact
ratio of the two rates, under a low-pass filter whose window rejects the neighbouring
channels by 80 dB.

An LTE cell from a HackRF
-------------------------

The six central resource blocks of an LTE carrier hold the synchronisation signals and
the broadcast channel in every bandwidth. At 1.92 MS/s, :func:`comnum.lte_cell_search`
finds the PSS of each identity over a grid of frequency offsets (an oscillator 20 ppm
off is 15 kHz at 770 MHz), refines the offset on the cyclic prefixes, and reads the
SSS to get the cell identity and the start of the frames; :func:`comnum.lte_decode_mib`
then decodes the master information block of every frame (3GPP TS 36.211 Sec. 6.6,
6.10.1, 6.11; TS 36.212 Sec. 5.3.1), following the drift of the sampling clock on the
PSS:

.. code-block:: python

   # hackrf_transfer -r capture.sigmf-data -f 775000000 -s 15360000 -l 32 -g 34 -a 1 -n 15360000
   # and capture.sigmf-meta: {"global": {"core:datatype": "ci8", "core:sample_rate": 15360000,
   #   "core:version": "1.2.0"}, "captures": [{"core:sample_start": 0,
   #   "core:frequency": 775000000}], "annotations": []}
   import comnum

   x, fs, meta = comnum.read_sigmf("capture", channel=0)
   y = comnum.select_channel(x, fs, 770.5e6 - 775e6, 1.92e6, 0.6e6)
   for cell in comnum.lte_cell_search(y, max_cfo_hz=25e3):
       frames = comnum.lte_decode_mib(y, cell)
       print(cell["n_cell_id"], cell["cfo_hz"], [(f["sfn"], f["ports"], f["n_rb"]) for f in frames][:4])

On one second of a 20 MHz cell of band 28 captured by a HackRF One with its telescopic
antenna, the cell search gives the identity, an offset of -13.06 kHz (the oscillator of
the HackRF 17.0 ppm fast), and the MIB of the 100 frames decodes with its CRC: two
antenna ports, 100 resource blocks, frame numbers that follow one another; the 19 200.32
samples counted per frame put the same oscillator 16.8 ppm fast. The Rust example
``lte_mib`` (``cargo run --release -p comnum --features spec --example lte_mib --
capture.sigmf-meta 770.5e6``) prints the same.

An FM station
-------------

An FM station occupies about 200 kHz: :func:`comnum.select_channel` brings it to
228 kHz (12 times the 19 kHz pilot, 192 samples per RDS bit),
:func:`comnum.fm_discriminator` gives the multiplex in hertz of deviation,
:func:`comnum.fm_stereo_decode` the left and right channels (ITU-R BS.450-4, pilot-tone
system, de-emphasis of 50 µs) and :func:`comnum.rds_receive` the data of the 57 kHz
subcarrier: the programme identification, the programme service name, the radiotext and
the clock time (NRSC-4 / EN 50067):

.. code-block:: python

   # hackrf_transfer -r fm.sigmf-data -f 95600000 -s 2400000 -n 24000000 -l 32 -g 30 -a 1
   x, fs, meta = comnum.read_sigmf("fm", channel=0)
   y = comnum.select_channel(x, fs, 95.9e6 - 95.6e6, 228e3, 110e3)
   mpx = comnum.fm_discriminator(y, 228e3)
   audio = comnum.fm_stereo_decode(mpx - mpx.mean(), 228e3)     # 48 kHz, left and right
   rds = comnum.rds_receive(mpx, 228e3)
   print(audio["stereo"], hex(rds["pi"] or 0), rds["ps"], rds["radiotext"], rds["clock"])

Demodulation needs a carrier about 10 dB above the noise in its 200 kHz (the threshold
of FM); the telescopic antenna of a HackRF indoors may give less.
