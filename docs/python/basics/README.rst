Learn the basics
================

A short course in digital communications, written as programs. Each chapter is a
script that runs as is, adds one idea to the chapter before it, and ends by checking
what it has built against the theory. The thread through all of them is one link: a
text message sent from a transmitter to a receiver.

The chapters assume the mathematics of a first course in signals (complex numbers,
the Gaussian distribution, the Fourier transform) and give the formula and the reason
for it, not the proof: the references at the end of each chapter have the proofs.

0. **Quickstart** — a whole link in twenty lines: a text, its bits, a modulation,
   noise, decisions, the text back, and the error rate against its closed form. Read
   it first; the next chapters take it apart.
1. **Bits and symbols** — from a text to bits and from bits to points of the complex
   plane: constellations, their energy, their labels, and what is traded when a
   symbol carries more bits.
2. **The noisy channel** — additive white Gaussian noise, the three ways of writing a
   signal-to-noise ratio, and how to draw the noise for a given :math:`E_b/N_0`.
3. **Decisions and error rates** — the nearest point as the best decision, the error
   probability of BPSK from a picture, the difference between bit and symbol errors,
   and how many errors a simulation must count before its result means something.

4. **Pulse shaping and the matched filter** — why a symbol needs a pulse, Nyquist's
   criterion, the root-raised cosine at both ends of the link, the eye diagram, and
   what a receiver loses when it samples without filtering.
5. **Synchronisation** — a receiver that knows neither the sampling instant, nor the
   carrier, nor where the frame starts: each recovered in turn, and the threshold
   below which the recovery fails.
6. **Multipath and equalisation** — echoes as a filter, the interference that power
   cannot overcome, zero forcing and its noise enhancement, and an equaliser that
   learns the channel from a training sequence.

*In preparation:* channel coding; from the simulation to a loudspeaker and a
microphone; campaigns and specifications.

Where to go next: the :doc:`tutorials <../auto_tutorials/index>` teach the library
itself (chains, campaigns, profiling, measured signals), the :doc:`gallery
<../auto_examples/index>` has the classical curves of the textbooks, and the
:doc:`user guide <../user_guide/index>` is the reference.
