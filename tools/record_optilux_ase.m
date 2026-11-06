% The amplified link of Optilux's example 21 (ex21_opt_modulation_ase_ber.m: a
% polarisation-multiplexed QPSK signal through 50 spans of 150 km of lossy fibre, each
% followed by a noisy amplifier), run by tools/record_optilux.py under Octave with a
% tighter stopping rule than the example's (the example stops at 20 000 samples):
%
%   octave --no-gui --quiet tools/record_optilux_ase.m <optilux dir> <out.csv>
%
% The chain, with the functions of Optilux: LASERSOURCE at 0 dBm, PBS into two
% polarisations of equal power, two random QPSK patterns (seed 1), DIGITALMOD with
% root-raised-cosine pulses of roll-off 0.01 at 32 samples per symbol, 10 GBd,
% IQMODULATOR, PBC; per span FIBER with alpha 0.2 dB/km over 150 km, no dispersion, no
% Kerr effect, and AMPLIFLAT with the gain 30 dB and the noise figure 6 dB; RXFRONTEND
% (no optical filter, the root-raised-cosine electrical filter of bandwidth 0.5 symbol
% rate), RXDSP with data-aided timing, SAMP2PAT, SAMP2SNR; the bit error rate estimated
% by MEANESTIMATE until a relative accuracy of 3 % at 95 % confidence or 400 000 bits.
% The CSV holds one row per span: span, mean BER, its standard deviation, the number of
% bits, the SNR estimated by SAMP2SNR (dB), the SNR of the example's formula (dB) and
% the BER of that SNR (0.5 erfc(sqrt(SNR/2)), the example's theory for QPSK).

args = argv(); optilux_dir = args{1}; out_csv = args{2};
addpath(optilux_dir); pkg load signal; pkg load control;
global CONSTANTS

Nsymb = 1024; Nt = 32; symbrate = 10;
tx.rolloff = 0.01; tx.emph = 'asin'; modfor = 'qpsk'; PdBm = 0; lam = 1550;
Nspan = 50;
ft.length = 150E3; ft.lambda = 1550; ft.alphadB = 0.2; ft.disp = 0; ft.slope = 0;
ft.pmdpar = 0; ft.aeff = 80; ft.n2 = 0;
amp.gain = ft.length * ft.alphadB * 1e-3; amp.f = 6;
rx.modformat = modfor; rx.sync.type = 'da'; rx.oftype = 'gauss'; rx.obw = Inf;
rx.eftype = 'rootrc'; rx.ebw = 0.5; rx.epar = tx.rolloff; rx.type = 'bin';
mc.stop = [0.03 95]; mc.maxsamp = 4e5;

Nsamp = Nsymb * Nt; fs = symbrate * Nt; inigstate(Nsamp, fs);
avgber = zeros(1, Nspan); stdber = zeros(1, Nspan); snr_hat = zeros(1, Nspan); nbits = zeros(1, Nspan);

Plin = 10 .^ (PdBm / 10);
E = lasersource(Plin, lam);
[Ex, Ey] = pbs(E);
rng(1);
[patx, patbinx] = pattern(Nsymb, 'rand', struct('format', modfor));
[paty, patbiny] = pattern(Nsymb, 'rand', struct('format', modfor));
[elecx, normx] = digitalmod(patx, modfor, symbrate, 'rootrc', tx);
[elecy, normy] = digitalmod(paty, modfor, symbrate, 'rootrc', tx);
Ex = iqmodulator(Ex, elecx, struct('norm', normx));
Ey = iqmodulator(Ey, elecy, struct('norm', normy));
Esp = pbc(Ex, Ey);
patbin = [patbinx patbiny];
for ns = 1:Nspan
  cond = true;
  while cond
    E = Esp;
    E = fiber(E, ft);
    E = ampliflat(E, amp);
    rsig = rxfrontend(E, lam, symbrate, rx);
    akhat = rxdsp(rsig, symbrate, [patx paty], rx);
    patbinhat = samp2pat(akhat, modfor, rx);
    snr_hat(ns) = samp2snr(patbin, akhat, modfor);
    [cond, avgber(ns), nruns, stdber(ns)] = meanestimate(patbin ~= patbinhat, mc);
    nbits(ns) = nruns;
  end
  Esp = E;
end
CLIGHT = CONSTANTS.CLIGHT; HPLANCK = CONSTANTS.HPLANCK;
fac58 = 10 * log10(HPLANCK * CLIGHT / E.lambda * 1e9 * 12.5e9) + 30;
n0b = fac58 + amp.gain + amp.f + 10 * log10(1:Nspan) + 10 * log10(symbrate / 12.5);
snr_theory = PdBm - n0b;
ber_theory = 0.5 * erfc(sqrt(10 .^ (snr_theory / 10) / 2));
csvwrite(out_csv, [(1:Nspan)', avgber', stdber', nbits', snr_hat', snr_theory', ber_theory']);
printf("done\n");
