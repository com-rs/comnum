% The DPSK link of Optilux's example 15 (ex15_dtc_dpsk_ber.m: binary DPSK in additive
% white Gaussian noise, a moving-average optical filter and a delay interferometer),
% run by tools/record_optilux.py under Octave with a tighter stopping rule than the
% example's (relative accuracy 0.1 at 95 % confidence):
%
%   octave --no-gui --quiet tools/record_optilux_dpsk.m <optilux dir> <out.csv>
%
% The chain, with the functions of Optilux: LASERSOURCE at 0 dBm (one polarisation),
% a random binary pattern (seed 1), DIFFENCODER, DIGITALMOD with the 'costails' pulse of
% roll-off 0.01 at 32 samples per symbol (a rectangle of one symbol at this roll-off),
% 10 GBd, MZMODULATOR; per run complex Gaussian noise of variance 10^(-SNR/10) fs/R per
% sample (the SNR over the symbol rate); RXFRONTEND with the 'movavg' optical filter
% of bandwidth 2 R (the integrator over one symbol, H(f) = sinc(f/R)), the one-symbol
% delay interferometer and the balanced photodiode, RXDSP with data-aided timing,
% SAMP2PAT; the bit error rate of bits 2 to 1024 (the example's count, first bit
% rejected) estimated by MEANESTIMATE until 3 % at 95 % confidence or 10^7 bits.
%
% Two Octave departures from MATLAB are undone, both on comparisons of complex
% arrays (Octave compares the moduli, MATLAB the real parts): the differential
% encoder is DIFFENCODER's own computation followed by the real part, and the samples
% of RXDSP (real values with a zero imaginary part) are given to SAMP2PAT as real.
%
% The CSV holds one row per SNR: SNR (dB), the mean BER of the example's count, its
% standard deviation, the number of bits, the errors and bits of bits 3 to 1024 (the
% second bit is not chained to the first by the encoder, see the card), and the
% example's theory 0.5 exp(-SNR).

args = argv(); optilux_dir = args{1}; out_csv = args{2};
addpath(optilux_dir); pkg load signal; pkg load control;

Nsymb = 1024; Nt = 32; symbrate = 10;
tx.rolloff = 0.01; tx.emph = 'asin'; modfor = 'dpsk'; PdBm = 0; lam = 1550;
SNRdB = 3:1:8;
rx.modformat = modfor; rx.sync.type = 'da'; rx.oftype = 'movavg'; rx.obw = 2;
rx.eftype = 'gauss'; rx.ebw = Inf; rx.type = 'bin';
mc.stop = [0.03 95]; mc.maxsamp = 1e7;

Nsamp = Nsymb * Nt; fs = symbrate * Nt; inigstate(Nsamp, fs);
rows = zeros(length(SNRdB), 7);

Plin = 10 .^ (PdBm / 10);
E = lasersource(Plin, lam, struct('pol', 'single'));
rng(1);
[patx, patbinx] = pattern(Nsymb, 'rand', struct('format', modfor));
stars = pat2samp(patbinx, modfor);
stars(2:end) = exp(filter(1, [1 1], log(stars(2:end))));
patbinex = samp2pat(real(stars), modfor);
[sigx, normx] = digitalmod(patbinex, modfor, symbrate, 'costails', tx);
E = mzmodulator(E, sigx, struct('norm', normx));

Er = E;
for k = 1:length(SNRdB)
  sigma2 = 10 ^ (-SNRdB(k) / 10) * fs / symbrate;
  cond = true; err3 = 0; bits3 = 0;
  while cond
    noise = sqrt(sigma2 / 2) * (randn(Nsamp, 1) + 1i * randn(Nsamp, 1));
    Er.field = E.field + noise;
    rsigx = rxfrontend(Er, lam, symbrate, rx);
    akxhat = rxdsp(rsigx, symbrate, patbinx, rx);
    patbinxhat = samp2pat(real(akxhat), modfor, rx);
    [cond, avgber, nruns, stdber] = meanestimate(patbinx(2:end, :) ~= patbinxhat(2:end, :), mc);
    err3 = err3 + sum(patbinx(3:end, :) ~= patbinxhat(3:end, :)); bits3 = bits3 + Nsymb - 2;
  end
  rows(k, :) = [SNRdB(k), avgber, stdber, nruns, err3, bits3, 0.5 * exp(-10 ^ (SNRdB(k) / 10))];
  printf("SNR %g dB: BER %.4e (%d bits), bits 3..: %d errors in %d\n", SNRdB(k), avgber, nruns, err3, bits3);
end
csvwrite(out_csv, rows);
printf("done\n");
