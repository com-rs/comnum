% The WDM back-to-back link of Optilux's example 16 (ex16_opt_wdm_save_and_print.m:
% five channels of polarisation-multiplexed 16-QAM generated, multiplexed and
% detected channel by channel), run by tools/record_optilux.py under Octave without
% the example's saving and printing of the host:
%
%   octave --no-gui --quiet tools/record_optilux_wdm.m <optilux dir> <out.csv>
%
% The chain, with the functions of Optilux: per channel k = 1..5 the seed
% 1e5 + k - 3, LASERSOURCE at 3 dBm on the grid GETLAMBDA (37.5 GHz around 1550 nm),
% PBS, two random 16-QAM patterns, DIGITALMOD with root-raised-cosine pulses of
% roll-off 0.1 at 32 samples per symbol, 32 GBd, IQMODULATOR, PBC, MULTIPLEXER around
% the central channel; per channel RXFRONTEND (no optical filter, the root-raised-
% cosine electrical filter of bandwidth 0.5 symbol rate), RXDSP with data-aided timing,
% SAMP2SNR. The CSV holds one row per channel: its index and the SNR (dB) of
% SAMP2SNR, which the example notes is finite only because of the numerical noise.

args = argv(); optilux_dir = args{1}; out_csv = args{2};
addpath(optilux_dir); pkg load signal; pkg load control;

Nsymb = 1024; Nt = 32; Nch = 5; Ncut = ceil(Nch / 2); symbrate = 32;
tx.rolloff = 0.1; tx.emph = 'asin'; modfor = '16qam'; PdBm = 3; lam = 1550; deltaf = 37.5;
rx.modformat = modfor; rx.sync.type = 'da'; rx.oftype = 'gauss'; rx.obw = Inf;
rx.eftype = 'rootrc'; rx.ebw = 0.5; rx.epar = tx.rolloff; rx.type = 'bin';

spac = deltaf * lam ^ 2 / 299792458; Plin = 10 .^ (PdBm / 10);
Nsamp = Nsymb * Nt; fs = symbrate * Nt; inigstate(Nsamp, fs);
lamv = getlambda(lam, spac, Nch);
for k = 1:Nch
  rng(1e5 + k - ceil(Nch / 2));
  E = lasersource(Plin, lamv(k));
  [Ex, Ey] = pbs(E);
  patx(:, k) = pattern(Nsymb, 'rand', struct('format', modfor));
  paty(:, k) = pattern(Nsymb, 'rand', struct('format', modfor));
  [elecx, normx] = digitalmod(patx(:, k), modfor, symbrate, 'rootrc', tx);
  [elecy, normy] = digitalmod(paty(:, k), modfor, symbrate, 'rootrc', tx);
  Ex = iqmodulator(Ex, elecx, struct('norm', normx));
  Ey = iqmodulator(Ey, elecy, struct('norm', normy));
  E = pbc(Ex, Ey);
  if k == 1
    Em = E;
  else
    Em = multiplexer(Em, E, struct('lambda', lamv(Ncut)));
  end
end
snr = zeros(Nch, 1);
for k = 1:Nch
  rsig = rxfrontend(Em, lamv(k), symbrate, rx);
  akhat = rxdsp(rsig, symbrate, [patx(:, k) paty(:, k)], rx);
  snr(k) = samp2snr([patx(:, k) paty(:, k)], akhat, modfor);
end
csvwrite(out_csv, [(1:Nch)', snr]);
printf("SNR %s dB\ndone\n", num2str(snr', '%.2f '));
