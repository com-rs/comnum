% The dispersion-managed link of Optilux's example 19 (ex19_opt_dm_link.m: on-off
% keying through five spans of standard fibre and dispersion-compensating fibre,
% Manakov equation), run by tools/record_optilux.py under Octave on a shorter block
% (64 symbols instead of 1024, the step rule unchanged: it depends on the sampling
% rate, not on the block) and without the plots:
%
%   octave --no-gui --quiet tools/record_optilux_dm.m <optilux dir> <out.csv> [asymm|symm]
%
% The step type of FIBER is the example's default, 'asymm' (the nonlinear step then the
% linear step), or 'symm' (the symmetric step), the third argument.
%
% The chain, with the functions of Optilux: LASERSOURCE at 3 dBm, a random OOK pattern
% (seed 1), DIGITALMOD with the 'costails' pulse of roll-off 0.2 at 32 samples per
% symbol, 10 GBd, IQMODULATOR; per span FIBER over 100 km of transmission fibre
% (0.2 dB/km, 17 ps/(nm km), slope 0.057 ps/(nm^2 km), 80 um^2, n2 2.5e-20 m^2/W,
% Manakov, no PMD, dzmax 20 km: the default step rule, the first step bounded by the
% four-wave-mixing phase 20 x 1.5^2 rad over the sampling bandwidth and the constant
% local error after it), FIBER over 1 km of compensating fibre (0.6 dB/km, the
% dispersion that leaves no residual, slope 0.057, 20 um^2: one step, its dzmax being
% beyond its length), AMPLIFLAT with the gain of both losses (20.6 dB, no noise).
% The CSV holds the samples, one row each: the input field x and y, the field after
% the first span and after the five (real and imaginary parts, sqrt(mW), Optilux's
% sign convention); its last row holds the numbers of steps of the two fibres.

args = argv(); optilux_dir = args{1}; out_csv = args{2};
steptype = 'asymm'; if numel(args) > 2, steptype = args{3}; end
addpath(optilux_dir); pkg load signal; pkg load control;

Nsymb = 64; Nt = 32; symbrate = 10;
tx.rolloff = 0.2; tx.emph = 'asin'; modfor = 'ook'; PdBm = 3; lam = 1550;
RDPS = 0; Nspan = 5;
ft.length = 100E3; ft.lambda = 1550; ft.alphadB = 0.2; ft.disp = 17; ft.slope = 0.057;
ft.pmdpar = 0; ft.ismanakov = true; ft.aeff = 80; ft.n2 = 2.5e-20; ft.dzmax = 2E4;
ft.trace = false; ft.steptype = steptype;
fc = ft; fc.length = 1e3; fc.alphadB = 0.6;
fc.disp = (RDPS * 1e3 - ft.disp * ft.length) / fc.length; fc.slope = 0.057;
fc.pmdpar = 0; fc.aeff = 20; fc.dzmax = 2E4;
amp.gain = (ft.length * ft.alphadB + fc.length * fc.alphadB) * 1e-3;

Nsamp = Nsymb * Nt; fs = symbrate * Nt; inigstate(Nsamp, fs);
Plin = 10 .^ (PdBm / 10);
E = lasersource(Plin, lam);
rng(1);
[patx, patbinx] = pattern(Nsymb, 'rand', struct('format', modfor));
[sigx, normx] = digitalmod(patbinx, modfor, symbrate, 'costails', tx);
E = iqmodulator(E, sigx, struct('norm', normx));
u0 = E.field;
for ns = 1:Nspan
  [E, parft] = fiber(E, ft);
  [E, parfc] = fiber(E, fc);
  E = ampliflat(E, amp);
  if ns == 1
    u1 = E.field;
  end
end
u5 = E.field;
out = [real(u0(:, 1)) imag(u0(:, 1)) real(u0(:, 2)) imag(u0(:, 2)) ...
       real(u1(:, 1)) imag(u1(:, 1)) real(u1(:, 2)) imag(u1(:, 2)) ...
       real(u5(:, 1)) imag(u5(:, 1)) real(u5(:, 2)) imag(u5(:, 2))];
out(end + 1, :) = [parft.ncycle parfc.ncycle parft.firstdz parfc.firstdz zeros(1, 8)];
dlmwrite(out_csv, out, 'delimiter', ',', 'precision', '%.17g');
printf("steps %d %d, first %.4f %.4f m\ndone\n", parft.ncycle, parfc.ncycle, parft.firstdz, parfc.firstdz);
