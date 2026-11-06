% The split-step of Optilux (GPL-3.0, Paolo Serena, University of Parma, Apr 2021) on
% a field written by tools/record_optilux.py, which calls this script under Octave:
%
%   octave --no-gui --quiet tools/record_optilux.m <optilux dir> <in.csv> <out.csv> \
%          <n spans> <gamma scale> <fs GHz>
%
% The link of benchmarks/libraries/s5_ssfm.py: 80 km spans, alpha 0.2 dB/km, D 17
% ps/nm/km, beta3 = 0 (the slope -2 D / lambda of the help of FIBER), gamma 1.3 /W/km
% times <gamma scale> (0: linear), 193.1 THz, sampled at <fs GHz>; a fixed symmetric
% step of 1 km (dzmax, the nonlinear-phase rule with an unreachable bound); a
% noiseless amplification of alpha L at the end of every span (AMPLIFLAT). The field
% of Optilux is in sqrt(mW): the input in sqrt(W) is scaled by sqrt(1000) and the
% output scaled back. Optilux writes the nonlinear Schroedinger equation with the
% opposite sign of the imaginary unit (its nonlinear step is exp(-i phi)): the driver
% propagates the conjugate of the field and conjugates the result (as the benchmark
% does for Sionna). The CSV files hold the real and the imaginary parts of every
% polarisation: two columns for one polarisation (the scalar NLSE), four for two
% (x then y: the Manakov equation of the unique-field mode, ismanakov true, without
% PMD and without birefringence: pmdpar 0 makes the one waveplate the identity).

args = argv();
optilux_dir = args{1}; in_csv = args{2}; out_csv = args{3};
nspans = str2double(args{4}); gscale = str2double(args{5}); fs = str2double(args{6});
addpath(optilux_dir);

fc = 193.1e12; clight = 299792458;
lam = clight / fc * 1e9;                              % nm
alphadB = 0.2; disp_ps = 17; gamma_w_km = 1.3;        % dB/km, ps/nm/km, /W/km
aeff = 80;                                            % um^2
% FIBER: gam [1/mW/m] = 2 pi n2 / (lambda aeff) 1e18; gamma 1.3 /W/km = 1.3e-6 /mW/m
n2 = gscale * gamma_w_km * 1e-6 * lam * aeff / (2 * pi) * 1e-18;

xin = csvread(in_csv);
npol = size(xin, 2) / 2;
u = zeros(size(xin, 1), npol);
for p = 1:npol
  u(:, p) = (xin(:, 2 * p - 1) + 1i * xin(:, 2 * p)) * sqrt(1000);
end
inigstate(size(u, 1), fs);
E.lambda = lam;
E.field = u;                                          % one column per polarisation

ft.length = 80e3; ft.lambda = lam; ft.alphadB = alphadB; ft.disp = disp_ps;
ft.slope = -2 * disp_ps / lam;                        % beta3 = 0
ft.n2 = n2; ft.aeff = aeff;
ft.steptype = 'symm'; ft.dzmax = 1000;
ft.dphi1fwm = false; ft.stepupd = 'nlp'; ft.dphimax = 1e9;  % every step is dzmax
ft.trace = false;
if npol == 2
  ft.ismanakov = true;                                % the Manakov equation, 8/9 gamma
  ft.pmdpar = 0;                                      % no PMD: one waveplate, the identity
end
amp.gain = alphadB * ft.length * 1e-3;                % dB, noiseless

steps = zeros(1, nspans);
for span = 1:nspans
  [E, out] = fiber(E, ft);
  steps(span) = out.ncycle;
  E = ampliflat(E, amp);
end
y = E.field / sqrt(1000);
cols = zeros(size(y, 1), 2 * npol);
for p = 1:npol
  cols(:, 2 * p - 1) = real(y(:, p));
  cols(:, 2 * p) = imag(y(:, p));
end
csvwrite(out_csv, cols);
printf("%s\n", mat2str(steps));
