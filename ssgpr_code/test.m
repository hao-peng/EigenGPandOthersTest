rng(0);
disp('[m s2] = EigenGP(hyp, meanfunc, covfunc, likfunc, x, y, z);;')
%hyp = minimize(hyp, @gp, -100, @(hyp, mean, cov, lik, x, y)infEigenGP(hyp, mean, cov, lik, x, y, par), [], covfunc, likfunc, X_tr, T_tr);
%[NMSE, m6, s2, NMLP, loghyper, convergence] = ssgpr_ui(x, y, xall(21:end), yall(21:end), 100);
[NMSE, m6, s2, NMLP, loghyper, convergence] = ssgpr_ui(x, y, z, ones(size(z)), 100);

figure(6)
set(gca, 'FontSize', 24)
disp(' ')
disp('f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];') 
f = [m6+2*sqrt(s2); flipdim(m6-2*sqrt(s2),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8);')
fill([z; flipdim(z,1)], f, [7 7 7]/8);
disp('hold on; plot(z, m); plot(x, y, ''+'')')
hold on; plot(z, m6, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
axis([-4 4 -3 3])
grid on
xlabel('input, x')
ylabel('output, y')
if write_fig, print -depsc f2.eps; end