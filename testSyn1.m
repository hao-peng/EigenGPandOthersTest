function testSyn1
addpath('ARDEigenGP');
addpath('CompositeEigenGP');
addpath('GPML');
startup;
addpath('lightspeed');
addpath('SPGP_dist');
addpath('ssgpr_code');

set(gcf,'defaultlinelinewidth',1.5);
load('syn1/syn1.mat');
[N,D] = size(x);
numTest = 10;


%% number of pseudo-inputs
M = 5;

%% different range to compute nmse
%range = 1:277; %[-3 9]
%range = 48:231; %[-1 7]
range = 71:208; %[0 6]

%% initialize hyperparameters 
opt.cov(1:D) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
opt.cov(D+1) = log(var(y,1)); % log size 
opt.lik = log(var(y,1)/4); % log noise
opt.nIter = 50;

%% fullGP
hyp1 = minimize(opt, @gp, 100, @infExact, [], {@covSEard}, @likGauss, x, y);
[mu_full, s2_full] = gp(hyp1, @infExact, [], {@covSEard}, @likGauss, x, y, xtest);


%% randommize initial values
seed = 0;
rand('seed',seed); randn('seed',seed);

for tid = 1:numTest
opts{tid} = opt;
opts{tid}.cov = opt.cov.*rand(1,D+1);
opts{tid}.logA1 = 0.1*rand();
opts{tid}.logA2 = 0.1*rand();
end

%% Composite EigenGP
seed = 1;
rand('seed',seed); randn('seed',seed);
for tid = 1:numTest
model.logSigma = opts{tid}.lik;
model.logEta = -opts{tid}.cov(1:D,1);
model.logA0 = opts{tid}.cov(D+1);
model.logA1 = opts{tid}.logA1;
model.logA2 = opts{tid}.logA2;

trained_model = EigenGPNS_train(model, x, y, M, 50);
[mu s2] = EigenGPNS_pred(trained_model, x, y, xtest);

nmse_compositeEigenGP(tid) = getNMSE(mu(range), mu_full(range));
plotResult(x, y, xtest, mu_full, mu, s2, trained_model.B);
filename = strcat('syn1/figs/syn1_CompositeEigenGP_M', int2str(M),  '_', int2str(tid),'.pdf');
saveas(gcf, filename, 'pdf');
end

%% ARD update kerB
seed = 0;
rand('seed',seed); randn('seed',seed);
for tid = 1:numTest
[nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, mu_full, M, opts{tid});
nmse_kerBEigenGP(tid) = getNMSE(mu(range), mu_full(range));
plotResult(x, y, xtest, mu_full, mu, s2, post.opt.B);
filename = strcat('syn1/figs/syn1_kerBEigenGP_M', int2str(M),  '_', int2str(tid),'.pdf');
saveas(gcf, filename, 'pdf');
end

%% FITC
seed = 0;
rand('seed',seed); randn('seed',seed);
for tid =1:numTest
hyp_init(1:D,1) = opts{tid}.cov(1:D,1); % log 1/(lengthscales)^2
hyp_init(D+1,1) = opts{tid}.cov(D+1); % log size 
hyp_init(D+2,1) = opts{tid}.lik; % log noise
% random initialize pseudo-inputs
[dum,I] = sort(rand(N,1)); clear dum;
I = I(1:M);
xb_init = x(I,:);
w_init = [reshape(xb_init,M*D,1);hyp_init];
% optimization
[w,f] = minimize(w_init,'spgp_lik',-200,y,x,M);
% [w,f] = lbfgs(w_init,'spgp_lik',200,10,y0,x,M); % an alternative
xb = reshape(w(1:M*D,1),M,D);
hyp = w(M*D+1:end,1);
% PREDICTION
[mu,s2] = spgp_pred(y,x,xb,xtest,hyp);
% if you want predictive variances to include noise variance add noise:
s2 = s2 + exp(hyp(end));

nmse_fitc(tid) = getNMSE(mu(range), mu_full(range));
plotResult(x, y, xtest, mu_full, mu, s2, xb);
filename = strcat('syn1/figs/syn1_fitc_M', int2str(M),  '_', int2str(tid),'.pdf');
saveas(gcf, filename, 'pdf');
end

%% SSGPR
seed = 0;
rand('seed',seed); randn('seed',seed);
for tid = 1:numTest
hyp_init(1:D,1) = opts{tid}.cov(1:D,1); % log 1/(lengthscales)^2
hyp_init(D+1,1) = opts{tid}.cov(D+1); % log size 
hyp_init(D+2,1) = opts{tid}.lik; % log noise
[nmse, mu, s2, nmlp, newloghyper, convergence] = ssgpr_ui(x, y, xtest, mu_full, M, 1000, hyp_init);

nmse_ssgpr(tid) = getNMSE(mu(range), mu_full(range));
plotResult(x, y, xtest, mu_full, mu, s2);
filename = strcat('syn1/figs/syn1_ssgpr_M', int2str(M),  '_', int2str(tid),'.pdf');
saveas(gcf, filename, 'pdf');
end

%% print result
fprintf('composite EigenGP: %f +- %f\n', mean(nmse_compositeEigenGP), std(nmse_compositeEigenGP)/sqrt(numTest));
fprintf('kerB EigenGP: %f +- %f\n', mean(nmse_kerBEigenGP), std(nmse_kerBEigenGP)/sqrt(numTest));
fprintf('FITC: %f +- %f\n', mean(nmse_fitc), std(nmse_fitc)/sqrt(numTest));
fprintf('SSGPR: %f +- %f\n', mean(nmse_ssgpr), std(nmse_ssgpr)/sqrt(numTest));
end

function plotResult(x, y, xs, ys, mu, s2, B)
clf
hold on
plot(x,y,'.m', 'MarkerSize', 12)% data points in magenta
plot(xs, ys, '-', 'Color', [0 .5 0]);
plot(xs,mu,'b') % mean predictions in blue
plot(xs,mu,'b') % mean predictions in blue
plot(xs,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation predictions in red
plot(xs,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
if nargin > 6
    plot(B,-2.75*ones(size(B)),'k+','markersize',20)
end
hold off
axis([-3 9 -3 2]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')
end

function nmse = getNMSE(mu, ys)
nmse = mean((mu-ys).^2)/mean((mean(mu)-ys).^2);
end