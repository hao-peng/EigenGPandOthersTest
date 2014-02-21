% one-step prediction for synthetic sinc data using curve fitting method
% The input (x) is the time
% The ouput (y) is the value.
% M - number of basis
function testSincCF(M)
addpath('ARDEigenGP');
addpath('CompositeEigenGP');
addpath('GPML');
startup;
addpath('lightspeed');
addpath('SPGP_dist');
addpath('ssgpr_code');
mkdir('sinc/figs');
GENERATE_DATA = false; % whether to generate data
NORMALIZE_DATA = false; % whether to normalize data
TEST_FULL = false; % whether to test full
%% generating data
if GENERATE_DATA
seed = 0;
rand('seed',seed); randn('seed',seed);
N = 500; % number of data points
sigma = 0.08;
raw_x = linspace(0,5,N)';
raw_y = sinc(raw_x)+normrnd(0, sigma, N, 1);
save('sinc/sinc.mat', 'raw_x', 'raw_y');
%scatter(xall, yall);
end

%% initialize arguments if they are not assigned.
if ~exist('M', 'var')
    M = 5;
end

%% prepare data
if ~GENERATE_DATA
load('sinc/sinc.mat');
N = size(raw_x, 1);
end
all_x = raw_x;
all_y = raw_y;

%% normalize the data
n = 50; % starting training data size
ns = N-n; % number of tests

if NORMALIZE_DATA
meanx = mean(all_x(1:n,:));
stdx = std(all_x(1:n,:));
meany = mean(all_y(1:n));
stdy = std(all_y(1:n));
all_x = bsxfun(@rdivide, bsxfun(@minus, all_x, meanx), stdx);
all_y = bsxfun(@rdivide, bsxfun(@minus, all_y, meany), stdy);
end

%% full GP
if TEST_FULL
    seed = 0;
    rand('seed',seed); randn('seed',seed);
    x = all_x(1:n,:);
    y = all_y(1:n);
    xtest = all_x(n+1,:);
    ytest = all_y(n+1);
    if NORMALIZE_DATA
        opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
    else
        opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
    end
    opt.cov(2) = log(var(y,1)); % log size 
    opt.lik = log(var(y,1)/4); % log noise

    hyp1 = minimize(opt, @gp, 100, @infExact, [], {@covSEard}, @likGauss, x, y);
    for tid = 1:ns
        xtest = all_x(n+tid,:);
        ytest = all_y(n+tid);
        hyp1 = minimize(hyp1, @gp, 10, @infExact, [], {@covSEard}, @likGauss, x, y);
        [mu, s2] = gp(hyp1, @infExact, [], {@covSEard}, @likGauss, x, y, xtest);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;
        nses_full(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];
    end
    if NORMALIZE_DATA
        pred_mu = bsxfun(@plus, bsxfun(@times, pred_mu, stdy), meany);
        pred_s2 = bsxfun(@times, pred_s2, stdy^2);
    end
    nmse_full = mean(nses_full);
    plotResult(raw_x, raw_y, raw_x(1+n:n+ns), pred_mu, pred_s2);
    filename = strcat('sinc/figs/sinc_full.pdf');
    saveas(gcf, filename, 'pdf');
end

%% composite eigenGP
seed = 0;
rand('seed',seed); randn('seed',seed);
x = all_x(1:n,:);
y = all_y(1:n);
model.logSigma = log(var(y,1));
if NORMALIZE_DATA
    model.logEta = 2*log((max(x)-min(x))'/2); %log eta
else
    model.logEta = 2*log((max(x)-min(x))'); %log eta
end

model.logA0 = log(var(y,1)/4);
model.logA1 = 0.1;
model.logA2 = 0.1;
trained_model = EigenGPNS_train(model, x, y, M, 100);
for tid = 1:ns
    xtest = all_x(n+tid,:);
    ytest = all_y(n+tid);
    %[b i] = min(trained_model.B);
    %trained_model.B(i) = x(end);
    if tid ~= 1
        trained_model = rmfield(trained_model, 'B');
    end
    trained_model = EigenGPNS_train(trained_model, x, y, M, 20);
    [mu s2] = EigenGPNS_pred(trained_model, x, y, xtest);
    pred_mu(tid) = mu;
    pred_s2(tid) = s2;
    
    nses_compositeEigenGP(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
    x = [x; xtest];
    y = [y; ytest];
    
end
if NORMALIZE_DATA
    pred_mu = bsxfun(@plus, bsxfun(@times, pred_mu, stdy), meany);
    pred_s2 = bsxfun(@times, pred_s2, stdy^2);
end

nmse_compositeEigenGP = mean(nses_compositeEigenGP);
plotResult(raw_x, raw_y, raw_x(1+n:n+ns), pred_mu, pred_s2, trained_model.B);
filename = strcat('sinc/figs/sinc_compositeEigenGP_M', int2str(M), '.pdf');
saveas(gcf, filename, 'pdf');

%% kerB EigenGP
seed = 0;
rand('seed',seed); randn('seed',seed);
x = all_x(1:n,:);
y = all_y(1:n);
xtest = all_x(n+1,:);
ytest = all_y(n+1);
if NORMALIZE_DATA
    opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
else
    opt.cov(1) = -2*log((max(x)-min(x))'); % log 1/(lengthscales)^2
end
opt.cov(2) = log(var(y,1)); % log size 
opt.lik = log(var(y,1)/4); % log noise
opt.nIter = 100;
[nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, ytest, M, opt);
post.opt.nIter = 20;
for tid = 1:ns
    xtest = all_x(n+tid,:);
    ytest = all_y(n+tid);
    [nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, ytest, M, post.opt);
    pred_mu(tid) = mu;
    pred_s2(tid) = s2;
    nses_kerB(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
    x = [x; xtest];
    y = [y; ytest];
end
if NORMALIZE_DATA
    pred_mu = bsxfun(@plus, bsxfun(@times, pred_mu, stdy), meany);
    pred_s2 = bsxfun(@times, pred_s2, stdy^2);
end
nmse_kerB = mean(nses_kerB);
plotResult(raw_x, raw_y, raw_x(1+n:n+ns), pred_mu, pred_s2, post.opt.B);
filename = strcat('sinc/figs/sinc_kerB_M', int2str(M), '.pdf');
saveas(gcf, filename, 'pdf');

%% FITC
seed = 1;
rand('seed',seed); randn('seed',seed);
x = all_x(1:n,:);
y = all_y(1:n);
if NORMALIZE_DATA
    hyp_init(1,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
else
    hyp_init(1,1) = -2*log((max(x)-min(x))'); % log 1/(lengthscales)^2
end
hyp_init(2,1) = log(var(y,1)); % log size 
hyp_init(3,1) = log(var(y,1)/4); % log noise
% random initialize pseudo-inputs
[dum,I] = sort(rand(n,1)); clear dum;
I = I(1:M);
xb_init = x(I,:);
w_init = [reshape(xb_init,M,1);hyp_init];
% optimization
[w,f] = minimize(w_init,'spgp_lik',-100,y,x,M);
for tid = 1:ns
    xtest = all_x(n+tid,:);
    ytest = all_y(n+tid);
    [w,f] = minimize(w,'spgp_lik',-20,y,x,M);
    xb = reshape(w(1:M,1),M,1);
    hyp = w(M+1:end,1);
    % PREDICTION
    [mu,s2] = spgp_pred(y,x,xb,xtest,hyp);
    pred_mu(tid) = mu;
    pred_s2(tid) = s2;
    nses_fitc(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
    x = [x; xtest];
    y = [y; ytest];
end
if NORMALIZE_DATA
    pred_mu = bsxfun(@plus, bsxfun(@times, pred_mu, stdy), meany);
    pred_s2 = bsxfun(@times, pred_s2, stdy^2);
end
nmse_fitc = mean(nses_fitc);
plotResult(raw_x, raw_y, raw_x(1+n:n+ns), pred_mu, pred_s2, xb);
filename = strcat('sinc/figs/sinc_fitc_M', int2str(M), '.pdf');
saveas(gcf, filename, 'pdf');
%% SSGPR
seed = 1;
rand('seed',seed); randn('seed',seed);
x = all_x(1:n,:);
y = all_y(1:n);
xtest = all_x(n+1,:);
ytest = all_y(n+1);
if NORMALIZE_DATA
    hyp_init(1,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
else
    hyp_init(1,1) = -2*log((max(x)-min(x))'); % log 1/(lengthscales)^2
end
hyp_init(2,1) = log(var(y,1)); % log size 
hyp_init(3,1) = log(var(y,1)/4); % log noise
[nmse, mu, s2, nmlp, newhyp, convergence] = ssgpr_ui(x, y, xtest, ytest, M, 100, hyp_init);
for tid = 1:ns
    xtest = all_x(n+tid,:);
    ytest = all_y(n+tid);
    [nmse, mu, s2, nmlp, newhyp, convergence] = ssgpr_ui(x, y, xtest, ytest, M, 20, newhyp);
    pred_mu(tid) = mu;
    pred_s2(tid) = s2;
    nses_ssgpr(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
    x = [x; xtest];
    y = [y; ytest];
end
if NORMALIZE_DATA
    pred_mu = bsxfun(@plus, bsxfun(@times, pred_mu, stdy), meany);
    pred_s2 = bsxfun(@times, pred_s2, stdy^2);
end
nmse_ssgpr = mean(nses_ssgpr);
plotResult(raw_x, raw_y, raw_x(1+n:n+ns), pred_mu, pred_s2);
filename = strcat('sinc/figs/sinc_ssgpr_M', int2str(M), '.pdf');
saveas(gcf, filename, 'pdf');

%% print result
if TEST_FULL
    fprintf('full GP: %f\n', nmse_full);
end
fprintf('composite EigenGP: %f\n', nmse_compositeEigenGP);
fprintf('kerB EigenGP: %f\n', nmse_kerB);
fprintf('FITC: %f\n', nmse_fitc);
fprintf('SSGPR: %f\n', nmse_ssgpr);

end

function plotResult(raw_x, raw_y, xs, mu, s2, B) 
clf
hold on
plot(raw_x,raw_y,'.m', 'MarkerSize', 5)% data points in magenta
plot(xs, sinc(xs), '-', 'Color', [0 .5 0]);
plot(xs, mu,'b') % mean predictions in blue
plot(xs,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation predictions in red
plot(xs,mu-2*sqrt(s2),'r')
if nargin > 5
    plot(B,-0.4*ones(size(B)),'k+','markersize',20)
end
hold off
%axis([0 5 -0.5 1]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')
end