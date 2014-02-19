% one-step prediction for time series data
% increasing the training data
% The input (x) is a window of previous D days' open prices.
% The ouput (y) is the next day's open price.
% M - number of basis
% D - window size
function testTS(M, D)
addpath('ARDEigenGP');
addpath('CompositeEigenGP');
addpath('GPML');
startup;
addpath('lightspeed');
addpath('SPGP_dist');
addpath('ssgpr_code');

%% initialize arguments if they are not assigned.
if ~exist('M', 'var')
    M = 20;
end
if ~exist('D', 'var')
    D = 5;
end

%% prepare data
load('ts/sandp500.mat');
N = size(data, 1);
xall = zeros(N-D, D);
col = 1; % column 1 is the open price
for i = 1:D
    xall(:,i) = data(i:i+N-D-1,col);
end
yall = data(D+1:N,col);

%% normalize the data
n = 100; % starting training data size
ns = 1000; N-n-D; % number of tests


meanx = mean(xall(1:n,:));
stdx = std(xall(1:n,:));
meany = mean(yall(1:n));
stdy = std(yall(1:n));
xall = bsxfun(@rdivide, bsxfun(@minus, xall, meanx), stdx);
yall = bsxfun(@rdivide, bsxfun(@minus, yall, meany), stdy);

seed = 0;
rand('seed',seed); randn('seed',seed);


%% composite eigenGP
seed = 1;
rand('seed',seed); randn('seed',seed);
x = xall(1:n,:);
y = yall(1:n);
model.logSigma = log(var(y,1));
model.logEta = 2*log((max(x)-min(x))'/2);
model.logA0 = log(var(y,1)/4);
model.logA1 = 0.1;
model.logA2 = 0.1;
trained_model = EigenGPNS_train(model, x, y, M, 100);
for tid = 1:ns
    xtest = xall(n+tid,:);
    ytest = yall(n+tid);
    trained_model = EigenGPNS_train(trained_model, x, y, M, 10);
    [mu s2] = EigenGPNS_pred(trained_model, x, y, xtest);
    nses_compositeEigenGP(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
    x = [x; xtest];
    y = [y; ytest];
end
nmse_compositeEigenGP = mean(nses_compositeEigenGP);
%% FITC
seed = 1;
rand('seed',seed); randn('seed',seed);
x = xall(1:n,:);
y = yall(1:n);
hyp_init(1:D,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
hyp_init(D+1,1) = log(var(y,1)); % log size 
hyp_init(D+2,1) = log(var(y,1)/4); % log noise
% random initialize pseudo-inputs
[dum,I] = sort(rand(n,1)); clear dum;
I = I(1:M);
xb_init = x(I,:);
w_init = [reshape(xb_init,M*D,1);hyp_init];
% optimization
[w,f] = minimize(w_init,'spgp_lik',-100,y,x,M);
for tid = 1:ns
    xtest = xall(n+tid,:);
    ytest = yall(n+tid);
    [w,f] = minimize(w,'spgp_lik',-10,y,x,M);
    xb = reshape(w(1:M*D,1),M,D);
    hyp = w(M*D+1:end,1);
    % PREDICTION
    [mu,s2] = spgp_pred(y,x,xb,xtest,hyp);
    nses_fitc(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
    x = [x; xtest];
    y = [y; ytest];
end
nmse_fitc = mean(nses_fitc);

%% SSGPR
seed = 1;
rand('seed',seed); randn('seed',seed);
x = xall(1:n,:);
y = yall(1:n);
xtest = xall(n+1,:);
ytest = yall(n+1);
hyp_init(1:D,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
hyp_init(D+1,1) = log(var(y,1)); % log size 
hyp_init(D+2,1) = log(var(y,1)/4); % log noise
[nmse, mu, s2, nmlp, newhyp, convergence] = ssgpr_ui(x, y, xtest, ytest, M, 100, hyp_init);
for tid = 1:ns
    xtest = xall(n+tid,:);
    ytest = yall(n+tid);
    [nmse, mu, s2, nmlp, newhyp, convergence] = ssgpr_ui(x, y, xtest, ytest, M, 10, newhyp);
    nses_ssgpr(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
    x = [x; xtest];
    y = [y; ytest];
end
nmse_ssgpr = mean(nses_ssgpr);

fprintf('composite EigenGP: %f\n', nmse_compositeEigenGP);
fprintf('FITC: %f\n', nmse_fitc);
fprintf('SSGPR: %f\n', nmse_ssgpr);

end