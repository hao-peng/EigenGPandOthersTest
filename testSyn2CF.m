% one-step prediction for synthetic sinc data using curve fitting method
% The input (x) is the time
% The ouput (y) is the value.
% M - number of basis
function testGPCF(M)
addpath('ARDEigenGP');
addpath('CompositeEigenGP');
addpath('GPML');
startup;
addpath('lightspeed');
addpath('SPGP_dist');
addpath('ssgpr_code');
mkdir('syn2/figs2');

TEST_FULL = true; % whether to test full

%% initialize arguments if they are not assigned.
if ~exist('M', 'var')
    M = 20;
end

%% number of pseudo-inputs
seed = 0;
rand('seed',seed); randn('seed',seed);
N = 1000;
Ns = 500;
D = 1;
numTest=1;
for ind = 1:numTest
x = linspace(0,2.5,N)';
%x = rand(N,1)*3;
%x = rand(N,1)*2*pi;
y = x.*sin(x.^3) + randn(N, 1)*0.5;
%y = sin(2*pi*2*x);
%y(x < pi) = sin(2*pi*x(x<pi));
txtest = linspace(0, 2.5, Ns)';
%xtest = linspace(0, 2*pi, Ns)';
tytest = txtest.*sin(txtest.^3);
save(strcat('syn2/syn2CF_', int2str(ind), '.mat'), 'x', 'y', 'txtest', 'tytest');
end

n = 400;
ns = N - n;

%% full GP
if TEST_FULL
    seed = 0;
    rand('seed',seed); randn('seed',seed);
    for cid = 1:numTest
        load(strcat('syn2/syn2CF_',int2str(cid),'.mat'));
        tic;
        N = size(x, 1);
        [all_x IND] = sort(x);
        all_y = y(IND);
        x = all_x(1:n,:);
        y = all_y(1:n);
        xtest = all_x(n+1,:);
        ytest = all_y(n+1);
        %opt.cov(1) = -2*log((max(x)-min(x))'*10); % log 1/(lengthscales)^2
        opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
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
            nses(tid) = (mu-ytest)^2;
            base(tid) = (mean(y)-ytest)^2;
            x = [x; xtest];
            y = [y; ytest];
        end
        times_full(cid) = toc;
        nmses_full(cid) = sum(nses)/sum(base);
        mean_nses_full(cid) = mean(nses);
        stderr_nses_full(cid) = std(nses)/sqrt(ns);
        
        mu_full(cid,:) = pred_mu;
        s2_full(cid,:) = pred_s2;
        nses_full(cid,:) = nses;
        
        plotResult(x, y, txtest, tytest, x(1+n:n+ns), pred_mu, pred_s2);
        filename = strcat('syn2/figs2/syn2_full_',int2str(cid),'.fig');
        saveas(gcf, filename, 'fig');
    end
end

%% composite eigenGP
seed = 0;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('syn2/syn2CF_',int2str(cid),'.mat'));
    tic;
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    model.logSigma = log(var(y,1));
    %model.logEta = 2*log((max(x)-min(x))'*20); %log eta
    model.logEta = 2*log((max(x)-min(x))'); %log eta
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

        nses(tid) = (mu-ytest)^2;
        base(tid) = (mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];

    end
    
    times_compositeEigenGP(cid) = toc;
    nmses_compositeEigenGP(cid) = sum(nses)/sum(base);
    mean_nses_compositeEigenGP(cid) = mean(nses);
    stderr_nses_compositeEigenGP(cid) = std(nses)/sqrt(ns);
    
    
    mu_compositeEigenGP(cid,:) = pred_mu;
    s2_compositeEigenGP(cid,:) = pred_s2;
    nses_compositieEigenGP(cid,:) = nses;
    
    plotResult(x, y, txtest, tytest, x(1+n:n+ns), pred_mu, pred_s2, trained_model.B);
    filename = strcat('syn2/figs2/syn2_compositeEigenGP_M', int2str(M), '_', int2str(cid),'.fig');
    saveas(gcf, filename, 'fig');
end

%% kerB EigenGP
seed = 0;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('syn2/syn2CF_',int2str(cid),'.mat'));
    
    tic;
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    xtest = all_x(n+1,:);
    ytest = all_y(n+1);

    %opt.cov(1) = -2*log((max(x)-min(x))'*10); % log 1/(lengthscales)^2
    opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
    opt.cov(2) = log(var(y,1)); % log size 
    opt.lik = log(var(y,1)/4); % log noise
    opt.nIter = 100;
    [nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, ytest, M, opt);
    post.opt.nIter = 20;
    for tid = 1:ns
        xtest = all_x(n+tid,:);
        ytest = all_y(n+tid);
        if tid ~= 1
            post.opt = rmfield(post.opt, 'B');
        end
        [nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, ytest, M, post.opt);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;
        nses(tid) = (mu-ytest)^2;
        base(tid) = (mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];
    end
    
    times_kerB(cid) = toc;
    nmses_kerB(cid) = sum(nses)/sum(base);
    mean_nses_kerB(cid) = mean(nses);
    stderr_nses_kerB(cid) = std(nses)/sqrt(ns);
    
    mu_kerB(cid,:) = pred_mu;
    s2_kerB(cid,:) = pred_s2;
    nses_kerB(cid,:) = nses;
    
    plotResult(x, y, txtest, tytest, x(1+n:n+ns), pred_mu, pred_s2, post.opt.B);
    filename = strcat('syn2/figs2/syn2_kerB_M', int2str(M),'_',int2str(cid),'.fig');
    saveas(gcf, filename, 'fig');
end


%% seq EigenGP
seed = 0;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('syn2/syn2CF_',int2str(cid),'.mat'));
    
    tic;
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    xtest = all_x(n+1,:);
    ytest = all_y(n+1);

    %opt.cov(1) = -2*log((max(x)-min(x))'*13); % log 1/(lengthscales)^2
    opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
    opt.cov(2) = log(var(y,1)); % log size 
    opt.lik = log(var(y,1)/4); % log noise
    opt.nIter = 100;
    [nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, ytest, M, opt);
    for tid = 1:ns
        xtest = all_x(n+tid,:);
        ytest = all_y(n+tid);
        if tid ~= 1
            post.opt = rmfield(post.opt, 'B');
        end
        post.opt.nIter = 20;
        [nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, ytest, M, post.opt);
        
        post.opt.nIter = 10;
        [nmse, mu, s2, nmlp, post] = EigenGP_Upd_W_UI(x, y, xtest, ytest, M, post.opt);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;
        nses(tid) = (mu-ytest)^2;
        base(tid) = (mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];
    end
    times_seq(cid) = toc;
    nmses_seq(cid) = sum(nses)/sum(base);
    mean_nses_seq(cid) = mean(nses);
    stderr_nses_seq(cid) = std(nses)/sqrt(ns);
    
    mu_seq(cid,:) = pred_mu;
    s2_seq(cid,:) = pred_s2;
    nses_seq(cid,:) = nses;    
    
    plotResult(x, y, txtest, tytest, x(1+n:n+ns), pred_mu, pred_s2, post.opt.B);
    filename = strcat('syn2/figs2/syn2_seq_M', int2str(M),'_',int2str(cid),'.fig');
    saveas(gcf, filename, 'fig');
end

%% Nystrom
seed = 0;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('syn2/syn2CF_',int2str(cid),'.mat'));
    
    tic;
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    xtest = all_x(n+1,:);
    ytest = all_y(n+1);

    %opt.cov(1) = -2*log((max(x)-min(x))'*10); % log 1/(lengthscales)^2
    opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
    opt.cov(2) = log(var(y,1)); % log size 
    opt.lik = log(var(y,1)/4); % log noise
    opt.nIter = 100;
    [nmse, mu, s2, nmlp, post] = Nystrom_gradient_UI(x, y, xtest, ytest, M, opt);
    post.opt.nIter = 20;
    for tid = 1:ns
        xtest = all_x(n+tid,:);
        ytest = all_y(n+tid);
        if tid ~= 1
            post.opt = rmfield(post.opt, 'B');
        end
        [nmse, mu, s2, nmlp, post] = Nystrom_gradient_UI(x, y, xtest, ytest, M, post.opt);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;
        nses(tid) = (mu-ytest)^2;
        base(tid) = (mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];
    end
    
    times_nystrom(cid) = toc;
    nmses_nystrom(cid) = sum(nses)/sum(base);
    mean_nses_nystrom(cid) = mean(nses);
    stderr_nses_nystrom(cid) = std(nses)/sqrt(ns);
    
    mu_nystrom(cid,:) = pred_mu;
    s2_nystrom(cid,:) = pred_s2;
    nses_nystrom(cid,:) = nses;
    
    plotResult(x, y, txtest, tytest, x(1+n:n+ns), pred_mu, pred_s2, post.opt.B);
    filename = strcat('syn2/figs2/syn2_nystrom_M', int2str(M),'_',int2str(cid),'.fig');
    saveas(gcf, filename, 'fig');
end

%% FITC
seed = 1;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('syn2/syn2CF_',int2str(cid),'.mat'));
    tic;
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    
    hyp_init(1,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
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
        if tid ~= 1
            [dum,I] = sort(rand(size(x,1),1));
            w(1:M,1) = x(I(1:M),:);
        end
        [w,f] = minimize(w,'spgp_lik',-210,y,x,M);
        xb = reshape(w(1:M,1),M,1);
        hyp = w(M+1:end,1);
        % PREDICTION
        [mu,s2] = spgp_pred(y,x,xb,xtest,hyp);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;
        nses(tid) = (mu-ytest)^2;
        base(tid) = (mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];
    end
    times_fitc(cid) = toc;
    nmses_fitc(cid) = sum(nses)/sum(base);
    mean_nses_fitc(cid) = mean(nses);
    stderr_nses_fitc(cid) = std(nses)/sqrt(ns);
    
	mu_fitc(cid,:) = pred_mu;
    s2_fitc(cid,:) = pred_s2;
    nses_fitc(cid,:) = nses;
    
    plotResult(x, y, txtest, tytest, x(1+n:n+ns), pred_mu, pred_s2, xb);
    filename = strcat('syn2/figs2/syn2_fitc_M', int2str(M), '_', int2str(cid), '.fig');
    saveas(gcf, filename, 'fig');
end
%% SSGPR
seed = 1;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('syn2/syn2CF_',int2str(cid),'.mat'));
    tic;
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    xtest = all_x(n+1,:);
    ytest = all_y(n+1);
    hyp_init(1,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
    hyp_init(2,1) = log(var(y,1)); % log size 
    hyp_init(3,1) = log(var(y,1)/4); % log noise
    [nmse, mu, s2, nmlp, newhyp, convergence] = ssgpr_ui(x, y, xtest, ytest, M, 100, hyp_init);
    for tid = 1:ns
        xtest = all_x(n+tid,:);
        ytest = all_y(n+tid);
        [nmse, mu, s2, nmlp, newhyp, convergence] = ssgpr_ui(x, y, xtest, ytest, M, 20, newhyp);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;
        nses(tid) = (mu-ytest)^2;
        base(tid) = (mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];
    end
    times_ssgpr(cid) = toc;
    nmses_ssgpr(cid) = sum(nses)/sum(base);
    mean_nses_ssgpr(cid) = mean(nses);
    stderr_nses_ssgpr(cid) = std(nses)/sqrt(ns);
    
	mu_ssgpr(cid,:) = pred_mu;
    s2_ssgpr(cid,:) = pred_s2;
    nses_ssgpr(cid,:) = nses;
    
    plotResult(x, y, txtest, tytest, x(1+n:n+ns), pred_mu, pred_s2);
    filename = strcat('syn2/figs2/syn2_ssgpr_M', int2str(M), '_', int2str(cid), '.fig');
    saveas(gcf, filename, 'fig');
end

%% print result

save('tmpResult.mat');
if TEST_FULL
   fprintf('ARD fullGP: %f +- %f\n', mean(nmses_full), std(nmses_full)/sqrt(numTest));
end
fprintf('composite EigenGP: %f +- %f\n', mean(nmses_compositeEigenGP), std(nmses_compositeEigenGP)/sqrt(numTest));
fprintf('kerB EigenGP: %f +- %f\n', mean(nmses_kerB), std(nmses_kerB)/sqrt(numTest));
fprintf('seq EigenGP: %f +- %f\n', mean(nmses_seq), std(nmses_seq)/sqrt(numTest));
fprintf('Nystrom: %f +- %f\n', mean(nmses_nystrom), std(nmses_nystrom)/sqrt(numTest));
fprintf('FITC: %f +- %f\n', mean(nmses_fitc), std(nmses_fitc)/sqrt(numTest));
fprintf('SSGPR: %f +- %f\n', mean(nmses_ssgpr), std(nmses_ssgpr)/sqrt(numTest));

end

function plotResult(x, y, xtest, ytest, xs, mu, s2, B) 
clf
hold on
set(gcf,'defaultlinelinewidth',1.5);
axis([0 2.5, -3 3]);
%plot(x,y,'.m', 'MarkerSize', 10)% data points in magenta
%plot(xtest, ytest, '-', 'Color', [0 .5 0]);
plot(xtest, ytest, '-', 'Color', 'c');
plot(xs, mu,'b') % mean predictions in blue
plot(xs,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation predictions in red
plot(xs,mu-2*sqrt(s2),'r')
if nargin > 7
    plot(B,(min(ytest)-1)*ones(size(B)),'k+','markersize',20)
end
hold off
%axis([0 5 -0.5 1]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')
end