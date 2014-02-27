% one-step prediction for synthetic sinc data using curve fitting method
% The input (x) is the time
% The ouput (y) is the value.
% M - number of basis
function testGPCF_G(M)
addpath('ARDEigenGP');
addpath('CompositeEigenGP');
addpath('GPML');
startup;
addpath('lightspeed');
addpath('SPGP_dist');
addpath('ssgpr_code');
mkdir('synGP/figs2');

GENERATE_DATA = false;
TEST_FULL = true; % whether to test full

%% initialize arguments if they are not assigned.
if ~exist('M', 'var')
    M = 10;
end

global MAX_X;
MAX_X = 15;
%% number of pseudo-inputs
M = 10;

%% generate data
seed = 0;
rand('seed',seed); randn('seed',seed);

N = 200;
Ns = 500;
D = 1;
% a0 = alpha*x+0.1
% 1/eta = beta*x
alpha = 0.1;
beta = 0.1;
sigma = 0.1;

numTest = 5;


if GENERATE_DATA
for cid = 1:numTest
% generating
all_x = rand(N+Ns, 1)*MAX_X;

C = zeros(N+Ns);
for i = 1:N+Ns
    for j = 1:N+Ns
        C(i,j) = (alpha*(all_x(i)+all_x(j))+0.1)*(beta*all_x(i))^0.25*(beta*all_x(j))^0.25*(beta*(all_x(i)+all_x(j))/2)^-0.5...
            *exp(-(all_x(i)-all_x(j))^2/(beta*(all_x(i)+all_x(j))/2));
    end
end
all_y = mvnrnd(zeros(N+Ns,1), C)';

x = all_x(1:N, 1);
y = all_y(1:N, 1) + normrnd(0, sigma, N, 1);


[xtest iX] = sort(all_x(N+1:end,1));
ytest = all_y(N+1:end,1);
ytest = ytest(iX);

save(strcat('synGP/synGP_', int2str(cid),'.mat'), 'x', 'y','xtest','ytest');
end
end

n = 50;
ns = N - n;

%% full GP
if TEST_FULL
    seed = 0;
    rand('seed',seed); randn('seed',seed);
    for cid = 1:numTest
        load(strcat('synGP/synGP_',int2str(cid),'.mat'));
        N = size(x, 1);
        [all_x IND] = sort(x);
        all_y = y(IND);
        x = all_x(1:n,:);
        y = all_y(1:n);
        xtest = all_x(n+1,:);
        ytest = all_y(n+1);
        opt.cov(1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
        opt.cov(2) = log(var(y,1)); % log size 
        opt.lik = log(var(y,1)/4); % log noise

        hyp1 = minimize(opt, @gp, 100, @infExact, [], {@covSEard}, @likGauss, x, y);
        for tid = 1:ns
            xtest = all_x(n+tid,:);
            ytest = all_y(n+tid);
            %hyp1 = minimize(hyp1, @gp, 0, @infExact, [], {@covSEard}, @likGauss, x, y);
            [mu, s2] = gp(hyp1, @infExact, [], {@covSEard}, @likGauss, x, y, xtest);
            pred_mu(tid) = mu;
            pred_s2(tid) = s2;
            nses_full(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
            x = [x; xtest];
            y = [y; ytest];
        end
        nmses_full(cid) = mean(nses_full);
        plotResult(x, y, xtest, ytest, x(1+n:n+ns), pred_mu, pred_s2);
        filename = strcat('synGP/figs2/synGP_full_',int2str(cid),'.pdf');
        saveas(gcf, filename, 'pdf');
    end
end

%% composite eigenGP
seed = 0;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('synGP/synGP_',int2str(cid),'.mat'));
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    model.logSigma = log(var(y,1));
    model.logEta = log((max(x)-min(x))')/2; %log eta
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
        trained_model = EigenGPNS_trainB(trained_model, x, y, M, 20);
        [mu s2] = EigenGPNS_pred(trained_model, x, y, xtest);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;

        nses_compositeEigenGP(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];

    end

    nmses_compositeEigenGP(cid) = mean(nses_compositeEigenGP);
    plotResult(x, y, xtest, ytest, x(1+n:n+ns), pred_mu, pred_s2, trained_model.B);
    filename = strcat('synGP/figs2/synGP_compositeEigenGP_M', int2str(M), '_', int2str(cid),'.pdf');
    saveas(gcf, filename, 'pdf');
end

%% kerB EigenGP
seed = 0;
rand('seed',seed); randn('seed',seed);

for cid = 1:numTest
    load(strcat('synGP/synGP_',int2str(cid),'.mat'));
    N = size(x, 1);
    [all_x IND] = sort(x);
    all_y = y(IND);
    x = all_x(1:n,:);
    y = all_y(1:n);
    xtest = all_x(n+1,:);
    ytest = all_y(n+1);

    opt.cov(1) = -2*log((max(x)-min(x))'); % log 1/(lengthscales)^2
    opt.cov(2) = log(var(y,1)); % log size 
    opt.lik = log(var(y,1)/4); % log noise
    opt.nIter = 100;
    [nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_B_UI(x, y, xtest, ytest, M, opt);
    post.opt.nIter = 20;
    for tid = 1:ns
        xtest = all_x(n+tid,:);
        ytest = all_y(n+tid);
        if tid ~= 1
            post.opt = rmfield(post.opt, 'B');
        end
        [nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_B_UI(x, y, xtest, ytest, M, post.opt);
        pred_mu(tid) = mu;
        pred_s2(tid) = s2;
        nses_kerB(tid) = (mu-ytest)^2/(mean(y)-ytest)^2;
        x = [x; xtest];
        y = [y; ytest];
    end
    nmses_kerB(cid) = mean(nses_kerB);
    plotResult(x, y, xtest, ytest, x(1+n:n+ns), pred_mu, pred_s2, post.opt.B);
    filename = strcat('synGP/figs2/synGP_kerB_M', int2str(M),'_',int2str(cid),'.pdf');
    saveas(gcf, filename, 'pdf');
end

%% print result

save('tmpResult.mat');
if TEST_FULL
   fprintf('ARD fullGP: %f +- %f\n', mean(nmses_full), std(nmses_full)/sqrt(numTest));
end
fprintf('composite EigenGP: %f +- %f\n', mean(nmses_compositeEigenGP), std(nmses_compositeEigenGP)/sqrt(numTest));
fprintf('kerB EigenGP: %f +- %f\n', mean(nmses_kerB), std(nmses_kerB)/sqrt(numTest));

end

function plotResult(x, y, xtest, ytest, xs, mu, s2, B) 
clf
hold on
plot(x,y,'.m', 'MarkerSize', 5)% data points in magenta
plot(xtest, ytest, '-', 'Color', [0 .5 0]);
plot(xs, mu,'b') % mean predictions in blue
plot(xs,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation predictions in red
plot(xs,mu-2*sqrt(s2),'r')
if nargin > 7
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