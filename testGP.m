% generate non-stationary 
% according to nonstationary covariance functions for gaussian process
% regression
% covariance matrix is C_ij =
% A_ij*(sigma_i)^0.25*(sigma_j)^0.25*((sigma_i+simga_j)/2)^-0.5*exp(-Q_ij);
% Q_ij = (x_i-x_j)^2/((sigma_1-sigma_2)/2)
% A_ij = alpha*(x_i+x_j)/2+0.1
% sigma_i = beta*x_i


function testGP
addpath('ARDEigenGP');
addpath('CompositeEigenGP');
addpath('GPML');
startup;
addpath('lightspeed');
addpath('SPGP_dist');
addpath('ssgpr_code');

set(gcf,'defaultlinelinewidth',1.5);

mkdir('synGP/figs');
GENERATE_DATA = true;

%% generate data
if GENERATE_DATA
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

% generating
all_x = rand(N+Ns, 1)*5;

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

save('synGP/synGP.mat', 'x', 'y','xtest','ytest');
else
   load('synGP/synGP.mat');
   [N D] = size(x);
end

numTest = 10;


%% number of pseudo-inputs
M = 10;

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

nmse_compositeEigenGP(tid) = getNMSE(mu, ytest);
plotResult(x, y, xtest, ytest, mu, s2, trained_model.B);
filename = strcat('synGP/figs/synGP_CompositeEigenGP_M', int2str(M),  '_', int2str(tid),'.pdf');
saveas(gcf, filename, 'pdf');
plotCompositeEigenFunctions(trained_model, xtest);
filename = strcat('synGP/figs/synGP_Eigen_CompositeEigenGP_M', int2str(M),  '_', int2str(tid),'.pdf');
saveas(gcf, filename, 'pdf');
end

%% ARD update kerB
seed = 0;
rand('seed',seed); randn('seed',seed);
for tid = 1:numTest
[nmse, mu, s2, nmlp, post] = EigenGP_Upd_kerB_UI(x, y, xtest, mu_full, M, opts{tid});
nmse_kerBEigenGP(tid) = getNMSE(mu, ytest);
plotResult(x, y, xtest, ytest, mu, s2, post.opt.B);
filename = strcat('synGP/figs/synGP_kerBEigenGP_M', int2str(M),  '_', int2str(tid),'.pdf');
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

nmse_fitc(tid) = getNMSE(mu, ytest);
plotResult(x, y, xtest, ytest, mu, s2, xb);
filename = strcat('synGP/figs/synGP_fitc_M', int2str(M),  '_', int2str(tid),'.pdf');
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

nmse_ssgpr(tid) = getNMSE(mu, ytest);
plotResult(x, y, xtest, ytest, mu, s2);
filename = strcat('synGP/figs/synGP_ssgpr_M', int2str(M),  '_', int2str(tid),'.pdf');
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
plot(xs,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation predictions in red
plot(xs,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
if nargin > 6
    plot(B,-1.9*ones(size(B)),'k+','markersize',20)
end
hold off
axis([0 5 -2 2.5]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
set(gca, 'fontsize',20);
set(gcf, 'PaperSize', [6.2 4.8]);
set(gcf, 'PaperPositionMode', 'auto')
end

function plotCompositeEigenFunctions(model, xtest)
sigma2 = exp(model.logSigma*2);
eta = exp(model.logEta);
a0 = exp(model.logA0);
a1 = exp(model.logA1);
a2 = exp(model.logA2);
B = model.B;
M = size(B, 1);
% to avoid semi positive definite
epsilon = 1e-10;
% for later use
B2 = B.*B;
X2= xtest.*xtest;
X_B = xtest*B';
B_B = B*B';
X_eta = bsxfun(@times,xtest,eta');
B_eta = bsxfun(@times,B,eta');
expF = exp(bsxfun(@minus,bsxfun(@minus,2*B_eta*B',B2*eta),(B2*eta)'));
Kbb = a0*expF+a1*(B_B)+a2 + epsilon*eye(M);
[Uq, Lambdaq] = eig(Kbb);
[Lambda, sort_ind] = sort(abs(diag(Lambdaq)),'descend');
U = real(Uq(:,sort_ind(1:M)));
expH = exp(bsxfun(@minus,bsxfun(@minus,2*X_eta*B',X2*eta),(B2*eta)'));
Ksb = a0*expH+a1*(X_B)+a2;
Kerfun = Ksb * scale_cols(U, 1./Lambda);
clf
hold on 
%plot(xtest, mu, '-.b');
%plot(xtest, mu_full, '--r');
plot(xtest, Kerfun(:,1), '-.', 'Color', [0.54118,0.18039,0.90196]);
plot(xtest, Kerfun(:,2), '--', 'Color', [0.18039,0.90196,0.90196]);
plot(xtest, Kerfun(:,3), '-', 'Color', [0,0.90196,0]);
plot(xtest, Kerfun(:,4), '-', 'Color', [0.8,0.6,0.1]); %plot(xtest, Kerfun(:,5), '-', 'Color', [0.18039,0.90196,0.36078]);
plot(xtest, Kerfun(:,5), '-', 'Color', [0.90196,0.18039,0.18039]);
plot(B,-1.9*ones(size(B)),'k+','markersize',20);
hold off
axis([0 5 -2 2.5]);
xlabel('x', 'fontsize', 20);
ylabel('y', 'fontsize', 20);
% lg = legend( 'predictive mean', ...
%         '1st eigenfunction', ...
%         '2nd eigenfunction', ...
%         '3rd eigenfunction', ...
%         '4th eigenfunction', ...
%         '5th egienfunction');
set(gca, 'fontsize',20);
%set(lg, 'fontsize',15);
set(gcf, 'PaperSize', [6.2 4.6]);
set(gcf, 'PaperPositionMode', 'auto');
end

function nmse = getNMSE(mu, ys)
nmse = mean((mu-ys).^2)/mean((mean(mu)-ys).^2);
end