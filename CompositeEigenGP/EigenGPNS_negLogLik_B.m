% Compute the negative log likelihood and its derivative respect to eacch
% model paramter. We use an ARD kernel plus a linear kernel:
% k(x,y) = a0*exp(-(x-y)'*diag(eta)*(x-y))+a1*x'*y+a2
% paramters:
% param - current parameters for the model
%       element (1) - (D*M): B (reshaped B matrix as a vector)
% model - other parameters
% X - input data point
%     N by D matrix, where each row is a data point
% t - labels
%     N by 1 vector
% M - number of basis point used

function [f df] = EigenGPNS_negLogLik_B(param, model, X, t, M)
[N D] = size(X);
% load parameters
sigma2 = exp(2*model.logSigma);
eta = exp(model.logEta);
a0 = exp(model.logA0);
a1 = exp(model.logA1);
a2 = exp(model.logA2);
B = reshape(param, M, D);
% to avoid semi positive definite
epsilon = 1e-10;
% Some commonly used terms
X2 = X.*X;
B2 = B.*B;
X_B = X*B';
B_B = B*B';
X_eta = bsxfun(@times,X,eta');
B_eta = bsxfun(@times,B,eta');
% Compute gram matrices
expH = exp(bsxfun(@minus,bsxfun(@minus,2*X_eta*B',X2*eta),(B2*eta)'));
Kxb = a0*expH+a1*(X_B)+a2;
expF = exp(bsxfun(@minus,bsxfun(@minus,2*B_eta*B',B2*eta),(B2*eta)'));
Kbb = a0*expF+a1*(B_B)+a2 + epsilon*eye(M);

% Define Q = Kbb + 1/sigma2 * Kbx *Kxb
Q = Kbb+(Kxb'*Kxb)/sigma2;
% Cholesky factorization for stable computation
cholKbb = chol(Kbb,'lower');
cholQ = chol(Q,'lower');
% Other commonly used terms
lowerOpt.LT = true; upperOpt.LT = true; upperOpt.TRANSA = true;
invCholQ_Kbx_invSigma2 = linsolve(cholQ,Kxb'/sigma2,lowerOpt);
invCholQ_Kbx_invSigma2_t = invCholQ_Kbx_invSigma2*t;
diagInvCN = 1/sigma2-sum(invCholQ_Kbx_invSigma2.^2, 1)';
invCN_t = t/sigma2-invCholQ_Kbx_invSigma2'*invCholQ_Kbx_invSigma2_t;

% compute negative log likelihood function f = (ln|CN|+t'*CN*t+ln(2*pi))/2
f = sum(log(diag(cholQ)))-sum(log(diag(cholKbb)))+(log(sigma2)*N+...
    t'*t/sigma2-invCholQ_Kbx_invSigma2_t'*invCholQ_Kbx_invSigma2_t...
    +N*log(2*pi))/2;

%f = sum(log(diag(cholQ)))-sum(log(diag(cholKbb)))+(log(sigma2)*N)/2;

%-----------------------
% compute gradient
%-----------------------
% prepare things that may be used later
invKbb_Kbx_invCN = linsolve(cholQ,invCholQ_Kbx_invSigma2,upperOpt);
invKbb_Kbx_invCN_Kxb_invKbb = linsolve(cholKbb, linsolve(cholKbb, Kxb'*invKbb_Kbx_invCN',lowerOpt),upperOpt)';
%invKbb_Kbx_invCN_Kxb_invKbb = inv(Kbb) - inv(Q)
invKbb_Kbx_invCN_t = invKbb_Kbx_invCN*t;
invKbb_Kbx_invCN_t_t_invCN = invKbb_Kbx_invCN_t*invCN_t';
invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb = invKbb_Kbx_invCN_t*invKbb_Kbx_invCN_t';

R1 = invKbb_Kbx_invCN.*(a0*expH)';
S1 = invKbb_Kbx_invCN_Kxb_invKbb.*(a0*expF);
R2 = invKbb_Kbx_invCN_t_t_invCN.*(a0*expH)';
S2 = invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb.*(a0*expF);

% compute dB
% part1 = tr(inv(CN)*dCN)/2
part1 = 2*(2*R1*X_eta-2*repmat(sum(R1,2),1,D).*B_eta+a1*invKbb_Kbx_invCN*X)...
    +(-4*S1*B_eta+4*repmat(sum(S1,2),1,D).*B_eta-2*a1*invKbb_Kbx_invCN_Kxb_invKbb*B);

part2 = 2*(2*R2*X_eta-2*repmat(sum(R2,2),1,D).*B_eta+a1*invKbb_Kbx_invCN_t_t_invCN*X)...
    +(-4*S2*B_eta+4*repmat(sum(S2,2),1,D).*B_eta-2*a1*invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb*B);

dB = (part1-part2)/2;

% combine all gradients in a vector
df = reshape(dB,D*M,1);
end
