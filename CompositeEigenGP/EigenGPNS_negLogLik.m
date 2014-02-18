% Compute the negative log likelihood and its derivative respect to eacch
% model paramter. We use an ARD kernel plus a linear kernel:
% k(x,y) = a0*exp(-(x-y)'*diag(eta)*(x-y))+a1*x'*y+a2
% paramters:
% param - current parameters for the model
%       element 1: log(sigma)
%       element 2 - (1+D): log(eta)
%       element (D+2): log(a0)
%       element (D+3): log(a1)
%       element (D+4): log(a2)
%       element (D+5) - (D+4+D*M): B (reshaped B matrix as a vector)
% X - input data point
%     N by D matrix, where each row is a data point
% t - labels
%     N by 1 vector
% M - number of basis point used

function [f df] = EigenGPNS_negLogLik(param, X, t, M)
[N D] = size(X);
% load parameters
sigma2 = exp(2*param(1));
eta = exp(param(2:1+D));
a0 = exp(param(2+D));
a1 = exp(param(3+D));
a2 = exp(param(4+D));
B = reshape(param(5+D:D*M+D+4), M, D);
% to avoid semi positive definite
epsilon = 0;%1e-11;
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

% compute dlogSigma
dlogSigma = sigma2*(sum(diagInvCN)-invCN_t'*invCN_t);

% compute dlogEta
% part1 = tr(inv(CN)*dCN)
part1 = 2*(2*sum(B'.*(X'*R1'), 2)-B2'*sum(R1,2)-X2'*sum(R1,1)')...
    +(-2*sum(B.*(S1*B),1)'+2*B2'*sum(S1, 1)');
% part2 = tr(inv(CN)*t*t'*inv(CN)*dCN)
part2 = 2*(2*sum(B'.*(X'*R2'), 2)-B2'*sum(R2,2)-X2'*sum(R2,1)')...
    +(-2*sum(B.*(S2*B),1)'+2*B2'*sum(S2, 1)');
dlogEta = eta.*(part1-part2)/2;

% compute dlogA0
% part1 = tr(inv(CN)*dCN)
part1 = 2*sum(sum(invKbb_Kbx_invCN.*expH'))-sum(sum(invKbb_Kbx_invCN_Kxb_invKbb.*expF'));
%part1 = 2*sum(sum(invKbb_Kbx_invCN.*expH'))-trace(invKbb_Kbx_invCN_Kxb_invKbb_expF);
% part2 = tr(inv(CN)*t*t'*inv(CN)*dCN)
part2 = 2*sum(sum(invKbb_Kbx_invCN_t_t_invCN.*expH'))-sum(sum(invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb.*expF'));
dlogA0 = a0*(part1-part2)/2;

% compute dlogA1
% part1 = tr(inv(CN)*dCN)
part1 = 2*sum(sum(invKbb_Kbx_invCN.*X_B'))-sum(sum(invKbb_Kbx_invCN_Kxb_invKbb.*B_B));
% part2 = tr(inv(CN)*t*t'*inv(CN)*dCN)
part2 = 2*sum(sum(invKbb_Kbx_invCN_t_t_invCN.*X_B'))-sum(sum(invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb.*B_B'));
dlogA1 = a1*(part1-part2)/2;

% compute dlogA2
% part1 = tr(inv(CN)*dCN)
part1 = 2*sum(sum(invKbb_Kbx_invCN))-sum(sum(invKbb_Kbx_invCN_Kxb_invKbb));
% part2 = tr(inv(CN)*t*t'*inv(CN)*dCN)
part2 = 2*sum(sum(invKbb_Kbx_invCN_t_t_invCN))-sum(sum(invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb));
dlogA2 = a2*(part1-part2)/2;

% compute dB
% part1 = tr(inv(CN)*dCN)/2
part1 = 2*(2*R1*X_eta-2*repmat(sum(R1,2),1,D).*B_eta+a1*invKbb_Kbx_invCN*X)...
    +(-4*S1*B_eta+4*repmat(sum(S1,2),1,D).*B_eta-2*a1*invKbb_Kbx_invCN_Kxb_invKbb*B);

part2 = 2*(2*R2*X_eta-2*repmat(sum(R2,2),1,D).*B_eta+a1*invKbb_Kbx_invCN_t_t_invCN*X)...
    +(-4*S2*B_eta+4*repmat(sum(S2,2),1,D).*B_eta-2*a1*invKbb_Kbx_invCN_t_t_invCN_Kxb_invKbb*B);

dB = (part1-part2)/2;

% combine all gradients in a vector
df = [dlogSigma; dlogEta; dlogA0; dlogA1; dlogA2; reshape(dB,D*M,1)];
end
