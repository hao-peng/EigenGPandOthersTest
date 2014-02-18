function model = EigenGPNS_param2model(param, D, M)
if size(param, 1) ~= D*M+D+4
   error('The size of param does not match. (Reuiqred: %d; Given: %d)',...
       D*M+D+4, size(param,1)); 
end
model.logSigma = param(1);
model.logEta = param(2:1+D);
model.logA0 = param(2+D);
model.logA1 = param(3+D);
model.logA2 = param(4+D);
model.B = reshape(param(5+D:D*M+D+4), M, D);
end