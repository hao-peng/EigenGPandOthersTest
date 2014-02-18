function param = EigenGPNS_model2param(model, D, M)
param = [model.logSigma; model.logEta; model.logA0; model.logA1;...
    model.logA2; reshape(model.B, D*M, 1)];
end