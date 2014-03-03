setwd('/Users/pengh/Documents/eigenGP/2014_02_04/EigenGPandOthersTest/syn2/')
set.seed(0);
numTest = 1;
n = 400;

for (i in 1:numTest) {
  ptm <- proc.time()
  data <- readMat(paste('syn2CF_', toString(i), '.mat', sep=''))
  y_all = as.numeric(unlist(data['y']));
  y = y_all[1:n];
  mu = c();
  se = c();
  for (j in 1:(length(y_all)-n)) {
    #mu[j] = as.numeric(predict(object =arima(x = y, order=c(1, 0, 10))));
    #model = ar(x = y);
    p = predict(object = arima(x=y, order = c(0, 0, 10)));
    mu[j] = as.numeric(p['pred']);
    se[j] = as.numeric(p['se']);
    y = c(y, y_all[n+j]);
  }
  time = as.numeric(proc.time() - ptm);
  
  writeMat(paste('ARMA_', toString(i),'.mat',sep=''), mu=mu, se=se, time=time);
}