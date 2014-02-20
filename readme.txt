testSyn1.m:
test file for the first synthetic data from fitc.

testSyn2.m:
test file for the second synthetic data
y = x.*sin(x.^3) + randn(N, 1)*0.5;

testTS.m:
test file for the time series data s&p500
Each input data point is a D dimesion historical open price
The prediction value is next open price.

testSinc.m:
test file for sinc data
Each input data point is a D dimesion historcal targets
The prediction value is next target.