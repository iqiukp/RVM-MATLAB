# Relevance-Vector-Machine

## Prediction based on Relevance Vector Machine (RVM), SB2_Release_200

---------------------------------------------------------
Updated on 5 July 2019	
1. Fixed some errors 
2. Optimized the code
3. Added some functions
---------------------------------------------------------  

---------------------------------------------------------
Updated on 11 May 2019	
1. Fixed some errors 
2. Optimized the code
3. Modified the function 'SparseBayes.m'
---------------------------------------------------------  

## demo: Prediction for a numerical example using RVM

```
clc
clear all
close all
addpath(genpath(pwd))

% Generate data
[x, y, xt, yt] = generateData;

% Train RVM model
model = rvm_train(x,y,'s',7,'b',0);

% Predict the training samples
[y_mu,y_var] = rvm_test(model,x);

% Predict the testing samples
[yt_mu,yt_var] = rvm_test(model,xt);

% Plot the training results 
plottrainingResult(x,y,model)

% Plot the testing results 
plottestingResult(xt,yt,yt_mu,yt_var)

% Compute regression performance evaluation index
[RMSE,CD,MAE] = computePretIndex(yt,yt_mu);
```

![](img/img1.png)![](img/img2.png)  
  
