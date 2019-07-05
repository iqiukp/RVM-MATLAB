%%%%%%%%%%      Relevance Vector Machine (RVM)              %%%%%%%%%%%%%%%
% Demo: prediction for a numerical example using RVM
% Improve the performance by adjusting the  kernel width
%
% Created on 5th July 2019, by Kepeng Qiu.
% ---------------------------------------------------------------------%

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

