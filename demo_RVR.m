%{
        A demo for regression using RVM
%}

clc
clear all
close all
addpath(genpath(pwd))

% sinc funciton
load sinc_data
trainData = x;
trainLabel = y;
testData = xt;
testLabel = yt;

% kernel function
kernel = Kernel('type', 'gaussian', 'gamma', 0.1);

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', kernel);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
rvm.draw(results)


