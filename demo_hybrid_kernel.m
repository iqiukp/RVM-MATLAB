%{
        A demo for regression using RVM with hybrid_kernel
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
kernel_1 = Kernel('type', 'gaussian', 'gamma', 0.3);
kernel_2 = Kernel('type', 'polynomial', 'degree', 2);
kernelWeight = [0.5, 0.5];
% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', [kernel_1, kernel_2],...
                    'kernelWeight', kernelWeight);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
rvm.draw(results)


