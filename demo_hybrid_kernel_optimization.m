%{
    A demo for hybrid-kernel RVM model with Parameter Optimization
%}


clc
clear all
close all
addpath(genpath(pwd))

% data
load UCI_data
trainData = x;
trainLabel = y;
testData = xt;
testLabel = yt;

% kernel function
kernel_1 = Kernel('type', 'gaussian', 'gamma', 0.5);
kernel_2 = Kernel('type', 'polynomial', 'degree', 2);

% parameter optimization
opt.method = 'bayes'; % bayes, ga, pso
opt.display = 'on';
opt.iteration = 30;

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', [kernel_1, kernel_2],...
                    'optimization', opt);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
rvm.draw(results)