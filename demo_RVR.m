%{
        A demo for regression using RVM
%}

clc
clear all
close all
addpath(genpath(pwd))

% sinc funciton
fun = @(x) sin(abs(x))/abs(x);
x = linspace(-10, 10, 100);
y = arrayfun(fun, x);
trainData = x';
trainLabel = y';
xt = linspace(-10, 10, 20);
yt = arrayfun(fun, xt);
testData = xt';
testLabel = yt';

% kernel function
kernel = Kernel('type', 'gaussian', 'gamma', 0.02);

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVR',...
                    'kernelFunc', kernel);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(trainData, trainLabel);
rvm.draw(results)


