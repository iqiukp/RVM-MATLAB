%{
    RVM application for a numerical example.
%}

clc
clear all
close all
addpath(genpath(pwd))

% generate data
[x, y, xt, yt] = generateData;

% parameter setting
kernel = Kernel('type', 'gauss', 'width', 2);
option = struct('freeBasis', 'on',...
                'display', 'on');
            
% train RVM model
model = rvm_train(x, y, 'kernel', kernel, 'option', option);

% predict the test samples
result = rvm_test(model,xt, yt);

% visualization
plotRelevanceVector(model, y)
plotResult(yt, result)






