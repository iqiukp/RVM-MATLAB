
% A demo for RVM model using hybrid kernel function

clc
clear all
close all
addpath(genpath(pwd))
load UCI_data

%{
------------------------------------------------------------------------
Notice:  

training samples
    x               sample data (input, n*d)
                     n: number of samples
                     d: number of features
    y               sample data (ouput, n*1)
                     n: number of samples

test samples
    xt              sample data (input, n*d)
                     n: number of samples
                     d: number of features
    yt              sample data (ouput, n*1)
                     n: number of samples
------------------------------------------------------------------------
%}

% kernel function (default: kernel = kernel_1+kernel_2)
kernel_1 = Kernel('type', 'gauss', 'width', 2);
kernel_2 = Kernel('type', 'sigm', 'gamma', 0.6, 'offset', 0);

% parameter setting
parameter = struct( 'freeBasis', 'on',...
                    'display', 'on',...
                    'maxIter', 1000,...
                    'kernel', [kernel_1, kernel_2],...
                    'weight', [1, 1]);
                
% RVM model training, testing, and visualization
rvm = RVM(parameter);
rvm.train(x, y);
rvm.test(xt, yt);
rvm.draw