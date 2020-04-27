
% demo: optimize the kernel parameters and weight

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

%{
------------------------------------------------------------------------
Notice:  
    'method'         'pso' or 'ga'
    'target'         name of the kernel that need to be optimized
    'lb'             lower boundary of parameters
    'ub'             upper boundary of parameters
    'numVariable'    number of parameters that need to be optimized
    'maxIter'        max iterations

optional
    'Kfolds'         K fold cross validation

(1) 'weight' is the weight of each kernel. The number of weight is equal to 
    the number of kernels.

------------------------------------------------------------------------
%}
optimization = struct('method', 'pso',...
                      'target', {{kernel_1, kernel_2, 'weight'}},...
                      'lb', [2^-5,  10^-2, 10^-3, 10^-3],...
                      'ub', [2^5, 10^0, 10^3, 10^3],...
                      'numVariable', 4,...
                      'maxIter', 10,...
                      'Kfolds', 5);

% parameter setting
parameter = struct( 'freeBasis', 'on',...
                    'display', 'on',...
                    'maxIter', 1000,...
                    'kernel', [kernel_1, kernel_2],...
                    'optimization', optimization);
                
% RVM model training, testing, and visualization
rvm = RVM(parameter);
rvm.train(x, y);
rvm.test(xt, yt);
rvm.draw


