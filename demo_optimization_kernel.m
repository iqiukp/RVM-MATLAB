
% demo: optimize the kernel parameters

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

% kernel function 
kernel = Kernel('type', 'gauss', 'width', 2);

%{
------------------------------------------------------------------------
Notice:  
    'method'         'pso' or 'ga'
    'target'         name of the kernel that need to be optimized
                          single kernel: {{kernel}}
                          hybrid kernel: {{kernel_1, kernel_2}}
                        
    'lb'             lower boundary of parameters
    'ub'             upper boundary of parameters
    'numVariable'    number of parameters that need to be optimized
    'maxIter'        max iterations

optional
    'Kfolds'         K fold cross validation

------------------------------------------------------------------------
%}
optimization = struct('method', 'ga',...
                      'target', {{kernel}},...
                      'lb', 2^-6,...
                      'ub', 2^6,...
                      'numVariable', 1,...
                      'maxIter', 10,...
                      'Kfolds', 5);

% parameter setting
parameter = struct( 'freeBasis', 'on',...
                    'display', 'on',...
                    'maxIter', 1000,...
                    'kernel', kernel,...
                    'optimization', optimization);
                
% RVM model training, testing, and visualization
rvm = RVM(parameter);
rvm.train(x, y);
rvm.test(xt, yt);
rvm.draw


