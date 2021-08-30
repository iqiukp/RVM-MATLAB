%{
    A demo for hybrid-kernel RVM model with Parameter Optimization
%}


clc
clear all
close all
addpath(genpath(pwd))

% use fisheriris dataset
load fisheriris
inds = ~strcmp(species, 'setosa');
data_ = meas(inds, 3:4);
label_ = species(inds);
cvIndices = crossvalind('HoldOut', length(data_), 0.3);
trainData = data_(cvIndices, :);
trainLabel = label_(cvIndices, :);
testData = data_(~cvIndices, :);
testLabel = label_(~cvIndices, :);

% kernel function
kernel_1 = Kernel('type', 'gaussian', 'gamma', 0.5);
kernel_2 = Kernel('type', 'polynomial', 'degree', 2);
% kernel_3 = Kernel('type', 'sigmoid', 'gamma', 2);
% parameter optimization
opt.method = 'bayes'; % bayes, ga, pso
opt.display = 'on';
opt.iteration = 30;

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVC',...
                    'kernelFunc', [kernel_1, kernel_2],...
                    'optimization', opt);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(trainData, trainLabel);
rvm.draw(results)


