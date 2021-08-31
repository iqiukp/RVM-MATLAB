%{
        A demo for classification using RVM
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
kernel = Kernel('type', 'gaussian', 'gamma', 0.2);

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVC',...
                    'kernelFunc', kernel);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
rvm.draw(results)