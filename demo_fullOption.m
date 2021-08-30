%{
        A demo for option
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

% parameter optimization
opt.method = 'pso'; % bayes, ga, pso
opt.display = 'on';
opt.iteration = 20;
opt.gaussian.parameterName = {'gamma'};
opt.gaussian.parameterType = {'real'};
opt.gaussian.lowerBound = 2^-6;
opt.gaussian.upperBound = 2^6;

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVC',...
                    'maxIter', 1000,...
                    'bias', 'on',...
                    'kernelFunc', kernel,...
                    'optimization', opt);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(testData, testLabel);
rvm.draw(results)