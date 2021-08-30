<p align="center">
  <img width="70%" height="70%" src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/imgs/RVR.png">
</p>

<h3 align="center">Relevance Vector Machine (RVM)</h3>

<p align="center">MATLAB code for Relevance Vector Machine</p>
<p align="center">Version 2.1, 31-AUG-2021</p>
<p align="center">Email: iqiukp@outlook.com</p>

<div align=center>

[![View Support Vector Data Description (SVDD) on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://ww2.mathworks.cn/matlabcentral/fileexchange/69296-support-vector-data-description-svdd)
<img src="https://img.shields.io/github/v/release/iqiukp/Relevance-Vector-Machine?label=version" />
<img src="https://img.shields.io/github/repo-size/iqiukp/Relevance-Vector-Machine" />
<img src="https://img.shields.io/github/languages/code-size/iqiukp/Relevance-Vector-Machine" />
<img src="https://img.shields.io/github/languages/top/iqiukp/Relevance-Vector-Machine" />
<img src="https://img.shields.io/github/stars/iqiukp/Relevance-Vector-Machine" />
<img src="https://img.shields.io/github/forks/iqiukp/Relevance-Vector-Machine" />
</div>

<hr />

## Main features

- RVM model for binary classification (RVC) or regression (RVR)
- Multiple kinds of kernel functions (linear, gaussian, polynomial, sigmoid, laplacian)
- Hybrid kernel functions
- Parameter Optimization using Bayesian optimization, Genetic Algorithm, and Particle Swarm Optimization

## Notices

- This version of the code is not compatible with the versions lower than R2016b.
- Detailed applications please see the demonstrations.
- Hybrid kernel functions: K = w1*K1+w2*K2+...+wn*Kn
- This code is for reference only.

## How to use

### 01. Classification using RVM (RVC)

A demo for classification using RVM
```
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
```
<p align="center">
  <img width="50%" height="50%" src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/imgs/RVC_1.png">
  <img width="50%" height="50%" src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/imgs/RVC_2.png">
</p>


### 02. Regression using RVM (RVR)

A demo for regression using RVM
```
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
```
<p align="center">
  <img width="50%" height="50%" src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/imgs/RVR.png">
</p>


### 03. Kernel funcions

A class named ***Kernel*** is defined to compute kernel function matrix.
```
%{
        type   -
        
        linear      :  k(x,y) = x'*y
        polynomial  :  k(x,y) = (γ*x'*y+c)^d
        gaussian    :  k(x,y) = exp(-γ*||x-y||^2)
        sigmoid     :  k(x,y) = tanh(γ*x'*y+c)
        laplacian   :  k(x,y) = exp(-γ*||x-y||)
    
    
        degree -  d
        offset -  c
        gamma  -  γ
%}
kernel = Kernel('type', 'gaussian', 'gamma', value);
kernel = Kernel('type', 'polynomial', 'degree', value);
kernel = Kernel('type', 'linear');
kernel = Kernel('type', 'sigmoid', 'gamma', value);
kernel = Kernel('type', 'laplacian', 'gamma', value);
```
For example, compute the kernel matrix between **X** and **Y**
```
X = rand(5, 2);
Y = rand(3, 2);
kernel = Kernel('type', 'gaussian', 'gamma', 2);
kernelMatrix = kernel.computeMatrix(X, Y);
>> kernelMatrix

kernelMatrix =

    0.5684    0.5607    0.4007
    0.4651    0.8383    0.5091
    0.8392    0.7116    0.9834
    0.4731    0.8816    0.8052
    0.5034    0.9807    0.7274
```

### 04. Hybrid kernel

A demo for regression using RVM with hybrid_kernel (K = w1*K1+w2*K2+...+wn*Kn)
```
clc
clear all
close all
addpath(genpath(pwd))

% sinc funciton
fun = @(x) sin(abs(x))/abs(x);
x = linspace(-10,10, 100);
y = arrayfun(fun, x);
trainData = x';
trainLabel = y';
xt = linspace(-10, 10, 20);
yt = arrayfun(fun, xt);
testData = xt';
testLabel = yt';

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
results = rvm.test(trainData, trainLabel);
rvm.draw(results)
```

### 05. Parameter Optimization for single-kernel-RVM

A demo for RVM model with Parameter Optimization

```
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
kernel = Kernel('type', 'gaussian', 'gamma', 5);

% parameter optimization
opt.method = 'bayes'; % bayes, ga, pso
opt.display = 'on';
opt.iteration = 20;

% parameter
parameter = struct( 'display', 'on',...
                    'type', 'RVC',...
                    'kernelFunc', kernel,...
                    'optimization', opt);
rvm = BaseRVM(parameter);

% RVM model training, testing, and visualization
rvm.train(trainData, trainLabel);
results = rvm.test(trainData, trainLabel);
rvm.draw(results)

```

<p align="center">
  <img width="50%" height="50%" src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/imgs/bayes_1.png">
  <img width="50%" height="50%" src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/imgs/bayes_2.png">
</p>


### 06. Parameter Optimization for hybrid-kernel-RVM

A demo for RVM model with Parameter Optimization

```
lc
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

```

### 07. Cross Validation

In this code, two cross-validation methods are supported: 'K-Folds' and 'Holdout'.
For example, the cross-validation of 5-Folds is
```
parameter = struct( 'display', 'on',...
                    'type', 'RVC',...
                    'kernelFunc', kernel,...
                    'KFold', 5);
```
For example, the cross-validation of the Holdout method with a ratio of 0.3 is 
```
parameter = struct( 'display', 'on',...
                    'type', 'RVC',...
                    'kernelFunc', kernel,...
                    'Holdout', 0.3);
```

### 08. Other option
