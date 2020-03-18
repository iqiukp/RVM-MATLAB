# Relevance Vector Machine (RVM)

[![View Relevance Vector Machine (RVM) on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://ww2.mathworks.cn/matlabcentral/fileexchange/69407-relevance-vector-machine-rvm)

MATLAB Code for prediction based on Relevance Vector Machine (RVM) using 'SB2_Release_200 toolbox'.

Version 1.3, 18-MAR-2020
    
Email: iqiukp@outlook.com

-------------------------------------------------------------------

## Main features

* Easy-used API for training and testing RVM model
* Multiple kinds of kernel functions
* Visualization module 
-------------------------------------------------------------------

## A simple application

```

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

```


<p align="middle">
  <img src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/img/img1.png" width="420">
</p>

<p align="middle">
  <img src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/img/img2.png" width="420">
</p>

<p align="middle">
  <img src="https://github.com/iqiukp/Relevance-Vector-Machine/blob/master/img/img3.png" width="420">
</p>
