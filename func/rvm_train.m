function [model]= rvm_train(X,Y,varargin)
% DESCRIPTION
% Prediction based on Relevance Vector Machine (RVM)
% Using SB2_Release_200 toolbox
% http://www.miketipping.com/sparsebayes.htm
%
%       [model]= rvm_train(X,Y,varargin)
%
% INPUT
%   X            Training samples (N*d)
%                N: number of samples
%                d: number of features
%   Y            Target samples (N*1)
%
% OUTPUT
%   model        RVM model
%
% Created on 11st May 2019, by Kepeng Qiu.
%-------------------------------------------------------------%

% Default Parameters setting
sigma = 3;       % kernel width
bias = 0;        % 0: no FreeBasis' --- bias
                 % 1: add bias
%
if rem(nargin,2)
    error('Parameters to rvm_train should be pairs')
end
numParameters = nargin/2-1;

for n =1:numParameters
    Parameters = varargin{(n-1)*2+1};
    value	= varargin{(n-1)*2+2};
    switch Parameters
            %
        case 'sigma'
            sigma = value;
            %
        case 'bias'
            bias = value;
    end
end

% 
N = size(X,1);

% Compute the kernel matrix
K = computeKM(X,X,sigma);

% Construct the basis vectors
if ~bias
    % No bias
    BASIS = K;
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ... 
    SparseBayes('Gaussian', BASIS, Y);
else
    % Add bias
    indexBias = N+1;
    BASIS = [K,ones(N,1)];
    OPTIONS = SB2_UserOptions('freeBasis',indexBias);
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ... 
    SparseBayes('Gaussian', BASIS, Y,OPTIONS);
end

% RVM model
model.rv_index = PARAMETER.Relevant;
model.rv_mu = PARAMETER.Value;
model.width = sigma;
model.X = X;
model.beta = HYPERPARAMETER.beta;
model.sigma = DIAGNOSTIC.Sigma;
model.bias = PARAMETER.Bias;

% prediction
model.y_mu = BASIS(:,model.rv_index)*model.rv_mu+model.bias;

% variance of prediction (training samples)
model.y_var = ones(N,1)*model.beta^-1+ ... 
    diag(BASIS(:,model.rv_index)* ... 
    model.sigma*BASIS(:,model.rv_index)');

end