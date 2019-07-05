function model = rvm_train(X,Y,varargin)
% DESCRIPTION
% Prediction based on Relevance Vector Machine (RVM)
% Using SB2_Release_200 toolbox
% http://www.miketipping.com/sparsebayes.htm
%
%       model = rvm_train(X,Y,varargin)
%
% INPUT
%   X            training samples (N*d)
%                N: number of samples
%                d: number of features
%   Y            target samples (N*1)
%
% OUTPUT
%   model        RVM model
%
% Created on 5th July 2019, by Kepeng Qiu.
%-------------------------------------------------------------%

% Default Parameters setting
width = 3;          % kernel width
bias = 0;           % 0: no free basis (bias)
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
        case 's'
            width = value;
            %
        case 'b'
            bias = value;
    end
end

% 
N = size(X,1);

% Compute the kernel matrix
K = computeKM(X,X,width);

% Construct the basis vectors
if ~bias
    % No bias
    BASIS = K;
    [PARAMETER, HYPERPARAMETER, ~] = ... 
    SparseBayes('Gaussian', BASIS, Y);
else
    % Add bias
    indexBias = N+1;  % index of bias
    BASIS = [K,ones(N,1)];
    OPTIONS = SB2_UserOptions('freeBasis',indexBias);
    [PARAMETER, HYPERPARAMETER, ~] = ... 
    SparseBayes('Gaussian', BASIS, Y,OPTIONS);
end


% RVM model
model.rv_index = PARAMETER.Relevant; % index of relevant vetors
model.bias = bias;                   % bias
if bias
    model.bias_index = indexBias;    % index of bias
    if size(model.rv_index,1) == 1
        if model.rv_index == indexBias
            warning('No relevant vetors were found! Please reduce the value of kernel width.')
        end
    end
end

model.rv_mu = PARAMETER.Value;       % vector of weight values
model.beta = HYPERPARAMETER.beta;    % noise precision
model.alpha = HYPERPARAMETER.Alpha;  % vector of weight precision values
model.width = width;            % kernel width of Gaussian kernel function
model.X = X;                         % training samples

%
PHI = BASIS(:,model.rv_index);
tmp = diag(HYPERPARAMETER.Alpha)+HYPERPARAMETER.beta*(PHI'*PHI);
model.sigma = inv(tmp);              % posterior covariance ¡Æ

end