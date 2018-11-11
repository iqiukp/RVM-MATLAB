function [model]= rvm_train(X,Y)
% DESCRIPTION
% Prediction based on Relevance Vector Machine (RVM)
% Using SB2_Release_200 toolbox
% http://www.miketipping.com/sparsebayes.htm
%
%       [model]= rvm_train(X,Y)
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
%


% kernei width
sigma = 5.5;  


L = size(X,1);
%
BASIS = [ones(L,1),computeKM(X,X,sigma)];
%
% SETTINGS = SB2_ParameterSettings('NoiseStd',0.1);
OPTIONS = SB2_UserOptions('diagnosticLevel','medium','monitor',10, ... 
    'diagnosticFile', 'logfile.txt');
[PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ... 
    SparseBayes('Gaussian', BASIS, Y, OPTIONS);

model.rv_index = PARAMETER.Relevant;
model.rv_mu = PARAMETER.Value;
model.width = sigma;
model.X = X;
model.beta = HYPERPARAMETER.beta;
model.sigma = DIAGNOSTIC.Sigma;

% mean of prediction (training samples)
model.y_mu = BASIS(:,model.rv_index)*model.rv_mu;

% variance of prediction (training samples)
model.y_var = ones(L,1)*model.beta^-1+ ... 
    diag(BASIS(:,model.rv_index)* ... 
    model.sigma*BASIS(:,model.rv_index)');

end