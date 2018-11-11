function [y_mu,y_var] = rvm_test(model,X)
% DESCRIPTION
% Prediction based on Relevance Vector Machine (RVM)
% Using SB2_Release_200 toolbox
% http://www.miketipping.com/sparsebayes.htm
%
%       [y_mu,y_var] = rvm_test(model,X)
%
% INPUT
%   X            Test samples (N*d)
%                N: number of samples
%                d: number of features
%   y_mu         Mean of prediction
%   y_var        Variance of prediction
%
% OUTPUT
%   model        RVM model
%
%

L = size(X,1);
BASIS = [ones(L,1),computeKM(X,model.X,model.width)];

% mean of prediction (test samples)
y_mu = BASIS(:,model.rv_index)*model.rv_mu;

% variance of prediction (test samples)
y_var = ones(L,1)*model.beta^-1+ ... 
    diag(BASIS(:,model.rv_index)* ... 
    model.sigma*BASIS(:,model.rv_index)');

end