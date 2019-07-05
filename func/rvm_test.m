function [y_mu,y_var] = rvm_test(model,X)
% DESCRIPTION
% Prediction based on Relevance Vector Machine (RVM)
% Using SB2_Release_200 toolbox
% http://www.miketipping.com/sparsebayes.htm
%
%       [y_mu,y_var] = rvm_test(model,X)
%
% INPUT
%   X            testing samples (n*d)
%                n: number of samples
%                d: number of features
%   model        RVM model%
%
% OUTPUT
%   y_mu         prediction
%   y_var        variance of prediction
%
%
% Created on 5th July 2019, by Kepeng Qiu.
%-------------------------------------------------------------%

N = size(X,1);
% Compute the kernel matrix
K = computeKM(X,model.X,model.width);

% Construct the basis vectors
if ~model.bias
    % No bias
    BASIS = K;
else
    % Add bias
    BASIS = [K,ones(N,1)];
end

% prediction
y_mu = BASIS(:,model.rv_index)*model.rv_mu;

% variance of prediction
y_var = ones(N,1)*model.beta^-1+ ... 
    diag(BASIS(:,model.rv_index)* ... 
    model.sigma*BASIS(:,model.rv_index)');

end