function [RMSE,CD,MAE] = computePretIndex(y,ypre)
% DESCRIPTION
% Compute regression performance evaluation index
%
%    [x, y, xt, yt] = generateData
%
% INPUT
%   y         true value
%   ypre      predicted value
%
%
% OUTPUT
%   RMSE      root-mean-square error
%   CD        coefficient of determination
%   MAE       mean absolute error
%
% Created on 5th July 2019, by Kepeng Qiu.
%-------------------------------------------------------------%

% RMSE
RMSE= sqrt(sum((y-ypre).^2)/size(y,1));

% CD
SSR = sum((ypre-mean(y)).^2);
SSE = sum((y-ypre).^2);
SST = SSR+SSE;
CD = 1-SSE./SST;

% MAE
MAE = sum(abs(y-ypre))/size(y,1);

end