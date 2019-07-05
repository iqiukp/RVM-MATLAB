function [x, y, xt, yt] = generateData
% DESCRIPTION
% Generate data using sinc function
%
%    [x, y, xt, yt] = generateData
%
% OUTPUT
%   x         trainging data (input)
%   y         trainging data (output)
%   xt        testing data (input)
%   yt        testing data (output)
%
% Created on 5th July 2019, by Kepeng Qiu.
%-------------------------------------------------------------%

% sinc funciton
fun = @(x) sin(abs(x))/abs(x);
N = 100;
Nt = 50;

% training samples
x = linspace(-10,10,N);
y = arrayfun(fun,x);
x = x';
y = y';

% testing samples
xt = linspace(-10,10,Nt);
yt = arrayfun(fun,xt);
xt = xt';
yt = yt';

end