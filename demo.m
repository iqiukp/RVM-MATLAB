% demo
clc
clear all
close all
addpath(genpath(pwd))

% sinc funciton
fun = @(x) sin(abs(x))/abs(x);
% training samples
x = linspace(-10,10,100);
y = arrayfun(fun,x);
X = x';
Y = y';
% test samples
xtest = linspace(-10,10,20);
ytest = arrayfun(fun,xtest);
Xtest = xtest';
Ytest = ytest';

% train RVM model
[model]= rvm_train(X,Y);

% test  RVM model
[y_mu,y_var] = rvm_test(model,Xtest);

%
ypre = y_mu;

% 
figure
plot(X,Y,':o','LineWidth',1.5,'MarkerSize',4)
hold on
plot(X,model.y_mu,':o','LineWidth',1.5,'MarkerSize',4)
plot(X(model.rv_index),Y(model.rv_index),'go', ... 
    'LineWidth',1.5,'MarkerSize',8)

legend('training samples','prediction','relevance Vectors')

figure
xs = (1:size(Xtest,1))';
% 3¦Ò
f1 = [y_mu(:,1)+2*sqrt(y_var(:,1)); flip(y_mu(:,1)-2*sqrt(y_var(:,1)),1)];
fill([xs; flip(xs,1)], f1, [7 7 7]/8)
hold on
plot(Ytest,'b:o','LineWidth',1.5,'MarkerSize',4)
plot(y_mu,'r:o','LineWidth',1.5,'MarkerSize',4)

xlabel('Samples')
ylabel('Values')
legend('3¦Ò boundary','test samples','prediction')




