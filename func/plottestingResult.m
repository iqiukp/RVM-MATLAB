function plottestingResult(x,y,y_mu,y_var)
% DESCRIPTION
% Plot the testing results 
%
%    plottestingResult(x,y,ypre,y_var)
%
% INPUT
%   x         testing data
%   y         testing data
%   y_mu      prediction
%   y_var     variance of prediction
%
% Created on 5th July 2019, by Kepeng Qiu.
%-------------------------------------------------------------%

%
figure
hold on
grid on

% 3¦Ò boundary
f1 = [y_mu(:,1)+2*sqrt(y_var(:,1)); flip(y_mu(:,1)-2*sqrt(y_var(:,1)),1)];
fill([x; flip(x,1)], f1,[1,0.9,0.9])

% testing samples
plot(x,y,'k:o','LineWidth',1,'MarkerSize',4, ... 
    'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k')

% prediction
plot(x,y_mu,'b:o','LineWidth',1,'MarkerSize',4, ... 
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b')

% axis settings
tgca = 12;  % font size
tfont = 'Helvetica'; % font type
% tfont = 'Arial'; % font type
% set(gca,'yscale','log')
set(gca,'FontSize',tgca,'FontName',tfont)

% legend settings
tlegend = tgca*0.9;
legend({'3¦Ò boundary','testing samples','prediction'},'FontSize',tlegend , ... 
    'FontWeight','normal','FontName',tfont)

% label settings
tlabel = tgca*1.1; 
xlabel('Samples','FontSize',tlabel,'FontWeight','normal', ... 
    'FontName',tfont,'Color','k')
ylabel('Value','FontSize',tlabel,'FontWeight','normal', ... 
    'FontName',tfont,'Color','k')

end
