function plotResult(yt, result)
    %{

        DESCRIPTION

        Plot the predicted results

              plotResult(yt, result)

        INPUT
          yt           target samples (n*1)
          result       predicted results

        OUTPUT


        Created on 18th March 2020, by Kepeng Qiu.
        -------------------------------------------------------------

    %}

    figure 
    hold on
    grid on
    % boundary
    index = (1:size(yt, 1))';
    area = [result.ypre(:,1)+2*sqrt(result.yvar(:, 1));...
         flip(result.ypre(:, 1)-2*sqrt(result.yvar(:, 1)), 1)];
    fill([index; flip(index, 1)], area, [234, 234, 242]/255)

    plot(index, yt,...
        'ok:', 'LineWidth',1,...
        'MarkerSize', 5, ... 
        'MarkerEdgeColor', 'k',...
        'MarkerFaceColor', [254, 67, 101]/255)

    plot(index, result.ypre,...
        'ok:','LineWidth', 1,...
        'MarkerSize', 5, ... 
        'MarkerEdgeColor','k',...
        'MarkerFaceColor', [0, 114, 189]/255)

    legend('3¦Ò boundary', 'Real value', 'Predicted value')
    xlabel('Observations');
    ylabel('Predictions');

    figure
    stem(index, result.ypre-yt,...
        'ok-.', 'LineWidth', 1,...
        'MarkerSize', 5, ...
        'MarkerEdgeColor', 'k',...
        'MarkerFaceColor', [254, 67, 101]/255)
    
    xlabel('Observations');
    ylabel('Residuals');
end