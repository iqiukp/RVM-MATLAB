function plotRelevanceVector(model, y)
    %{

        DESCRIPTION

        Plot the relevance vectors

              plotRelevanceVector(model, y)

        INPUT
          model        RVM model
          yt           target samples (n*1)


        OUTPUT

        Created on 18th March 2020, by Kepeng Qiu.
        -------------------------------------------------------------

    %}

    figure
    hold on
    grid on

    n = size(y, 1);
    index = 1:n;

    plot(index, y,...
        'ok-', 'LineWidth',1,...
        'MarkerSize', 5, ... 
        'MarkerEdgeColor', 'k',...
        'MarkerFaceColor', [255, 90, 95]/255)

    plot(index(model.relevant), y(model.relevant, :),...
        'ok','LineWidth', 1,...
        'MarkerSize', 7, ...
        'MarkerEdgeColor', 'k',...
        'MarkerFaceColor', [0, 114, 189]/255)

    legend('Training data', 'Relevance vectors')
    xlabel('Observations');
    ylabel('Values');

end
