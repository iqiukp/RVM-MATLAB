classdef RvmOption < handle
    %{
        CLASS DESCRIPTION

        Option of RVM model
    
    -----------------------------------------------------------------
    %}
    
    methods(Static) 
        function label_ = checkLabel(obj, label)
            label_ = zeros(length(label), 1);
            if length(obj.classNames) ~= 2
                error('RVM is only supported for binary classification.')
            end
            switch class(obj.classNames)
                case 'double'
                    label_(label==obj.classNames(1)) = 0;
                    label_(label==obj.classNames(1)) = 1;
                case 'cell'
                    label_(strcmp(label, obj.classNames{1})) = 0;
                    label_(strcmp(label, obj.classNames{2})) = 1;
            end
        end
        
        function label_ = restoreLabel(names, label)
            numLabel = length(label);
            switch class(names)
                case 'double'
                    label_ = zeros(numLabel, 1);
                    label_(label==0) = names(1);
                    label_(label==1) = names(2);
                case 'cell'
                    label_ = cell(numLabel, 1);
                    for i = 1:numLabel
                        if label(i) == 0
                            label_{i} = names{1};
                        else
                            label_{i} = names{2};
                        end
                    end
            end
        end
        
        function drawRVR(results)
            figure
            set(gcf, 'position', [300 150 600 400])
            hold on
            grid on
            % boundary
            index = (1:size(results.label, 1))';
            area_color = [229, 235, 245]/255;
            area = [results.predictedLabel(:,1)+2*sqrt(results.predictedVariance(:, 1));...
                flip(results.predictedLabel(:, 1)-2*sqrt(results.predictedVariance(:, 1)), 1)];
            fill([index; flip(index, 1)], area, area_color, 'EdgeColor', area_color)
            plot(index, results.label,...
                '-', 'LineWidth', 2, 'color', [254, 67, 101]/255)
            plot(index, results.predictedLabel,...
                '-','LineWidth', 2,'color', [0, 114, 189]/255)
            legend('3¦Ò boundary', 'Real value', 'Predicted value')
            xlabel('Observations');
            ylabel('Predictions');
        end
        
        function drawRVC(results)
            titleStr = ['Classification based on RVM (Accuracy: ',...
                sprintf('%.2f', 100*results.performance.accuracy), '%)'];
            figure
            set(gcf, 'position', [300 150 600 400])
            hold on
            grid on
            index = (1:length(results.label))';
            plot(index, results.label_,...
                'LineWidth', 2, 'Color', [0, 114, 189]/255,...
                'LineStyle', '-')
            plot(index, results.predictedLabel_,...
                'LineWidth', 2, 'Color', [162, 20, 47]/255,...
                'LineStyle', ':')
            legend('Real class', 'Predicted class')
            xlabel('Observations');
            ylabel('Predictions');
            ylim([-0.5, 1.5])
            yticks([0 1])
            yticklabels(results.classNames)
            title(titleStr)
            
            try
                figure
                set(gcf, 'position', [300 150 600 400])
                confusionchart(results.performance.confusionMatrix, results.performance.classOrder,...
                    'Title', titleStr,...
                    'RowSummary','row-normalized', ...
                    'ColumnSummary','column-normalized');
            catch
                fprintf('Confusion matrix chart is not effective with the versions lower than R2018b.')
                close
            end
            
        end
        
        function setParameter(obj, parameter)
            version_ = version('-release');
            year_ = str2double(version_(1:4));
            if year_ < 2016 || (year_ == 2016 && version_(5) == 'a')
                error('RVM V2.1 is not compatible with the versions lower than R2016b.')
            end
            %
            obj.crossValidation.switch = 'off'; 
            obj.optimization.switch = 'off';
            name_ = fieldnames(parameter);
            for i = 1:size(name_, 1)
                switch name_{i}
                    case {'HoldOut', 'KFold'}
                        obj.crossValidation.switch = 'on';
                        obj.crossValidation.method = name_{i, 1};
                        obj.crossValidation.param = parameter.(name_{i, 1});
                    case {'optimization'}
                        obj.(name_{i}) = parameter.(name_{i, 1});
                        obj.(name_{i}).switch = 'on';
                    otherwise
                        obj.(name_{i, 1}) = parameter.(name_{i, 1});
                end
            end
            obj.numKernel = numel(obj.kernelFunc);
            for i = 1:obj.numKernel
                obj.kernelFuncName{i, 1} = obj.kernelFunc(i).type;
            end
            if obj.numKernel == 1
                obj.kernelType = 'single';
                obj.kernelFuncName{1} = obj.kernelFunc.type;
            else
                obj.kernelType = 'hybrid';
                for i = 1:obj.numKernel
                    obj.kernelFuncName{i, 1} = obj.kernelFunc(i).type;
                end
            end
            if isempty(obj.kernelWeight)
                obj.kernelWeight = 1/obj.numKernel*ones(obj.numKernel, 1);
            end
        end
        
        function displayTrain(obj)
            fprintf('\n')
            switch obj.type
                case 'RVR'
                    fprintf('*** RVM model (regression) train finished ***\n')
                case 'RVC'
                    fprintf('*** RVM model (classification) train finished ***\n')
            end
            fprintf('running time            = %.4f seconds\n', obj.runningTime)
            fprintf('iterations              = %d \n', obj.numIterations)
            fprintf('number of samples       = %d \n', obj.numSamples)
            fprintf('number of RVs           = %d \n', obj.numRelevanceVectors)
            fprintf('ratio of RVs            = %.4f%% \n', 100*obj.numRelevanceVectors/obj.numSamples)
            switch obj.type
                case 'RVR'
                    fprintf('RMSE                    = %.4f\n', obj.performance.RMSE)
                    fprintf('R2                      = %.4f\n', obj.performance.R2)
                    fprintf('MAE                     = %.4f\n', obj.performance.MAE)
                case 'RVC'
                    fprintf('accuracy                = %.4f%%\n', 100*obj.performance.accuracy)
            end
            if strcmp(obj.optimization.switch, 'on')
                fprintf('Optimized parameter')
                display(obj.optimization.bestParam)
                fprintf('\n')
            end
            if obj.numRelevanceVectors/obj.numSamples > 0.5
                warning('The trained RVM model may be overfitting.')
            end
            fprintf('\n')
        end

        function displayTest(obj)
            fprintf('\n')
            switch obj.type
                case 'RVR'
                    fprintf('*** RVM model (regression) test finished ***\n')
                case 'RVC'
                    fprintf('*** RVM model (classification) test finished ***\n')
            end
            fprintf('running time            = %.4f seconds\n', obj.runningTime)
            fprintf('number of samples       = %d \n', obj.numSamples)
            switch obj.type
                case 'RVR'
                    fprintf('RMSE                    = %.4f\n', obj.performance.RMSE)
                    fprintf('R2                      = %.4f\n', obj.performance.R2)
                    fprintf('MAE                     = %.4f\n', obj.performance.MAE)
                case 'RVC'
                    fprintf('accuracy                = %.4f%%\n', 100*obj.performance.accuracy)
            end
            fprintf('\n')
        end
    end
end