classdef RvmFunction < handle
    %{
        Version 2.0, 19-APR-2020
        Email: iqiukp@outlook.com
    %}
    
        methods(Static)
            function checkInput(obj)
                obj.model.numKernel = numel(obj.parameter.kernel);
                if ~isfield(obj.parameter, 'weight')
                    obj.parameter.weight = ones(1, obj.model.numKernel); 
                end
                
                if obj.model.numKernel == 1
                    obj.model.kernelType = 'single';
                else
                    obj.model.kernelType = 'hybrid';
                end
                if isfield (obj.parameter, 'optimization')
                    if isfield (obj.parameter.optimization, 'Kfolds')
                        obj.parameter.optimization.validation = 'Kfolds';
                    else
                        obj.parameter.optimization.validation = 'all';
                    end
                end
                obj.model.x = [];
                obj.model.y = [];
                obj.prediction.xt = [];
                obj.prediction.yt = [];
            end
            
            function displayTrain(obj)
                fprintf('\n')
                fprintf('*** RVM model training finished ***\n')
                fprintf('iter           =  %d \n', obj.model.iter);
                fprintf('nRVs           =  %d \n', obj.model.nRVs)
                fprintf('radio of nRVs  =  %.2f%% \n', 100*obj.model.rnRVs)
                fprintf('time cost      =  %.4f s\n', obj.model.timeCost)
                fprintf('training RMSE  =  %.4f\n', obj.model.RMSE)
                fprintf('training CD    =  %.4f\n', obj.model.CD)
                fprintf('training MAE   =  %.4f\n', obj.model.MAE)
                fprintf('\n')
            end
            
            function displayTest(result)
                fprintf('\n')
                fprintf('*** RVM model test finished ***\n')
                fprintf('time cost      =  %.4f s\n', result.timeCost)
                fprintf('predicted RMSE =  %.4f\n', result.RMSE)
                fprintf('predicted CD   =  %.4f\n', result.CD)
                fprintf('predicted MAE  =  %.4f\n', result.MAE)
                fprintf('\n')
            end
            
            function [RMSE, CD, MAE] = getEvaluation(y, ypre)
                RMSE= sqrt(sum((y-ypre).^2)/size(y, 1));
                SSR = sum((ypre-mean(y)).^2);
                SSE = sum((y-ypre).^2);
                SST = SSR+SSE;
                CD = 1-SSE./SST;
                MAE = sum(abs(y-ypre))/size(y, 1);
            end
        end
end

