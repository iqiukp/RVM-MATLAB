classdef RvmOptimization < RVM
    %{
        Version 2.0, 19-APR-2020
        Email: iqiukp@outlook.com
    %}

    methods (Static)
        function rvm = getModel(rvm)
            if strcmp(rvm.parameter.optimization.method, 'ga')
                rvm = RvmOptimization.gaOptimize(rvm);
            end
            if strcmp(rvm.parameter.optimization.method, 'pso')
                rvm = RvmOptimization.psoOptimize(rvm);
            end
            rvm.getModel;
        end
        
        function rvm = gaOptimize(rvm)
            %{
                Optimize the parameters using Genetic Algorithm (GA)
                For detailed introduction of the algorithm and parameter 
                setting, please enter 'help ga' in the command line.
            %}
            
            lb = rvm.parameter.optimization.lb;
            ub = rvm.parameter.optimization.ub;
            nvars = rvm.parameter.optimization.numVariable;
            fun = @RvmOptimization.getObjValue;
            if ~isfield(rvm.parameter.optimization, 'numPopulation')
                rvm.parameter.optimization.numPopulation = 2*nvars;
            end
            options = optimoptions('ga','PopulationSize',rvm.parameter.optimization.numPopulation,...
                'MaxGenerations', rvm.parameter.optimization.maxIter,...
                'Display', 'iter',  'PlotFcn', 'gaplotbestf');
            bestx = ga(fun, nvars, [], [], [], [],lb,ub, [], [], options);
            rvm = RvmOptimization.setParameterValue(rvm, bestx);
        end

        
        function rvm = psoOptimize(rvm)
            %{
                Optimize the parameters using Particle Swarm Optimization (PSO)
                For detailed introduction of the algorithm and parameter 
                setting, please enter 'help particleswarm' in the command line.
            %}
            
            lb = rvm.parameter.optimization.lb;
            ub = rvm.parameter.optimization.ub;
            nvars = rvm.parameter.optimization.numVariable;
            fun = @RvmOptimization.getObjValue;
            if ~isfield(rvm.parameter.optimization, 'numPopulation')
                rvm.parameter.optimization.numPopulation = 2*nvars;
            end
            options = optimoptions('particleswarm','SwarmSize',rvm.parameter.optimization.numPopulation,...
                'MaxIterations', rvm.parameter.optimization.maxIter,...
                'Display', 'iter',  'PlotFcn', 'pswplotbestf');
            bestx = particleswarm(fun, nvars, lb, ub, options);
            rvm = RvmOptimization.setParameterValue(rvm, bestx);
        end

        function objValue = getObjValue(parameter)
            % compute the fitness (here use RMSE)
            
            rvm = evalin('base', 'rvm');
            rvm = RvmOptimization.setParameterValue(rvm, parameter);
            x_ = rvm.model.x;
            y_ = rvm.model.y;
            xt_ = rvm.prediction.xt;
            yt_ = rvm.prediction.yt;
            display_ = rvm.parameter.display;
            rvm.parameter.display = 'off';
            switch rvm.parameter.optimization.validation
                case 'Kfolds'
                    nKfolds = rvm.parameter.optimization.Kfolds;
                    tmp = rvm.model.x;
                    tmpLabel = rvm.model.y;
                    m = size(tmp ,1);
                    indices = crossvalind('Kfold', m, nKfolds);
                    objValue_ = Inf(nKfolds, 1);
                    for i = 1:nKfolds
                        test = (indices==i);
                        xt = tmp(test, :);
                        yt = tmpLabel(test, :);
                        rvm.model.x = tmp(~test, :);
                        rvm.model.y = tmpLabel(~test, :);
                        try
                            rvm.getModel;
                        catch
                            continue
                        end
                        rvm.test(xt, yt);
                        objValue_(i, 1) = rvm.prediction.RMSE;
                    end
                    objValue_(objValue_ == Inf) = [];
                    objValue = mean(objValue_);
                case 'all'
                    rvm.getModel;
                    objValue = rvm.model.RMSE;
            end
            rvm.model.x = x_;
            rvm.model.y = y_;
            rvm.prediction.xt = xt_;
            rvm.prediction.yt = yt_;
            rvm.parameter.display = display_;
        end

        function rvm = setParameterValue(rvm, parameter)
            numParameter = numel(rvm.parameter.optimization.target);
            numKernel = numel(rvm.parameter.kernel);
            k = 0;
            parameterType = cell(1, numParameter);
            for i = 1:numParameter
                parameterType{1, i} = class(rvm.parameter.optimization.target{1,i});
                if strcmp(parameterType{1, i}, 'Kernel')
                    k = k+1;
                    name = fieldnames(rvm.parameter.kernel(1, k).parameter);
                    rvm.parameter.kernel(1, k).parameter.(name{1, 1}) = parameter(1, i);
                end
                if strcmp(parameterType{1, i}, 'char')
                    rvm.parameter.weight = parameter(1, i:i+numKernel-1);
                end
            end
        end
    end
end