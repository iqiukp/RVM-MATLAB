classdef RvmOptimization < handle
    %{
        CLASS DESCRIPTION

        Parameter optimization for RVM model.
    
    --------------------------------------------------------------
    %}
    
    methods (Static)
        function getModel(rvm)
            RvmOptimizationOption.checkParameter(rvm);
            objFun = @(parameter) RvmOptimization.getObjValue(parameter, rvm);
            RvmOptimizationOption.constructParamTable(rvm);
            switch rvm.optimization.method
                case 'bayes'
                    RvmOptimization.bayesopt(rvm, objFun);
                case 'ga'
                    RvmOptimization.gaopt(rvm, objFun);
                case 'pso'
                    RvmOptimization.psoopt(rvm, objFun);
            end
            RvmOptimizationOption.setParameter(rvm);
            rvm.getModel;
        end
        
        function bayesopt(rvm, objFun)
            %{
                Optimize the parameters using Bayesian optimization.
                For detailed introduction of the algorithm and parameter
                setting, please enter 'help bayesopt' in the command line.
            %}
            parameter = RvmOptimizationOption.setParameterForBayes(rvm);
            switch rvm.optimization.display
                case 'on'
                    plotFcn_ = {@plotObjectiveModel, @plotMinObjective};
                case 'off'
                    plotFcn_ = [];
            end
            results = bayesopt(objFun, parameter, 'Verbose', 1,...
                'MaxObjectiveEvaluations', rvm.optimization.iteration,...
                'NumSeedPoints', rvm.optimization.point, 'PlotFcn', plotFcn_);
            % optimization results
            [rvm.optimization.bestParam, ~, ~] = bestPoint(results);
        end
        
        function gaopt(rvm, objFun)
            %{
                Optimize the parameters using Genetic Algorithm (GA)
                For detailed introduction of the algorithm and parameter
                setting, please enter 'help ga' in the command line.
            %}
            seedSize = 10*rvm.optimization.numParameter;
            [lowerBound_, upperBound_] = RvmOptimizationOption.setParameterForPsoAndGa(rvm);
            try
                switch rvm.optimization.display
                    case 'on'
                        plotFcn_ = 'gaplotbestf';
                    case 'off'
                        plotFcn_ = [];
                end
                options = optimoptions('ga', 'PopulationSize', seedSize,...
                    'MaxGenerations', rvm.optimization.iteration,...
                    'Display', 'iter', 'PlotFcn', plotFcn_);
                bestParam_ = ga(objFun, rvm.optimization.numParameter, [], [], [], [],...
                    lowerBound_, upperBound_, [], [], options);
            catch % older vision
                switch rvm.optimization.display
                    case 'on'
                        plotFcn_ = @gaplotbestf;
                    case 'off'
                        plotFcn_ = [];
                end
                options = optimoptions('ga', 'PopulationSize', seedSize,...
                    'MaxGenerations', rvm.optimization.iteration,...
                    'Display', 'iter', 'PlotFcn', plotFcn_);
                bestParam_ = ga(objFun, rvm.optimization.numParameter, [], [], [], [],...
                    lowerBound_, upperBound_, [], [], options);
            end
            % optimization results
            rvm.optimization.bestParam.Variables = bestParam_;
        end
        
        function psoopt(rvm, objFun)
            %{
                Optimize the parameters using Particle Swarm Optimization (PSO)
                For detailed introduction of the algorithm and parameter
                setting, please enter 'help particleswarm' in the command line.
            %}
            seedSize = 10*rvm.optimization.numParameter;
            switch rvm.optimization.display
                case 'on'
                    plotFcn_ = 'pswplotbestf';
                case 'off'
                    plotFcn_ = [];
            end
            options = optimoptions('particleswarm', 'SwarmSize', seedSize,...
                'MaxIterations', rvm.optimization.iteration,...
                'Display', 'iter', 'PlotFcn', plotFcn_);
            [lowerBound_, upperBound_] = RvmOptimizationOption.setParameterForPsoAndGa(rvm);
            bestParam_ = particleswarm(objFun, rvm.optimization.numParameter,...
                lowerBound_, upperBound_, options);
            % optimization results
            rvm.optimization.bestParam.Variables = bestParam_;
        end
        
        function objValue = getObjValue(parameter, rvm)
            %{
                Compute the value of objective function
            %}
            rvm_ = copy(rvm);
            rvm_.display = 'off';
            switch class(parameter)
                case 'table' % bayes
                    rvm_.optimization.bestParam = parameter;
                case 'double' % ga, pso
                    rvm_.optimization.bestParam.Variables = parameter;
            end
            % parameter setting
            RvmOptimizationOption.setParameter(rvm_);
            % cross validation
            if strcmp(rvm_.crossValidation.switch, 'on')
                objValue = 1-RvmOptimization.crossvalFunc(rvm_);
            else
                % train with all samples
                rvm_.getModel;
                rvm_.evaluationMode = 'train';
                results_ = test(rvm_, rvm_.data, rvm_.label);
                rvm_.performance = rvm_.evaluateModel(results_);
                switch rvm_.type
                    case 'RVR'
                        objValue = rvm_.performance.RMSE;
                    case 'RVC'
                        objValue = 1-rvm_.performance.accuracy;
                end
            end
        end
        
        function accuracy = crossvalFunc(rvm)
            %{
                Compute the cross validation accuracy
            %}
            rng('default')
            rvm_ = copy(rvm);
            data_ = rvm_.data;
            label_ = rvm_.label;
            rvm_.display = 'off';
            rvm_.evaluationMode = 'train';
            cvIndices = crossvalind(rvm.crossValidation.method, ...
                rvm_.numSamples, rvm.crossValidation.param);
            switch rvm.crossValidation.method
                case 'KFold'
                    accuracy_ = Inf(rvm.crossValidation.param, 1);
                    for i = 1:rvm.crossValidation.param
                        testIndices = (cvIndices == i);
                        testData = data_(testIndices, :);
                        testLabel = label_(testIndices, :);
                        rvm_.data = data_(~testIndices, :);
                        rvm_.label = label_(~testIndices, :);
                        try
                            rvm_.getModel;
                        catch
                            continue
                        end
                        results = rvm_.test(testData, testLabel);
                        switch rvm_.type
                            case 'RVR'
                                accuracy_(i, 1) = results.performance.RMSE;
                            case 'RVC'
                                accuracy_(i, 1) = 1-results.performance.accuracy;
                        end
                        
                    end
                    accuracy_(accuracy_ == Inf) = [];
                    accuracy = mean(accuracy_);
                case 'Holdout'
                    testIndices = (cvIndices == 0);
                    testData = data_(testIndices, :);
                    testLabel = label_(testIndices, :);
                    rvm_.data = data_(~testIndices, :);
                    rvm_.label = label_(~testIndices, :);
                    rvm_.getModel;
                    results = rvm_.test(testData, testLabel);
                    switch rvm_.type
                        case 'RVR'
                            accuracy = results.performance.RMSE;
                        case 'RVC'
                            accuracy = 1-results.performance.accuracy;
                    end
            end
        end
    end
end