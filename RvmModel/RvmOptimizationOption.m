classdef RvmOptimizationOption < handle
    %{
        CLASS DESCRIPTION

        Parameter optimization option for RVM model.
    
    ---------------------------------------------------------------

    %}
    
    methods (Static)
        function checkParameter(rvm)
            backup_ = rvm.optimization;
            % gaussian
            gaussian.parameterName = {'gamma'};
            gaussian.parameterType = {'real'};
            gaussian.lowerBound = 2^-6;
            gaussian.upperBound = 2^6;
            
            % laplacian
            laplacian.parameterName = {'gamma'};
            laplacian.parameterType = {'real'};
            laplacian.lowerBound = 2^-6;
            laplacian.upperBound = 2^6;
            
            % polynomial
            polynomial.parameterName = {'gamma'; 'offset'; 'degree'};
            polynomial.parameterType = {'real'; 'real'; 'integer'};
            polynomial.lowerBound = [2^-6; 2^-6; 1];
            polynomial.upperBound = [2^6; 2^6; 7];
            
            % sigmoid
            sigmoid.parameterName = {'gamma'; 'offset'};
            sigmoid.parameterType = {'real'; 'real'};
            sigmoid.lowerBound = [2^-6; 2^-6];
            sigmoid.upperBound = [2^6; 2^6];
            
            % others
            rvm.optimization.method = 'bayes'; % bayes, ga, pso
            rvm.optimization.display = 'on';
            rvm.optimization.iteration = 30;
            rvm.optimization.point = 10;
            
            % parameter initialization
            name_ = fieldnames(backup_);
            count_ = 0;
            %
            for i = 1:rvm.numKernel
                if ~isfield(backup_, rvm.kernelFuncName{i})
                    rvm.optimization.(rvm.kernelFuncName{i}) = eval(rvm.kernelFuncName{i});
                else
                    tmp_ = rvm.optimization.(rvm.kernelFuncName{i});
                    rvm.optimization.(name_{i}) = RvmOptimizationOption.setKernelFuncParam(rvm.kernelFuncName{i}, gaussian, tmp_);
                end
                count_ = count_+length(rvm.optimization.(rvm.kernelFuncName{i}).parameterName);
            end
            
            for i = 1:size(name_, 1)
                if isfield(rvm.optimization, name_{i})
                    rvm.optimization.(name_{i}) = backup_.(name_{i});
                end
            end
            
            if strcmp(rvm.kernelType, 'hybrid')
                tmp_ = cell(rvm.numKernel, 1);
                tmp_(:) = {'real'};
                rvm.optimization.weight.parameterType = tmp_(:);
                rvm.optimization.weight.lowerBound = zeros(rvm.numKernel, 1);
                rvm.optimization.weight.upperBound = ones(rvm.numKernel, 1);
                count_ = count_+rvm.numKernel;
            end
            
            % 
            rvm.optimization.numParameter = count_;
            if isfield(rvm.optimization, 'polynomial')
                if ~strcmp(rvm.optimization.method, 'bayes')
                    errorText = strcat(...
                        'The parameter of the polynomial kernel funcion should be optimized by Bayesian optimization.\n',...
                        'For examples, the optimization method should be set as follows:\n',...
                        '--------------------------------------------\n',...
                        'opt.method = ''bayes''\n');
                    error(sprintf(errorText))
                end
            end
        end
        
        function constructParamTable(rvm)
            name_ = cell(1, rvm.optimization.numParameter);
            count_ = 0;
            for i = 1:rvm.numKernel
                for j = 1:length(rvm.optimization.(rvm.kernelFuncName{i, 1}).parameterName)
                    count_ = count_+1;
                    name_{count_} = [rvm.kernelFuncName{i, 1},'_', rvm.optimization.(rvm.kernelFuncName{i, 1}).parameterName{j}];
                end
            end
            
            if strcmp(rvm.kernelType, 'hybrid')
                for i = 1:rvm.numKernel
                    count_ = count_+1;
                    name_{count_} = [rvm.kernelFuncName{i, 1}, '_', 'weight'];
                end
            end
            rvm.optimization.bestParam = array2table(zeros(1, rvm.optimization.numParameter));
            rvm.optimization.bestParam.Properties.VariableNames = name_;
        end
        
        function setParameter(rvm)
            for i = 1:rvm.numKernel
                switch rvm.kernelFuncName{i}
                    case 'gaussian'
                        rvm.kernelFunc(i).gamma = rvm.optimization.bestParam.gaussian_gamma;
                    case 'laplacian'
                        rvm.kernelFunc(i).gamma = rvm.optimization.bestParam.laplacian_gamma;
                    case 'polynomial'
                        rvm.kernelFunc(i).gamma = rvm.optimization.bestParam.polynomial_gamma;
                        rvm.kernelFunc(i).offset = rvm.optimization.bestParam.polynomial_offset;
                        rvm.kernelFunc(i).degree = rvm.optimization.bestParam.polynomial_degree;
                    case 'sigmoid'
                        rvm.kernelFunc(i).gamma = rvm.optimization.bestParam.sigmoid_gamma;
                        rvm.kernelFunc(i).offset = rvm.optimization.bestParam.sigmoid_offset;
                end
            end
            
            if strcmp(rvm.kernelType, 'hybrid')
                for i = 1:rvm.numKernel
                    rvm.kernelWeight(i) = rvm.optimization.bestParam.([rvm.kernelFuncName{i}, '_weight']);
                end
                rvm.kernelWeight = rvm.kernelWeight/sum(rvm.kernelWeight);
                for i = 1:rvm.numKernel
                    rvm.optimization.bestParam.([rvm.kernelFuncName{i}, '_weight']) = rvm.kernelWeight(i);
                end
            end
        end
        
        function parameter = setParameterForBayes(rvm)
            parameter = [];
            for i = 1:rvm.numKernel
                for j = 1: length(rvm.optimization.(rvm.kernelFuncName{i, 1}).parameterName)
                    name_ = [rvm.kernelFuncName{i, 1},'_', rvm.optimization.(rvm.kernelFuncName{i, 1}).parameterName{j}];
                    tmp_ = optimizableVariable(name_,...
                        [rvm.optimization.(rvm.kernelFuncName{i, 1}).lowerBound(j)...
                        rvm.optimization.(rvm.kernelFuncName{i, 1}).upperBound(j)],...
                        'Type', rvm.optimization.(rvm.kernelFuncName{i, 1}).parameterType{j});
                    parameter = [parameter; tmp_];
                end
            end
            if strcmp(rvm.kernelType, 'hybrid')
                for i = 1:rvm.numKernel
                    name_ = [rvm.kernelFuncName{i, 1}, '_', 'weight'];
                    tmp_ = optimizableVariable(name_,...
                        [rvm.optimization.weight.lowerBound(i)...
                        rvm.optimization.weight.upperBound(i)],...
                        'Type', rvm.optimization.weight.parameterType{i});
                    parameter = [parameter; tmp_];
                end
            end
        end
        
        function [lowerBound_, upperBound_] = setParameterForPsoAndGa(rvm)
            lowerBound_ = [];
            for i = 1:rvm.numKernel
                for j = 1: length(rvm.optimization.(rvm.kernelFuncName{i, 1}).parameterName)
                    tmp_ = rvm.optimization.(rvm.kernelFuncName{i, 1}).lowerBound(j);
                    lowerBound_ = [lowerBound_; tmp_];
                end
            end
            if strcmp(rvm.kernelType, 'hybrid')
                for i = 1:rvm.numKernel
                    tmp_ = rvm.optimization.weight.lowerBound(i);
                    lowerBound_ = [lowerBound_; tmp_];
                end
            end
            
            upperBound_ = [];
            for i = 1:rvm.numKernel
                for j = 1: length(rvm.optimization.(rvm.kernelFuncName{i, 1}).parameterName)
                    tmp_ = rvm.optimization.(rvm.kernelFuncName{i, 1}).upperBound(j);
                    upperBound_ = [upperBound_; tmp_];
                end
            end
            if strcmp(rvm.kernelType, 'hybrid')
                for i = 1:rvm.numKernel
                    tmp_ = rvm.optimization.weight.upperBound(i);
                    upperBound_ = [upperBound_; tmp_];
                end
            end
        end
        
        function defaultParam = setKernelFuncParam(paramType, defaultParam, inputParam)
            names = fieldnames(inputParam);
            lab = isfield(defaultParam, names);
            if ~all(lab(:) == 1)
                RvmOptimizationOption.showErrorText(paramType);
            end
            for i = 1:numel(names)
                defaultParam.(names{i}) = inputParam.(names{i});
            end
        end

        function showErrorText(paramType)
            switch paramType
                case 'gaussian'
                    errorText = strcat(...
                        'Incorrected optimization parameters of the gaussian kernel function.\n',...
                        'For examples, the parameters should be set as follows:\n',...
                        '--------------------------------------------\n',...
                        'opt.gaussian.parameterName = {''gamma''}\n',...
                        'opt.gaussian.parameterType = {''real''}\n',...
                        'opt.gaussian.lowerBound = 2^-6\n',...
                        'opt.gaussian.upperBound = 2^6\n');
                    
                case 'laplacian'
                    errorText = strcat(...
                        'Incorrected optimization parameters of the laplacian kernel function.\n',...
                        'For examples, the parameters should be set as follows:\n',...
                        '--------------------------------------------\n',...
                        'opt.laplacian.parameterName = {''gamma''}\n',...
                        'opt.laplacian.parameterType = {''real''}\n',...
                        'opt.laplacian.lowerBound = 2^-6\n',...
                        'opt.laplacian.upperBound = 2^6\n');
                    
                case 'polynomial'
                    errorText = strcat(...
                        'Incorrected optimization parameters of the polynomial kernel function.\n',...
                        'For examples, the parameters should be set as follows:\n',...
                        '--------------------------------------------\n',...
                        'opt.polynomial.parameterName = {''gamma''; ''offset''; ''degree''}\n',...
                        'opt.polynomial.parameterType = {''real''; ''real''; ''integer''}\n',...
                        'opt.polynomial.lowerBound = [2^-6; 2^-6; 1]\n',...
                        'opt.polynomial.upperBound = [2^6; 2^6; 7]\n');
                    
                case 'sigmoid'
                    errorText = strcat(...
                        'Incorrected optimization parameters of the sigmoid kernel function.\n',...
                        'For examples, the parameters should be set as follows:\n',...
                        '--------------------------------------------\n',...
                        'opt.sigmoid.parameterName = {''gamma''; ''offset''}\n',...
                        'opt.sigmoid.parameterType = {''real''; ''real''}\n',...
                        'opt.sigmoid.lowerBound = [2^-6; 2^-6]\n',...
                        'opt.sigmoid.upperBound = [2^6; 2^6]\n');
            end
            error(sprintf(errorText))
        end
    end
end