classdef BaseRVM < handle & matlab.mixin.Copyable
    %{
        Version 2.1, 31-AUG-2021
        Email: iqiukp@outlook.com
    %}
    
    properties
        data
        label
        label_
        type = 'RVR' % classification regression
        kernelFunc = Kernel('type', 'gaussian', 'gamma', 0.5)
        kernelFuncName
        kernelType = 'single'
        kernelWeight
        relevanceVectors
        relevanceVectorIndices
        numRelevanceVectors
        numIterations
        numKernel
        rationRelevanceVectors
        alpha
        alphaTolerance = 1e-6
        relevanceVectorAlpha
        freeBasis = 'off'
        runningTime
        maxIter = 1000
        bias
        sigma
        beta
        PHI
        weight
        predictedLabel
        predictedLabel_
        display = 'on'
        optimization
        crossValidation
        performance
        evaluationMode = 'test'
        classNames
    end
    
    properties (Dependent)
        numSamples
        numFeatures
    end
    
    methods
        % create an object of RVM
        function obj = BaseRVM(parameter)
            RvmOption.setParameter(obj, parameter);
        end
        
        % train RVM model
        function varargout = train(obj, data, label) % varargin
            tStart = tic;
            obj.label = label;
            obj.data = data;
            obj.classNames = unique(label);
            if strcmp(obj.type, 'RVC')
                obj.label_ = RvmOption.checkLabel(obj, label);
            else
                obj.label_ = label;
            end
            % parameter optimization
            if strcmp(obj.optimization.switch, 'on')
                RvmOptimization.getModel(obj);
            else
                getModel(obj);
            end
            
            obj.runningTime = toc(tStart);
            % display
            if strcmp(obj.display, 'on')
                RvmOption.displayTrain(obj);
            end
            % output
            if nargout == 1
                varargout{1} = obj;
            end
        end
        
        function getModel(obj)
            switch obj.type
                case 'RVC'
                    likelihoodFunc = 'Bernoulli';
                case 'RVR'
                    likelihoodFunc = 'Gaussian';
            end
            
            switch obj.kernelType
                case 'single'
                    K = obj.kernelFunc.computeMatrix(obj.data, obj.data);
                case 'hybrid'
                    K = 0;
                    for i = 1:obj.numKernel
                        K = K+obj.kernelWeight(i)*obj.kernelFunc(i).computeMatrix(obj.data, obj.data);
                    end
            end
            
            switch obj.freeBasis
                case 'off'
                    % no bias
                    BASIS = K;
                    options_ = SB2_UserOptions('iterations', obj.maxIter);
                case 'on'
                    BASIS = [K, ones(obj.numSamples, 1)];
                    options_ = SB2_UserOptions('freeBasis', obj.numSamples+1, 'iterations', obj.maxIter);
            end
            
            % model
            [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] =...
                SparseBayes(likelihoodFunc, BASIS, obj.label_, options_);
            
            % store the model
            obj.relevanceVectorIndices = PARAMETER.Relevant; % index of relevance vectors
            
            switch obj.freeBasis
                case 'off'
                    obj.bias = 0; % value of bias
                case 'on'
                    obj.bias = PARAMETER.Value(end); % value of bias
                    obj.relevanceVectorIndices(obj.relevanceVectorIndices == obj.numSamples+1) = [];
            end

            obj.weight = PARAMETER.Value;          % vector of weight values
            obj.beta = HYPERPARAMETER.beta;    % noise precision
            obj.alpha = HYPERPARAMETER.Alpha;  % vector of weight precision values
            obj.PHI = BASIS(:, PARAMETER.Relevant); % Used BASIS
            obj.numIterations = DIAGNOSTIC.iterations; % iteration
            obj.numRelevanceVectors = size(obj.relevanceVectorIndices, 1);  % number of relevance vectors
            obj.rationRelevanceVectors = obj.numRelevanceVectors/obj.numSamples;
            obj.relevanceVectors = obj.data(obj.relevanceVectorIndices, :); % relevance vectors
            
            switch obj.type
                case 'RVC'
                    obj.predictedLabel_ = zeros(obj.numSamples, 1);
                    index_ = SB2_Sigmoid(obj.PHI*obj.weight)>0.5;
                    obj.predictedLabel_(index_) = 1;
                    obj.predictedLabel = RvmOption.restoreLabel(obj.classNames, obj.predictedLabel_);
                case 'RVR'
                    obj.sigma = inv(diag(obj.alpha)+obj.beta*(obj.PHI'*obj.PHI));
                    obj.predictedLabel = obj.PHI*obj.weight;
            end
            
            % model evaluation
            display_ = obj.display;
            evaluationMode_ = obj.evaluationMode;
            obj.display = 'off';
            obj.evaluationMode = 'train';
            results_ = test(obj, obj.data, obj.label);
            obj.display = display_;
            obj.evaluationMode = evaluationMode_;
            obj.performance = evaluateModel(obj, results_);
        end
        
        function results = test(obj, data, label)
            tStart = tic;
            results.type = obj.type;
            results.label = label;
            results.data = data;
            results.classNames = unique(label);
            if strcmp(obj.type, 'RVC')
                results.label_ = RvmOption.checkLabel(results, label);
            end
            results.numSamples = size(results.data, 1);
            
            switch obj.kernelType
                case 'single'
                    K = obj.kernelFunc.computeMatrix(results.data, obj.relevanceVectors);
                case 'hybrid'
                    K = 0;
                    for i = 1:obj.numKernel
                        K = K+obj.kernelWeight(i)*...
                            obj.kernelFunc(i).computeMatrix(results.data, obj.relevanceVectors);
                    end
            end
            
            switch obj.freeBasis
                case 'off'
                    BASIS = K;
                case 'on'
                    BASIS = [K, ones(size(results.data, 1), 1)];
            end
            
            switch obj.type
                case 'RVC'
                    results.predictedLabel_ = zeros(size(results.data, 1), 1);
                    index_ = SB2_Sigmoid(BASIS*obj.weight)>0.5;
                    results.predictedLabel_(index_) = 1;
                    results.predictedLabel = RvmOption.restoreLabel...
                        (results.classNames, results.predictedLabel_);
                case 'RVR'
                    results.predictedLabel = BASIS*obj.weight;
                    results.predictedVariance = obj.beta^-1+diag(BASIS*obj.sigma*BASIS');
            end
            results.performance = obj.evaluateModel(results);
            results.runningTime = toc(tStart);
            
            % display
            if strcmp(obj.display, 'on')
                RvmOption.displayTest(results);
            end
        end
        
        function performance = evaluateModel(obj, results)
            switch obj.type
                case 'RVC'
                    performance.accuracy = sum(results.predictedLabel_ == results.label_)/results.numSamples;
                    [performance.confusionMatrix, performance.classOrder] = confusionmat...
                        (results.label, results.predictedLabel);
                case 'RVR'
                    performance.RMSE= sqrt(sum((results.label-results.predictedLabel).^2)/results.numSamples);
                    SSR = sum((results.predictedLabel-mean(results.predictedLabel)).^2);
                    SSE = sum((results.label-results.predictedLabel).^2);
                    SST = SSR+SSE;
                    performance.R2 = 1-SSE./SST;
                    performance.MAE = sum(abs(results.label-results.predictedLabel))/results.numSamples;
            end
        end
        
        function draw(obj, results)
            switch obj.type
                case 'RVR'
                    RvmOption.drawRVR(results);
                case 'RVC'
                    RvmOption.drawRVC(results);
            end
        end
            
        function numSamples = get.numSamples(obj)
            numSamples= size(obj.data, 1);
        end
        
        function numFeatures = get.numFeatures(obj)
            numFeatures= size(obj.data, 2);
        end
    end
end