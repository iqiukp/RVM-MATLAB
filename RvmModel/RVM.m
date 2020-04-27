classdef RVM < handle
    %{
        Version 2.0, 28-APR-2020
        Email: iqiukp@outlook.com
    %}

    properties
        parameter                        % RVM parameter
        model                            % RVM model
        prediction                       % prediciton results
    end
        
    methods
        function obj = RVM(parameter)
            obj.parameter = parameter;
            RvmFunction.checkInput(obj)
        end
        
        function train(obj, x, y)
            %{
                DESCRIPTION
                Train RVM model

                PARAMETER
                  x              sample data (input, n*d)
                                    n: number of samples
                                    d: number of features
            

                  y              sample data (ouput, n*1)
                                    n: number of samples
            %}
            obj.model.x = x;
            obj.model.y = y;
            if isfield(obj.parameter, 'optimization')
                RvmOptimization.getModel(obj);
            else
                getModel(obj);
            end
        end
            
        function getModel(obj)
            tic
            numSamples = size(obj.model.x, 1);
            % compute kernel matrix
            K = zeros(numSamples, numSamples);
            switch obj.model.kernelType
                case 'single'
                    K = obj.parameter.weight*obj.parameter.kernel.getKernelMatrix(obj.model.x, obj.model.x);
                case 'hybrid'
                    for i = 1:obj.model.numKernel
                        K = K+obj.parameter.weight(i)*obj.parameter.kernel(i).getKernelMatrix(obj.model.x, obj.model.x);
                    end
            end

            % train RVM
            switch obj.parameter.freeBasis
                case 'off'
                % no bias
                BASIS = K;
                [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] =...
                    SparseBayes('Gaussian', BASIS, obj.model.y, SB2_UserOptions('iterations', obj.parameter.maxIter));
                
                case 'on'
                % add bias
                indexBias = numSamples+1;  % index of bias
                BASIS = [K, ones(numSamples, 1)];
                [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] =...
                    SparseBayes('Gaussian', BASIS, obj.model.y, SB2_UserOptions('freeBasis', indexBias, 'iterations', obj.parameter.maxIter));
            end
            
            % store the model
            obj.model.relevanceIndex = PARAMETER.Relevant; % index of relevance vectors
            
            switch obj.parameter.freeBasis
                case 'off'
                    obj.model.bias = 0; % value of bias
                case 'on'
                    obj.model.bias = PARAMETER.Value(end); % value of bias
                    obj.model.relevanceIndex(obj.model.relevanceIndex == numSamples+1) = [];
            end
            
            obj.model.mu = PARAMETER.Value;          % vector of weight values
            obj.model.beta = HYPERPARAMETER.beta;    % noise precision
            obj.model.alpha = HYPERPARAMETER.Alpha;  % vector of weight precision values
            obj.model.kernel = obj.parameter.kernel; % kernel function
            obj.model.PHI = BASIS(:, PARAMETER.Relevant); % Used BASIS
            obj.model.iter = DIAGNOSTIC.iterations; % iteration
            obj.model.nRVs = size(obj.model.relevanceIndex, 1);  % number of relevance vectors
            obj.model.rnRVs = obj.model.nRVs/size(obj.model.x, 1);
            obj.model.sigma = inv(diag(HYPERPARAMETER.Alpha)+HYPERPARAMETER.beta*(obj.model.PHI'*obj.model.PHI));% posterior covariance
            obj.model.relevanceVector = obj.model.x(obj.model.relevanceIndex, :); % relevance vectors
            obj.model.timeCost = toc;
            
            % model evaluation
            obj.model.ypre = obj.model.PHI*obj.model.mu;
            [obj.model.RMSE, obj.model.CD, obj.model.MAE] = RvmFunction.getEvaluation(obj.model.y, obj.model.ypre);
            
            % display
            if strcmp(obj.parameter.display, 'on')
                RvmFunction.displayTrain(obj)
            end
        end

        function test(obj, xt, yt)
            %{
                DESCRIPTION
                Test RVM model

                PARAMETER
                  xt              sample data (input, n*d)
                                    n: number of samples
                                    d: number of features
            

                  yt              sample data (ouput, n*1)
                                    n: number of samples
            %}
            
            tic
            obj.prediction.xt = xt;
            obj.prediction.yt = yt;
            nt = size(obj.prediction.xt, 1);
            % Compute the kernel matrix
            K = zeros(nt, size(obj.model.relevanceVector, 1));
            
            switch obj.model.kernelType
                case 'single'
                    K = obj.parameter.weight*obj.parameter.kernel.getKernelMatrix(obj.prediction.xt, obj.model.relevanceVector);
                case 'hybrid'
                    for i = 1:obj.model.numKernel
                        K = K+obj.parameter.weight(i)*obj.parameter.kernel(i).getKernelMatrix(obj.prediction.xt, obj.model.relevanceVector);
                    end
            end
            
            switch obj.parameter.freeBasis
                case 'off'
                BASIS = K;
                case 'on'
                BASIS = [K, ones(nt, 1)];
            end
            
            % prediction
            obj.prediction.ypre = BASIS*obj.model.mu;
            
            % variance of prediction
            obj.prediction.yvar = obj.model.beta^-1+diag(BASIS*obj.model.sigma*BASIS');
            obj.prediction.timeCost = toc;
            
            % model evaluation
            [obj.prediction.RMSE, obj.prediction.CD, obj.prediction.MAE] = RvmFunction.getEvaluation(obj.prediction.yt, obj.prediction.ypre);
            
            if strcmp(obj.parameter.display, 'on')
                RvmFunction.displayTest(obj.prediction)
            end
        end
%         
            function draw(obj)
                figure
                set(gcf,'position',[300 150 600 400])
                hold on
                grid on
                % boundary
                index = (1:size(obj.prediction.yt, 1))';
                area_color = [229, 235, 245]/255;
                area = [obj.prediction.ypre(:,1)+2*sqrt(obj.prediction.yvar(:, 1));...
                    flip(obj.prediction.ypre(:, 1)-2*sqrt(obj.prediction.yvar(:, 1)), 1)];
                fill([index; flip(index, 1)], area, area_color, 'EdgeColor', area_color)
                
                plot(index, obj.prediction.yt,...
                    '-', 'LineWidth',1.5, 'color', [254, 67, 101]/255)                
                plot(index, obj.prediction.ypre,...
                    '-','LineWidth', 1.5,'color', [0, 114, 189]/255)
            
                legend('3σ boundary', 'Real value', 'Predicted value')
                xlabel('Observations');
                ylabel('Predictions');

            end
    end
end


