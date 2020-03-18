function model = rvm_train(x, y, varargin)
    %{

        DESCRIPTION

        Prediction based on Relevance Vector Machine (RVM)
        Using SB2_Release_200 toolbox
        -------------------------------------------------------------

              model = rvm_train(X,Y,varargin)

        INPUT
          x            training samples (n*d)
                       n: number of samples
                       d: number of features
          y            target samples (n*1)

        OUTPUT
          model        RVM model

        Created on 18th March 2020, by Kepeng Qiu.
        -------------------------------------------------------------%

    %}

    tic
    % default parameter setting
    kernel = Kernel('type', 'gauss', 'width', sqrt(size(x, 2)));
    option = struct('freeBasis', 'off',...
                    'display', 'on');
    %
    if rem(nargin, 2)
        error('Parameters of rvm_train should be pairs')
    end
    numParameters = nargin/2-1;

    for n =1:numParameters
        parameters = varargin{(n-1)*2+1};
        value	= varargin{(n-1)*2+2};
        switch parameters
                %
            case 'kernel'
                kernel = value;
                %
            case 'option'
                option = value;
        end
    end

    % 
    numSamples = size(x, 1);

    % compute the kernel matrix
    K = kernel.getKernelMatrix(x, x);

    % construct the basis vectors
    if ~strcmp(option.freeBasis, 'on')
        % no bias
        BASIS = K;
        [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ... 
        SparseBayes('Gaussian', BASIS, y);
    else
        % add bias
        indexBias = numSamples+1;  % index of bias
        BASIS = [K, ones(numSamples, 1)];
        [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ... 
        SparseBayes('Gaussian', BASIS, y, SB2_UserOptions('freeBasis', indexBias));
    end
    timeCost = toc;
    
    % RVM model
    model = struct();
    model.index = PARAMETER.Relevant;  % index of BASIS
    model.relevant = model.index; % index of relevance vectors
    if strcmp(option.freeBasis, 'on')
        model.bias = PARAMETER.Value(end); % value of bias
        model.relevant(model.relevant == numSamples+1) = []; 
        if isempty(model.relevant)
            warning('No relevant vetors were found! Please change the values of kernel parameters.')
        end
    else
        model.bias = 0; % value of bias
    end

    model.mu = PARAMETER.Value;          % vector of weight values
    model.beta = HYPERPARAMETER.beta;    % noise precision
    model.alpha = HYPERPARAMETER.Alpha;  % vector of weight precision values
    model.kernel = kernel; % kernel function
    model.PHI = BASIS(:, PARAMETER.Relevant); % Used BASIS
    model.iter = DIAGNOSTIC.iterations; % iteration
    model.nRVs = size(model.relevant, 1);  % number of relevance vectors
    model.option = option; % RVM model option
    model.sigma = inv(diag(HYPERPARAMETER.Alpha)+...
        HYPERPARAMETER.beta*(model.PHI'*model.PHI));% posterior covariance
    model.rv = x(model.relevant, :); % relevance vectors
    
    % model evaluation 
    ypre = model.PHI*model.mu; 
    [model.RMSE, model.CD, model.MAE] = computePretIndex(y, ypre);

    % display
    if strcmp(model.option.display, 'on')
        fprintf('\n')
        fprintf('*** RVM model training finished ***\n')
        fprintf('iter           =  %d \n', model.iter);
        fprintf('nRVs           =  %d \n', model.nRVs)
        fprintf('radio of nRVs  =  %.2f%% \n', 100*model.nRVs/numSamples)
        fprintf('time cost      =  %.4f s\n', timeCost)
        fprintf('training RMSE  =  %.4f\n', model.RMSE)
        fprintf('training CD    =  %.4f\n', model.CD)
        fprintf('training MAE   =  %.4f\n', model.MAE)
        fprintf('\n')
    end
end