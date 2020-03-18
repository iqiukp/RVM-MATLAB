function result = rvm_test(model, xt, yt)
    %{

        DESCRIPTION

              result = rvm_test(model, xt, yt)

        INPUT
          model        RVM model
          xt           testing samples (n*d)
                       n: number of samples
                       d: number of features
          yt           target samples (n*1)


        OUTPUT
          result       predicted results

        Created on 18th March 2020, by Kepeng Qiu.
        -------------------------------------------------------------

    %}
    if nargin < 3
        warning('Please input the real value of the test data')
    end
    
    
    tic
    nt = size(xt, 1);
    % Compute the kernel matrix
    K = model.kernel.getKernelMatrix(xt, model.rv);

    % Construct the basis vectors
    if ~strcmp(model.option.freeBasis, 'on')
        % No bias
        BASIS = K;
    else
        % Add bias
        BASIS = [K, ones(nt, 1)];
    end
    
    result = struct();
    % prediction
    result.ypre = BASIS*model.mu;

    % variance of prediction
    result.yvar = model.beta^-1+diag(BASIS*model.sigma*BASIS');
    timeCost = toc;
    
    % model evaluation 
    [result.RMSE, result.CD, result.MAE] = computePretIndex(yt, result.ypre);
    
    if strcmp(model.option.display, 'on')
        fprintf('\n')
        fprintf('*** RVM model test finished ***\n')
        fprintf('time cost      =  %.4f s\n', timeCost)
        fprintf('predicted RMSE =  %.4f\n', result.RMSE)
        fprintf('predicted CD   =  %.4f\n', result.CD)
        fprintf('predicted MAE  =  %.4f\n', result.MAE)
        fprintf('\n')
    end
end