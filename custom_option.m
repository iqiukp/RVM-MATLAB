%% custom optimization option
%{      
    opt.method = 'bayes'; % bayes, ga, pso
    opt.display = 'on';
    opt.iteration = 20;
    opt.point = 10;

    % gaussian kernel function
    opt.gaussian.parameterName = {'gamma'};
    opt.gaussian.parameterType = {'real'};
    opt.gaussian.lowerBound = 2^-6;
    opt.gaussian.upperBound = 2^6;

    % laplacian kernel function
    opt.laplacian.parameterName = {'gamma'};
    opt.laplacian.parameterType = {'real'};
    opt.laplacian.lowerBound = 2^-6;
    opt.laplacian.upperBound = 2^6;

    % polynomial kernel function
    opt.polynomial.parameterName = {'gamma'; 'offset'; 'degree'};
    opt.polynomial.parameterType = {'real'; 'real'; 'integer'};
    opt.polynomial.lowerBound = [2^-6; 2^-6; 1];
    opt.polynomial.upperBound = [2^6; 2^6; 7];

    % sigmoid kernel function
    opt.sigmoid.parameterName = {'gamma'; 'offset'};
    opt.sigmoid.parameterType = {'real'; 'real'};
    opt.sigmoid.lowerBound = [2^-6; 2^-6];
    opt.sigmoid.upperBound = [2^6; 2^6];
%}

%% RVM model parameter
%{
    'display'    :   'on', 'off'
    'type'       :   'RVR', 'RVC'
    'kernelFunc' :   kernel function
    'KFolds'     :   cross validation, for example, 5
    'HoldOut'    :   cross validation, for example, 0.3
    'freeBasis'  :   'on', 'off'
    'maxIter'    :   max iteration, for example, 1000
%}
