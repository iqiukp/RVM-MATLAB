function [RMSE, CD, MAE] = computePretIndex(y, ypre)
    %{

        DESCRIPTION

        Compute performance evaluation index

        -------------------------------------------------------------

              [RMSE, CD, MAE] = computePretIndex(y, ypre)

        INPUT
          y            real value
          ypre         predicted value


        OUTPUT
          RMSE         root-mean-square error
          CD           coefficient of determination
          MAE          mean absolute error

        Created on 18th March 2020, by Kepeng Qiu.
        -------------------------------------------------------------%

    %}

    % RMSE
    RMSE= sqrt(sum((y-ypre).^2)/size(y, 1));

    % CD
    SSR = sum((ypre-mean(y)).^2);
    SSE = sum((y-ypre).^2);
    SST = SSR+SSE;
    CD = 1-SSE./SST;

    % MAE
    MAE = sum(abs(y-ypre))/size(y, 1);

end