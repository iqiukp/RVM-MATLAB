function [x, y, xt, yt] = generateData
    %{

        DESCRIPTION

        Generate data using sinc function

        -------------------------------------------------------------

              [x, y, xt, yt] = generateData

        OUTPUT
          x            trainging input
          y            trainging output
          xt           test input
          yt           test output

        Created on 18th March 2020, by Kepeng Qiu.
        -------------------------------------------------------------%

    %}

    snr = 18;
    % sinc funciton
    fun = @(x) awgn(sin(abs(x))/abs(x), snr);
    N = 100;
    Nt = 50;

    % training samples
    x = awgn(linspace(-10, 10, N), snr);
    y = arrayfun(fun, x);
    x = x';
    y = y';

    % test samples
    xt= awgn(linspace(-10, 10, Nt), snr);
    yt = arrayfun(fun, xt);
    xt = xt';
    yt = yt';

end