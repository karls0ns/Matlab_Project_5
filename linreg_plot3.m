function linreg_plot3(x, a, degree)
% funkcija par�da line�r�s regresijas mode�a virsmas att�lu
% (der�ga tikai datiem ar diviem faktoriem, t.i., ja matricai z ir tie�i 2 kolonnas)
	m = size(x,2);
    if (m ~= 2)
        error('linreg_plot3: x matricai j�satur tie�i divas kolonnas.');
    end
    x1fit = linspace(min(x(:,1)), max(x(:,1)), 50);
    x2fit = linspace(min(x(:,2)), max(x(:,2)), 50);
    [X1FIT, X2FIT] = meshgrid(x1fit, x2fit);
    xq = [reshape(X1FIT, numel(X1FIT), 1) reshape(X2FIT, numel(X2FIT), 1)];
    yHat = linreg_predict(xq, a, degree);
    yHat = reshape(yHat, size(X1FIT,1), size(X2FIT,2));
    mesh(X1FIT, X2FIT, yHat);
    alpha(0);
    %surf(X1FIT, X2FIT, yHat);
    xlim([min(x(:,1)) max(x(:,1))]);
    ylim([min(x(:,2)) max(x(:,2))]);
    %zlim([min(y) max(y)]);
end
