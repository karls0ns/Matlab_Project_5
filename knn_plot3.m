function knn_plot3(x, y, k)
% funkcija par�da k-NN regresijas mode�a virsmas att�lu
% (der�ga tikai datiem ar diviem faktoriem, t.i., ja matricai x ir tie�i 2 kolonnas)
	m = size(x,2);
    if (m ~= 2)
        error('knn_plot3: x matricai j�satur tie�i divas kolonnas.');
    end
    x1fit = linspace(min(x(:,1)), max(x(:,1)), 50);
    x2fit = linspace(min(x(:,2)), max(x(:,2)), 50);
    [X1FIT, X2FIT] = meshgrid(x1fit, x2fit);
    xq = [reshape(X1FIT, numel(X1FIT), 1) reshape(X2FIT, numel(X2FIT), 1)];
    yHat = knn_predict(xq, x, y, k);
    yHat = reshape(yHat, size(X1FIT,1), size(X2FIT,2));
    mesh(X1FIT, X2FIT, yHat);
    alpha(0);
    %surf(X1FIT, X2FIT, yHat);
    xlim([min(x(:,1)) max(x(:,1))]);
    ylim([min(x(:,2)) max(x(:,2))]);
    %zlim([min(y) max(y)]);
end
