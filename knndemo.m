x = dataset(:,1);
y = dataset(:,2);

figure;
plot(x, y, '.', 'MarkerSize', 20);
xlabel('X');
ylabel('Y');
grid on;
hold on;

% nosakâm attiecîgâs pakâpes polinoma parametrus a
%degree = 2;
%a = linreg(x, y, degree);

k = 1;

% veicam prognozi visos zq punktos, lai uzzîmçtu modeïa grafiku
xq = (min(x):0.01:max(x))';
%yHat = linreg_predict(xq, a, degree);
yHat = knn_predict(xq, x, y, k);
plot(xq, yHat, '-');

% veicam prognozi vienâ zq punktâ un uzzîmçjam to tai paðâ attçlâ
xq = mean(xq);
%yHat = linreg_predict(xq, a, degree);
yHat = knn_predict(xq, x, y, k);
fprintf('Pie x = %.1f prognozçtais y ir %.1f\n', xq, yHat);
plot(xq, yHat, '*r');

% modeïa novçrtçðana apmâcîbas kopâ
%yHat = linreg_predict(x, a, degree);
yHat = knn_predict(x, x, y, k);
[SAE, MAE, SSE, MSE, RMSE, R2] = evaluate(y, yHat);
fprintf('SAE=%.2f, MAE=%.2f, SSE=%.2f, MSE=%.2f, RMSE=%.2f, R2=%.2f\n', SAE, MAE, SSE, MSE, RMSE, R2);
