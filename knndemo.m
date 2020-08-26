x = dataset(:,1);
y = dataset(:,2);

figure;
plot(x, y, '.', 'MarkerSize', 20);
xlabel('X');
ylabel('Y');
grid on;
hold on;

% nosak�m attiec�g�s pak�pes polinoma parametrus a
%degree = 2;
%a = linreg(x, y, degree);

k = 1;

% veicam prognozi visos zq punktos, lai uzz�m�tu mode�a grafiku
xq = (min(x):0.01:max(x))';
%yHat = linreg_predict(xq, a, degree);
yHat = knn_predict(xq, x, y, k);
plot(xq, yHat, '-');

% veicam prognozi vien� zq punkt� un uzz�m�jam to tai pa�� att�l�
xq = mean(xq);
%yHat = linreg_predict(xq, a, degree);
yHat = knn_predict(xq, x, y, k);
fprintf('Pie x = %.1f prognoz�tais y ir %.1f\n', xq, yHat);
plot(xq, yHat, '*r');

% mode�a nov�rt��ana apm�c�bas kop�
%yHat = linreg_predict(x, a, degree);
yHat = knn_predict(x, x, y, k);
[SAE, MAE, SSE, MSE, RMSE, R2] = evaluate(y, yHat);
fprintf('SAE=%.2f, MAE=%.2f, SSE=%.2f, MSE=%.2f, RMSE=%.2f, R2=%.2f\n', SAE, MAE, SSE, MSE, RMSE, R2);
