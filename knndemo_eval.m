% novçrtçsim k-NN pie daþâdâm k vçrtîbâm

%x = dataset(:,1);
%y = dataset(:,2);
idx1 = 3;
idx2 = 4; %6
x = mpgX(:,[idx1 idx2]);
y = mpgY;
for j = 1 : size(x,2)
    minX = min(x(:,j));
    range = max(x(:,j)) - minX;
    x(:,j) = (x(:,j) - minX) / range;
end

maxK = 30;
% rezervçti vektori modeïu novçrtçjumiem
result = zeros(maxK,2);
resultCV = zeros(maxK,2);

% cikls, lai izmçìinâtu tuvâko kainiòu skaitu k no 1 lîdz maxK
for k = 1 : maxK
    % aprçíinâm parametrus ar visu datu kopu un novçrtçjam ðo modeli tajos paðos (apmâcîbas) datos
    %a = linreg(x, y, degree);
    %yHat = linreg_predict(x, a, degree);
    yHat = knn_predict(x, x, y, k);
    [~, MAE, ~, ~, ~, R2] = evaluate(y, yHat);
    result(k,:) = [MAE R2];
    
    % novçrtçjam modeli, izmantojot LOOCV
    %[~, MAE, ~, ~, ~, R2] = linreg_loocv(x, y, degree);
    [~, MAE, ~, ~, ~, R2] = knn_loocv(x, y, k);
    resultCV(k,:) = [MAE R2];
end

[bestMAE, bestK] = min(resultCV(:,1));
fprintf('Ar k-NN metodi labâkais LOOCV MAE = %0.3f, kad k = %d\n', bestMAE, bestK);
[bestR2, bestK] = max(resultCV(:,2));
fprintf('Ar k-NN metodi labâkais LOOCV R2 = %0.3f, kad k = %d\n', bestR2, bestK);

% apskatîsim vizuâli, kâ k vçrtîbas izvçle ietekmç tâ kïûdas novçrtçjumu

figure;
plot(1:maxK, result(:,1), '-');
title('k-NN MAE');
xlabel('k');
ylabel('MAE');
hold on;
grid on;
plot(1:maxK, resultCV(:,1), '-');
legend({'Training', 'LOOCV'});
xlim([1 maxK]);

%{
figure;
plot(1:maxK, result(:,2), '-');
title('k-NN R^2');
xlabel('k');
ylabel('R^2');
hold on;
grid on;
plot(1:maxK, resultCV(:,2), '-');
legend({'Training', 'LOOCV'});
xlim([1 maxK]);
%}
