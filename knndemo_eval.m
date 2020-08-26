% nov�rt�sim k-NN pie da��d�m k v�rt�b�m

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
% rezerv�ti vektori mode�u nov�rt�jumiem
result = zeros(maxK,2);
resultCV = zeros(maxK,2);

% cikls, lai izm��in�tu tuv�ko kaini�u skaitu k no 1 l�dz maxK
for k = 1 : maxK
    % apr��in�m parametrus ar visu datu kopu un nov�rt�jam �o modeli tajos pa�os (apm�c�bas) datos
    %a = linreg(x, y, degree);
    %yHat = linreg_predict(x, a, degree);
    yHat = knn_predict(x, x, y, k);
    [~, MAE, ~, ~, ~, R2] = evaluate(y, yHat);
    result(k,:) = [MAE R2];
    
    % nov�rt�jam modeli, izmantojot LOOCV
    %[~, MAE, ~, ~, ~, R2] = linreg_loocv(x, y, degree);
    [~, MAE, ~, ~, ~, R2] = knn_loocv(x, y, k);
    resultCV(k,:) = [MAE R2];
end

[bestMAE, bestK] = min(resultCV(:,1));
fprintf('Ar k-NN metodi lab�kais LOOCV MAE = %0.3f, kad k = %d\n', bestMAE, bestK);
[bestR2, bestK] = max(resultCV(:,2));
fprintf('Ar k-NN metodi lab�kais LOOCV R2 = %0.3f, kad k = %d\n', bestR2, bestK);

% apskat�sim vizu�li, k� k v�rt�bas izv�le ietekm� t� k��das nov�rt�jumu

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
