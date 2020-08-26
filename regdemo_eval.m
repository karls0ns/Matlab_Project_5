% nov�rt�sim da��du k�rtu polinomu mode�us

x = dataset(:,1);
y = dataset(:,2);

maxDegree = 7;
% rezerv�jam vektorus mode�u nov�rt�jumiem
result = zeros(maxDegree+1,1);
resultCV = zeros(maxDegree+1,1);

% cikls, lai izveidotu visu k�rtu polinomu mode�us no 0 l�dz maxDegree
for degree = 0 : maxDegree
    % apr��in�m parametrus ar visu datu kopu un nov�rt�jam �o modeli tajos pa�os (apm�c�bas) datos
    a = linreg(x, y, degree);
    yHat = linreg_predict(x, a, degree);
    [~, MAE] = evaluate(y, yHat);
    result(degree+1,:) = MAE;
    
    % nov�rt�jam modeli, izmantojot LOOCV
    [~, MAE] = linreg_loocv(x, y, degree);
    resultCV(degree+1) = MAE;
end

[bestMAE, bestK] = min(resultCV(:,1));
fprintf('Ar linearo regresiju metodi lab�kais LOOCV MAE = %0.3f, kad degree = %d\n', bestMAE, bestK);

% apskat�sim vizu�li, k� mode�a sare���t�ba ietekm� t� k��das nov�rt�jumu

figure;
plot(0:maxDegree, result(:,1), '-');
xlabel('Polinoma k�rta');
ylabel('MAE');
hold on;
grid on;
plot(0:maxDegree, resultCV, '-');
legend({'Apm�c�bas MAE', 'LOOCV MAE'});
xlim([0 maxDegree-1]);
