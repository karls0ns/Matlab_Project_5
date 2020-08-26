% novçrtçsim daþâdu kârtu polinomu modeïus

x = dataset(:,1);
y = dataset(:,2);

maxDegree = 7;
% rezervçjam vektorus modeïu novçrtçjumiem
result = zeros(maxDegree+1,1);
resultCV = zeros(maxDegree+1,1);

% cikls, lai izveidotu visu kârtu polinomu modeïus no 0 lîdz maxDegree
for degree = 0 : maxDegree
    % aprçíinâm parametrus ar visu datu kopu un novçrtçjam ðo modeli tajos paðos (apmâcîbas) datos
    a = linreg(x, y, degree);
    yHat = linreg_predict(x, a, degree);
    [~, MAE] = evaluate(y, yHat);
    result(degree+1,:) = MAE;
    
    % novçrtçjam modeli, izmantojot LOOCV
    [~, MAE] = linreg_loocv(x, y, degree);
    resultCV(degree+1) = MAE;
end

[bestMAE, bestK] = min(resultCV(:,1));
fprintf('Ar linearo regresiju metodi labâkais LOOCV MAE = %0.3f, kad degree = %d\n', bestMAE, bestK);

% apskatîsim vizuâli, kâ modeïa sareþìîtîba ietekmç tâ kïûdas novçrtçjumu

figure;
plot(0:maxDegree, result(:,1), '-');
xlabel('Polinoma kârta');
ylabel('MAE');
hold on;
grid on;
plot(0:maxDegree, resultCV, '-');
legend({'Apmâcîbas MAE', 'LOOCV MAE'});
xlim([0 maxDegree-1]);
