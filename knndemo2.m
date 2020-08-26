idx1 = 3;
idx2 = 6; %6
x = mpgX(:,[idx1 idx2]);
y = mpgY;
for j = 1 : size(x,2)
    minX = min(x(:,j));
    range = max(x(:,j)) - minX;
    x(:,j) = (x(:,j) - minX) / range;
end

k = 5;

figure;
plot3(x(:,1), x(:,2), y, '.r', 'MarkerSize', 15);
xlabel(mpgNames{idx1});
ylabel(mpgNames{idx2});
zlabel('MPG');
hold on;
grid on;
knn_plot3(x, y, k);

%a = linreg(x, y, 1);
%linreg_plot3(x, a, 1);
