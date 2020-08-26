function yHat = knn_predict(xq, x, y, k)
% funkcija prognoz� yHat v�rt�bas no xq ieejas v�rt�b�m, izmantojot tuv�ko kaimi�u metodi,
% ja dota apm�c�bas kopa x,y un tuv�ko kaimi�u skaits k
    yHat = zeros(size(xq,1),1);
    for q = 1 : size(xq,1) %q ir rindi?as indekss
        distances = zeros(size(x,1),1);
        xquery = xq(q,:);
        for i = 1: size(x,1)
            % noteikt attalumi no xquery lidz x(i)
            %{
            for j = 1: size(x,2)
                distances(i) = distances(i) + (xquery(1,j) - x(i,j)) ^ 2;
            end
            distances(i) = sqrt(distances(i));
            %}
            distances(i) = norm(xquery - x(i,:));
        end
        [~, idx] = sort(distances);
        idx = idx(1:k);
        yNeigbors = y(idx);
        %yHat(q) = mean(yNeigbors);
        if (distances(idx(1)) == 0)
            yHat(q) = yNeigbors(1);
        else
            weights = 1 ./ distances(idx);
            yHat(q) = sum(yNeigbors .* weights) ./ sum(weights);
        end
    end
end
