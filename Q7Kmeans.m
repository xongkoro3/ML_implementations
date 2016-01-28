% CSCI5521 Machine Learning
% Shuai He
% hexxx477@umn.edu
%% K Means
function [label, m]  = Q7Kmeans(X)
label = zeros(3823,1);
m = randi(16,10,64);
m1 = zeros(10,64);
e = 10;
while (sum(sum(abs(m-m1))) >= e)
    for i = 1:3823
        min = Inf;
        index = 0;
        for j = 1:10
            temp = norm(X(i,:) - m(j,:));
            if (temp <= min)
                min = temp;
                index = j - 1;
            end
        end
        label(i) = index;
    end
    m1 = m;
    for i = 1:10
        count = 0;
        temp = zeros(1,64);
        for j = 1:3823
            if (label(j) == (i-1))
                count = count + 1;
                temp = temp + X(j,:);
            end
        end
        if (count ~= 0)
            temp = temp/count;
        end
        m(i,:) = temp;
    end
end
end