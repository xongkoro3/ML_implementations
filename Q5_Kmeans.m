% CSCI5521 Machine Learning
% Shuai He
% hexxx477@umn.edu

%% K-Means Implementation
function label = Q5_Kmeans(X)
label = zeros(3823,1);
centeroids = randi(16,10,64);
zero_m = zeros(10,64);
err = 2;
%% While loop repeat till convergence
while (sum(sum(abs(centeroids-zero_m))) >= err)
    for i = 1:3823
        min = Inf;
        index = 0;
        for j = 1:10
            temp = norm(X(i,:) - centeroids(j,:));
            if (temp <= min)
                min = temp;
                index = j - 1;
            end
        end
        label(i) = index;
    end
    zero_m = centeroids;
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
        centeroids(i,:) = temp;
    end
end
end
        
        
        
        
        
        
        
        
        
        