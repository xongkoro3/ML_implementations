% Shuai He
% CSCI5521 Machine Learning
% Question#5 K-Means Clustering Implementation

clc;
clear;
%% Load the data
A = importdata('optdigits.tra');
X = double(A(:,1:64)); 
label = A(:,65);
%% Randomly choose cluster centers
centroids = randi([0 16],10,64); % centroids/cluster centers 
zero_m = zeros(10,64);
%% Repeat till convergence
labels = ones(size(X,1),1);
min = 0; index = 0;
while (norm(centroids - zero_m)) >= 3
    for i=1:size(X,1) % for each data point x_i
        min = 0;
        index = 0;
        for j=1:10
            distance = norm(X(i,:) - centroids(j,:));
            if (distance <= min)
                min = distance;
                index = j - 1;
            end
        end
        labels(i) = index;
    end
    zero_m = labels;
    for j=1:size(centroids,1)% for each cluster
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


