% Shuai He
% CSCI5521 Machine Learning
% Question#11 Nearest neighbor classification

function [] = Q11KNN()
clc;
clear;
%% Load the data
A = importdata('optdigits.tra');
X = double(A(:,1:64)); 
%% Parameters
labeled_data_1000 = A(1:1000,:); validate_1000 = A(1001:3823,:);
labeled_data_2000 = A(1:2000,:); validate_2000 = A(2001:3823,:);
labeled_data_3000 = A(1:3000,:); validate_3000 = A(3001:3823,:);
%% Calling functions
e1 = knn(labeled_data_1000,validate_1000,10);
e2 = knn(labeled_data_2000,validate_2000,10);
e3 = knn(labeled_data_3000,validate_3000,10);
e4 = knn(labeled_data_1000,validate_1000,100);
e5 = knn(labeled_data_2000,validate_2000,100);
e6 = knn(labeled_data_3000,validate_3000,100);


    %% KNN Alogrithm
    function err = knn(l_data, v_data, k)
        l = size(l_data,1);
        r = size(v_data,1);
        % Calculate the distance 
        distance = pdist2(v_data, l_data, 'euclidean');
        [distance, idx] = sort(distance,2,'ascend'); % Indices of nearest neighbors
        distance = distance(:,1:k);
        idx = idx(:,1:k);
        labels = idx;
        for i=1:r
            for j=1:k
                labels(i,j) = l_data(idx(i,j),65);
            end
        end
        %% Comparison of predicted label and real label
        l_val_mat = zeros(r,3);
        l_val_mat(:,1) = [1:r]';
        l_val_mat(:,2) = mode(labels,2); %predicted label
        l_val_mat(:,3) = v_data(:,65); %real label

        error = sum(l_val_mat(:,2) ~= l_val_mat(:,3))/r;
        % disp(l_val_mat);
        fprintf('The error rate for k = %d on %d training set is %d\n',k,l,error);
    end

end




