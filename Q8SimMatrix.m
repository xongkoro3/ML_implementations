% Shuai He
% CSCI5521 Machine Learning
% Question#8 

clc;
clear;
%% Load the data
A = importdata('optdigits.tra');
X = double(A(:,1:64)); 
label = A(:,65);

%% Partition the data by label 
l_0 = find(A(:,65) == 0); m_0 = A(l_0',:);
l_1 = find(A(:,65) == 1); m_1 = A(l_1',:);
l_2 = find(A(:,65) == 2); m_2 = A(l_2',:);
l_3 = find(A(:,65) == 3); m_3 = A(l_3',:);
l_4 = find(A(:,65) == 4); m_4 = A(l_4',:);
l_5 = find(A(:,65) == 5); m_5 = A(l_5',:);
l_6 = find(A(:,65) == 6); m_6 = A(l_6',:);
l_7 = find(A(:,65) == 7); m_7 = A(l_7',:);
l_8 = find(A(:,65) == 8); m_8 = A(l_8',:);
l_9 = find(A(:,65) == 9); m_9 = A(l_9',:);

% Sorted matrix
Mat_sorted = [m_0;m_1;m_2;m_3;m_4;m_5;m_6;m_7;m_8;m_9];
%% Compute similarity matrix for all pairs of opt digits
Mat_similarity = ones(size(X,1),size(X,1));
for r = 1:size(X,1)
    for j = r:size(X,1)
        cbd = 0; %Initialize city block distance
        for dim = 1:size(X,2)
            %Summation over absolute value of difference between vectors
            cbd = cbd + abs(Mat_sorted(r,dim)-Mat_sorted(j,dim));
        end
        cbd = exp(-cbd); %Take exponential
        %Fill the similarity matrix
        Mat_similarity(r,j) = cbd;
        Mat_similarity(j,r) = cbd;
    end
end

