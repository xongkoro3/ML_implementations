% Shuai He
% CSCI5521 Machine Learning
% Question#3 MDS standardized Euclidean Distances 

clc;
clear;
%% Load the data
A = importdata('optdigits.tra');
X = double(A(:,1:64)); 
label = A(:,65);
%% MDS Algorithm
% Centralling data on origin by subtracting mean of each column
X = bsxfun(@minus, X, mean(X,1));     

% Compute correlation matrix 
C = corrcoef(X);
% Replace all NaN values in matrix with 0
C(isnan(C)) = 0;
% eigenvalue and eigenvector
[V, D] = eig(C);
% The eigenvector with the highest eigenvalue is the direction that has the
% highest variance. The order is sorted in a descending pattern
[D, order] = sort(diag(D), 'descend');
V = V(:,order);

%% Find number of components to capture 95% of variance 
p_v = cumsum(D)/sum(D);
comp = find(p_v >= 0.95);
n_comp = comp(1);
fprintf('Up to the %dth components, 95 percent variance explained\n',n_comp);

Xm = bsxfun(@minus, X, mean(X));
for i = 1:comp(1)
    Z = Xm*V(:,1:i); % MDS result
end






