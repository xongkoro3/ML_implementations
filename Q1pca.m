% Shuai He
% CSCI5521 Machine Learning
% Principle Component Analysis Implementation

clc;
clear;
%% Load the data
A = importdata('optdigits.tra');
X = double(A(:,1:64)); 
label = A(:,65);
%% PCA Algorithm
% Centralling data on origin by subtracting mean of each column
X = bsxfun(@minus, X, mean(X,1));     

% Compute covariance matrix 
%C = (X'*X)./(size(X,1)-1);                 
C = cov(X);
% Now that we have the covariance matrix, we find the eigenvalues D and 
% right eigenvectors V of the covariance matrix by using eig() 
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

%% Calculation of error rate
%err2 = sum(sum(Y .* Y));
error_array = zeros(64,2);
error_array(1:64,1) = (1:64)';
Xm = bsxfun(@minus, X, mean(X));
for i = 1:64
    Z = Xm*V(:,1:i); % back projection
    X_hat = Z*V(:,1:i)'; % reconstruction
    X_hat = bsxfun(@plus, X_hat, mean(X)); 
    Y = X_hat - X;
    err = (norm(Y,'fro'))^2; % error
    error_array(i,2) = err;
end
fprintf('Reconstruction error against number of components retained:\n');
disp(error_array);
% plot error vs. #components
plot(error_array(:,1), error_array(:,2));
title('Reconstruct Error Plot');
xlabel('Number of Components');
ylabel('Reconstruct Error');






