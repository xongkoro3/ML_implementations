% Shuai He
% CSCI5521 Machine Learning
% Question#2 Multicategory Linear Discriminant Analysis (LDA)
clc;
clear;
%% Load data
A = importdata('optdigits.tra');
X = double(A(:,1:64)); 
label = A(:,65);

dimension = size(X,2);
labels = unique(label);
C = length(labels);
Sw = zeros(dimension,dimension);
Sb = zeros(dimension,dimension);
mu = mean(X);

%% LDA implementation
for i = 1:C
    Xi = X(find(label == labels(i)),:);
    n = size(Xi,1);
    mu_i = mean(Xi);
    XMi = bsxfun(@minus, Xi, mu_i);
    Sw = Sw + (XMi' * XMi );
    MiM =  mu_i - mu;
    Sb = Sb + n * MiM' * MiM;
end

Co = inv(Sw+1e-6*eye(64,64))*Sb;
[W_lda, D] = eig(Co); % eigen analysis of (Sw^-1)*Sb
[D, i] = sort(diag(D), 'descend');
W_lda = W_lda(:,i);
X_proj = X*W_lda; %projection
Vector_a = W_lda(:,1);
Vector_b = W_lda(:,2);
w = horzcat(Vector_a, Vector_b);

% Find labeled row vectors, back projection
X_0 = find(A(:,65) == 0); class0 = X(X_0,:); c0 = w'*class0';
X_1 = find(A(:,65) == 1); class1 = X(X_1,:); c1 = w'*class1';
X_2 = find(A(:,65) == 2); class2 = X(X_2,:); c2 = w'*class2';
X_3 = find(A(:,65) == 3); class3 = X(X_3,:); c3 = w'*class3';
X_4 = find(A(:,65) == 4); class4 = X(X_4,:); c4 = w'*class4';
X_5 = find(A(:,65) == 5); class5 = X(X_5,:); c5 = w'*class5';
X_6 = find(A(:,65) == 6); class6 = X(X_6,:); c6 = w'*class6';
X_7 = find(A(:,65) == 7); class7 = X(X_7,:); c7 = w'*class7';
X_8 = find(A(:,65) == 8); class8 = X(X_8,:); c8 = w'*class8';
X_9 = find(A(:,65) == 9); class9 = X(X_9,:); c9 = w'*class9';

%% Plot the data
scatter(c0(1,:),c0(2,:),'magenta');
hold on;
scatter(c1(1,:),c1(2,:),'yellow');
scatter(c2(1,:),c2(2,:),'cyan');
scatter(c3(1,:),c3(2,:),'red');
scatter(c4(1,:),c4(2,:),'green');
scatter(c5(1,:),c5(2,:),'blue');
scatter(c6(1,:),c6(2,:),'black');
scatter(c7(1,:),c7(2,:),'yellow');
scatter(c8(1,:),c8(2,:),'magenta');
scatter(c9(1,:),c9(2,:),'cyan');

