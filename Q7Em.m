% CSCI5521 Machine Learning
% Shuai He
% hexxx477@umn.edu

clear;
clc;
%% Load data
load('data.mat');
%% initialize EM algorithm
[label1, m]  = Q7Kmeans(X); 
% m: estimates of cluster centers
% label1: lables estimated
PI = zeros(1,10);
h = zeros(3823, 10); %initialize h
for i = 1:3823
    for j = 1:10
        if (label1(i) == (j-1))
            h(i,j) = 1;
        end
    end
end
%% initialization of PI
for i = 1:3823
    PI = PI + h(i,:);
end
PI = PI/3823;
%% initialization of s2
s2 = 0;
for i = 1:3823
    for j = 1:10
        s2 = s2 + h(i,j) * (norm(X(i,:)-m(j,:)))^2;
    end
end
s2 = s2/(3823*64);
m1 = zeros(10,64);
e = 5;
%% While loop
while (sum(sum(abs(m-m1))) >= e)
    sum(sum(abs(m-m1)))
    %update h
    for t = 1:3823
        temp = zeros(1,10);
        for i = 1:10
            temp(i) = PI(i)*exp(-norm(X(t,:)-m(i,:))^2/(2*s2));
        end
        summation = sum(temp);
        temp = temp/summation;
        h(t,:) = temp;
    end
    %update PI
    PI = zeros(1,10);
    for i = 1:3823
        PI = PI + h(i,:);
    end
    PI = PI/3823;
    %save old m
    m1 = m;
    %update m
    for i= 1:10
        count = 0;
        temp = zeros(1,64);
        for j = 1:3823
            count = count + h(j, i);
            temp = temp + h(j,i)*X(j,:);
        end
        if (count ~= 0)
            temp = temp/count;
        end
        m(i,:) = temp;
    end
    %update s2
    s2 = 0;
    for i = 1:3823
        for j = 1:10
            s2 = s2 + h(i,j) * (norm(X(i,:)-m(j,:)))^2;
        end
    end
    s2 = s2/(3823*64);
end
%% Save data
%save P7data m s2 h  
