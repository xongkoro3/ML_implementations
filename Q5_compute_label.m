% CSCI5521 Machine Learning
% Shuai He
% hexxx477@umn.edu
clear;
clc;
%% Load the data
load('data.mat');
%% Calling K-means function to compute labels and error rate
label1 = Q5_Kmeans(X);
