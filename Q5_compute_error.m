% CSCI5521 Machine Learning
% Shuai He
% hexxx477@umn.edu

clear;
clc;
%% Load the computed data
load('Q5data.mat');
label_clustered = label1;
%% For loop to assign clustered labels
for i=1:size(label1,1)
     if (label1(i)==0)
        label_clustered(i) = 0;
     elseif (label1(i)==1)
        label_clustered(i) = 4;
     elseif (label1(i)==2)
        label_clustered(i) = 9;
     elseif (label1(i)==3)
        label_clustered(i) = 7;
     elseif (label1(i)==4)
        label_clustered(i) = 2;
     elseif (label1(i)==5)
        label_clustered(i) = 3;
     elseif (label1(i)==6)
        label_clustered(i) = 8;
     elseif (label1(i)==7)
        label_clustered(i) = 1;
     elseif (label1(i)==8)
        label_clustered(i) = 5;
     elseif (label1(i)==9)
        label_clustered(i) = 6;  
     end
end
%% Compute error rate
error_rate = sum(label_clustered ~= labels)/3823;
disp('The error rate is: ');
disp(error_rate);