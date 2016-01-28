% Shuai He
% CSCI5521 Machine Learning
% HW#3 
% Question#3 Back-propagation algorithm implementation

function [] = backProp(H, learning_rate)
%% Load the data
A = importdata('optdigits.tra');
X = double(A(:,1:64));
labels = A(:,65);
%% Parameters
threshold = 10^-6; % Manually set threshold
K = 10; % classes
v = (rand(10,H + 1)-0.5)/100; w = (rand(H, 65)-0.5)/100;
v_update = zeros(10, H+1); w_update = zeros(H, 65);

fprintf('Number of Hidden units:'); disp(H);
fprintf('Learning rate parameter:'); disp(learning_rate);
outputs = zeros(10, 1); y = zeros(10, 1); z = ones(H + 1, 1);
counter = 1; E = 0;
%% Back-propagation Algorithm
while (1)
    E_prev = E;
    for h = 2:(H+1)
        z(h) = 1/(1 + exp(-w(h-1,:) * [1,X(counter,:)]'));
    end
    sum = 0;
    for i = 1:K
        outputs(i) = v(i,:) * z;
        sum = sum + exp(outputs(i));
    end
    for i = 1:K
        y(i) = exp(outputs(i))/sum;
    end
    for i = 1:K
        if (labels(counter) + 1 == i)
            v_update(i,:) = learning_rate * (1 - y(i)) * z';
        else
            v_update(i,:) = learning_rate * (0 - y(i)) * z';
        end
    end
    for h = 1:H
        sum = 0;
        for j = 1:K
            if (labels(counter) + 1 == j)
                sum = sum + (1 - y(j)) * v(j,h);
            else
                sum = sum + (0 - y(j)) * v(j,h);
            end
        end
        w_update(h,:) = learning_rate * sum * z(h+1) * (1-z(h+1)) * [1,X(counter,:)];
    end
    v = v + v_update; % Update
    w = w + w_update;
    for i = 1:K
        if (labels(counter) + 1 == i)
            E = E - log(y(i))/log(2);
        end
    end
    
    % Convergence Method
    e = E/counter; % Use the mean
    e_prev = E_prev/(counter-1);
    delta_e = abs(e - e_prev);
    if (delta_e <= threshold)   % If change in error < threshold, terminate
        break;                          
    end
    counter = counter + 1;
    
end
error_rate = E/(counter - 1);

%% Print to Command Window
fprintf('threshold:');
disp(threshold);
fprintf('The index of row when it starts converge:');
disp(counter);
fprintf('Training Error:');
disp(error_rate);
end

