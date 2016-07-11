function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

theta_1 = theta(1);
theta_2 = theta(2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    sum_theta_1 = 0;
    sum_theta_2 = 0;
    for i = 1:m
      h = theta_1 + theta_2 * X(i,1);
      sum_theta_1 = sum_theta_1 + h - y(i);
      sum_theta_2 = sum_theta_2 + (h - y(i)) * X(i,2);
    endfor
    
    theta_1 = theta_1 - alpha * sum_theta_1 / m;
    theta_2 = theta_2 - alpha * sum_theta_2 / m;





    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
    theta(1) = theta_1;
    theta(2) = theta_2;

end

end
