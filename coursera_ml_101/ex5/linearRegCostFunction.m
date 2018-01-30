function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X * theta;
total_cost = (hypothesis - y) .^ 2;
J = 0.5 * mean(total_cost);
grad = (X' * (hypothesis - y)) / m;

theta_copy = [0; theta(2:length(theta))];
J = J + (0.5 * lambda * ( 1 / m ) * sum(theta_copy .^ 2));
grad = grad .+ ( (lambda / m ) * theta_copy);

% =========================================================================

grad = grad(:);

end
