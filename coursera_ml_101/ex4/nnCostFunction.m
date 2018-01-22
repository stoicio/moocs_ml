function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

## START FORWARD PROPAGATION##
input_layer = [ones(m, 1), X]; % Add bias column. Technically its layer a1
hidden_layer_hypothesis = input_layer * Theta1'; % Z2
hidden_layer_activations = sigmoid(hidden_layer_hypothesis); % Layer a2

% Add Bias column to hidden layer
hidden_layer_bias = ones(size(hidden_layer_activations,1),1); 
hidden_layer_activations = [hidden_layer_bias hidden_layer_activations];

output_layer_hypothesis = hidden_layer_activations * Theta2'; % Layer Z3
output_activations = sigmoid(output_layer_hypothesis); % hThetaX

## END FORWARD PROPAGATION ##

## START COST FUNCTION ##

y_label_matrix = zeros(m, num_labels);

for i = 1:num_labels
  this_label_indices = (y == i);
  y_label_matrix(this_label_indices, i) = 1;
end

first_term = -1 * y_label_matrix .* log(output_activations);
second_term = (1 - y_label_matrix) .* log( 1 - output_activations);

% Need a scalar cost. Sum across example, and sum across all labels
J = (1/m) * sum(sum(first_term - second_term));

% Compute regularization term
theta1_sum = sum(sum(Theta1(:, 2:end).^2));
theta2_sum = sum(sum(Theta2(:, 2:end).^2));
regularization_term = (lambda / (2 * m)) * ( theta1_sum + theta2_sum);
J = J + regularization_term;
## END COST FUNCTION ##


## START BACK PROPAGATION
for i = 1:m % For each example in training set
  % FORWARD PROPAGATION
  input_layer = [1; X(i, :)']; %a1
  hidden_layer_hypothesis = Theta1 * input_layer; %z2
  hidden_layer_activation = [1; sigmoid(hidden_layer_hypothesis)]; %a2
  output_layer_hypothesis = Theta2 * hidden_layer_activation; %z3
  output_activation = sigmoid(output_layer_hypothesis); % a3
  
  actual_output = zeros(num_labels,1);
  actual_output(y(i)) = 1;
  % FORWARD PROPAGATION ENDS
  
  % Calculate how much error each layer contributed to the final output
  output_error = output_activation - actual_output; %delta3
  
  error_due_to_hidden_layer = (Theta2' * output_error) .* [1; sigmoidGradient(hidden_layer_hypothesis)];
  error_due_to_hidden_layer = error_due_to_hidden_layer(2:end);
  % Accumulate Gradient
  % grad = grad + delta2 * a1'
  Theta1_grad = Theta1_grad + error_due_to_hidden_layer * input_layer'; 
  % grad = grad + delta3 * a2'
  Theta2_grad = Theta2_grad + output_error * hidden_layer_activation'; 
  
end

% Take average of all errors - Unregularized Gradient
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Regularized Gradient 
theta_1_bias_replacement = zeros(size(Theta1, 1), 1);
theta_2_bias_replacement = zeros(size(Theta2, 1), 1);

Theta1_grad = Theta1_grad + (lambda/m) * [theta_1_bias_replacement Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda/m) * [theta_2_bias_replacement Theta2(:, 2:end)];
## END BACK PROPAGATION


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
