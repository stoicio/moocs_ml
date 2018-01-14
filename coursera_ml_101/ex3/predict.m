function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
input_features = [ones(m, 1) X];
layer_1_activations = sigmoid(input_features * Theta1'); %z2

% add bias vector
layer_1_activations = [ones(size(layer_1_activations, 1), 1) layer_1_activations]; 

layer_2_activations = sigmoid(layer_1_activations * Theta2');

[max_probability, max_probability_index] = max(layer_2_activations, [], 2);

p = max_probability_index; 
% =========================================================================


end
