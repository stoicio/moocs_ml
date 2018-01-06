function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


admitted_student_indices = find(y==1);
rejected_student_indices = find(y==0);


% Plot Examples
plot(X(admitted_student_indices, 1), X(admitted_student_indices, 2),
     'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(rejected_student_indices, 1), X(rejected_student_indices, 2),
     'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

xlabel('Exam 1 Scores');
ylabel('Exam 2 Scores');

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
% =========================================================================



hold off;

end
