function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
if (0)
c_candidates = [0.01 0.03 0.1 0.3 1 3 10];
sigma_candidates = [0.01 0.03 0.1 0.3 1 3 10];

min_error = 1000000;
for c = 1:length(c_candidates)
    for s = 1:length(sigma_candidates)
        model = svmTrain(X, y, c_candidates(c), @(x1, x2) gaussianKernel(x1, x2, sigma_candidates(s)));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        if (err < min_error)
            min_error = err;
            C = c_candidates(c);
            sigma = sigma_candidates(s);
            fprintf('params %f, %f, %f', min_error, C, sigma);
        endif
    endfor
endfor
fprintf('params %f, %f, %f', min_error, C, sigma);
endif
% =========================================================================

end
