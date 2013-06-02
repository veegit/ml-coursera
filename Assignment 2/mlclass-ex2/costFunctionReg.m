function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hx = sigmoid(X*theta);
% Below is not a correct approach, since we need to do sum (100 elements after 
% implementing the CF). Below we are doing sum(100 elements after partial CF)
% and then doing sum(100 elements after another partial CF) which is wrong.
% J =  (-y' * log(hx)) - ((1 - y)' * log(1 .- hx)) / m;
reg_factor = (lambda/2/m) * ( (theta' * theta)-theta(1,1) ^ 2 );
J = ( sum( (-y .* log(hx) ) - ( (1-y) .* log(1 - hx) ) ) / m) + reg_factor;

reg_grad_factor = (lambda/m) * theta;
reg_grad_factor(1,1) = 0;
temp = ( ( (hx-y)' * X)' * 1 / m) + reg_grad_factor;
grad = temp;




% =============================================================

end
