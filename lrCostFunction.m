function [J, grad] = lrCostFunction(theta, X, y, lambda)
  % Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

J=(-y.*log(sigmoid(X*theta).+eps)-(1.-y).*log(1-sigmoid(X*theta).+eps));

%cost function
J=sum(J)/m+((lambda/(2*m))*sum(theta(2:end).^2));

grad=(1/m)*(X'*(sigmoid(X*theta)-y));%raw gradient i.e not regularized

temp = theta;
% we change the first value to zero so as when it is added with the raw gradient it doesnt make a difference
temp(1) = 0;

%regularized gradient 
grad = grad + lambda / m * temp;

grad = grad(:);
endfunction
