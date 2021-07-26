function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  
  % Some useful variables
  m = size(X, 1); %number of rows
  n = size(X, 2);% number of columns
  
  % You need to return the following variables correctly 
  all_theta = zeros(num_labels, n + 1);

  X=[ones(m,1) X];
  
  % Set Initial theta
  initial_theta = zeros(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 400);

%finding the parameters of each num_labels
for i=1:num_labels,
  all_theta(i,:)=fminunc(@(t)(lrCostFunction(t,X,(y==i),lambda)),initial_theta,options);
endfor

  
endfunction
