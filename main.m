data=load('train.csv');
%save traindata.mat data;

%setting up parameters
input_layer_size=784;
num_labels=10;

%placing the values from the file in the variables
y=data(:,1); %a vector of all the corresponding values of X
X=data(:,2:end);%matrix of 60000 by 784. Values of the grayscale image of the pictures

m = size(X, 1);%number of columns in  X

% Load Training Data
fprintf('Loading and Visualizing Data ...\n');
% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%training
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%predicting o determine accuracy
testData=load('test.csv');
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

