function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% a = X(1:10,1:10)
% b = all_theta(:,1:10)

% ps = sigmoid(a*b)

% Tengo 5000 ejemplos de numeros escritos, cada uno tiene 400 datos para evuluar,
% tengo los 10 numeros con las 10 filas de theta, que corresponden a cada numoer,
% entonces, multiplico X con cada theta de cada numero, es como preguntar a ejemplo, vos 
% el 1, sos el 2 y asi.... entoncs elq que mayor porcentaje de acierto de, ese es,
% ahora, como la posicion del mayor pocentaje se corresponde con la posicion de cada 
% columna, esa es mi prediccion Y

ps = sigmoid(X*all_theta');
[p_max, i_max]=max(ps, [], 2);
p = i_max;





% =========================================================================


end
