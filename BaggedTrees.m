function [ oobErr ] = BaggedTrees( X, Y, numbags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function

len = size(X, 1);
prediction = zeros(numbags, len); %keep track of ALL the predictions
bags = zeros(numbags, len); %keep track of who is bagged when

for i = 1:numbags
    
    boot = randsample (len, len, true); %choosing bootstrapped indices
    
    newX = X(boot,:);
    newY = Y(boot);
    
    bag = setdiff(1:len, boot); %keep track of out of bag elements
    bags(i,bag) = 1 ; %matrix with 1 if it is out of bag at in i'th bag
    
    tree = fitctree(newX, newY); %fit tree with boostrapped elements
    prediction(i,:) = predict(tree, X); %prediction matrix
    
end

outofbag = times(bags,prediction); %this is to make the not out of the bag zeros
prediction = sign(sum(outofbag)).';

aoobErr = zeros(1, numbags);

for i = 1:numbags
    %compute oob for number of bags from 1 to numbags
    prediction = sign(sum(outofbag(1:i,:))).';
    Y;
    aoobErr(i) = length(find(prediction ~= Y))/length(Y);
end

plot(1:numbags, aoobErr, 'o')
ylim([0,.25])
ylabel('Misclassification Error')
xlabel('Number of Bags')

oobErr = aoobErr(numbags);

aoobErr

end



