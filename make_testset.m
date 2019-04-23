% Description: Implementation to make test data
%
% Copyright @ Jieun Lee
% Laboratory for Imaging Science and Technology
% Seoul National University
% email: jjje0924@gmail.com
%
testset = normalize_data(curve_test);

save('processed_test.mat','-v7.3','testset')