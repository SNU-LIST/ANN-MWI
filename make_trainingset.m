% Description: Implementation to make training data
%
% Copyright @ Jieun Lee
% Laboratory for Imaging Science and Technology
% Seoul National University
% email: jjje0924@gmail.com
%
%% Normalize the data for training
% Healthy controls
% curve: the multi-echo GRASE data
% d: T2 distribution processed by fitting with stimulated echo correction
for i=1:6
    eval([ 'skcur' num2str(i) '=normalize_data(curve' num2str(i) ');' ...
        'skdis' num2str(i) '=normalize_data(d' num2str(i) ');' ])
end

% MS patients
% mscur: the multi-echo GRASE data
% msd: T2 distribution processed by fitting with stimulated echo correction
for i=1:6
    eval([ 'skmscur' num2str(i) '=normalize_data(mscur' num2str(i) ');' ...
        'skmsdis' num2str(i) '=normalize_data(msd' num2str(i) ');' ])
end

%% Select data
% TE = 10 ms

% Training data, 6 HC and 6 MS
CURVE = cat(2,skcur1,skcur2,skcur3,skcur4,skcur5,skcur6,skmscur1,skmscur2,skmscur3,skmscur4,skmscur5,skmscur6);
DIST = cat(2,skdis1,skdis2,skdis3,skdis4,skdis5,skdis6,skmsdis1,skmsdis2,skmsdis3,skmsdis4,skmsdis5,skmsdis6);

% Validation data, 1 HC and 1 MS
valcur = cat(2,skcur7,skmscur8);
valdis = cat(2,skdis7,skmsdis8);

%% Make data

% training data
[ncur,ndata] = size(CURVE);
[ndis,ndata] = size(DIST);
in_sum = sum(CURVE,1);

nozero = nnz(in_sum);
train = zeros(ncur,nozero);
target = zeros(ndis,nozero);
z = 0;

% Exclude the data with no information
for i=1:ndata
    if in_sum(1,i)~=0
        z = z+1;
        train(:,z) = CURVE(:,i);
        target(:,z) = DIST(:,i);
    end
end

% validation data
[ncur,ndata] = size(valcur);
[ndis,ndata] = size(valdis);
in_sum = sum(valcur,1);

nozero = nnz(in_sum);
val = zeros(ncur,nozero);
val_target = zeros(ndis,nozero);
z = 0;

for i=1:ndata
    if in_sum(1,i)~=0
        z = z+1;
        val(:,z) = valcur(:,i);
        val_target(:,z) = valdis(:,i);
    end
end

train(isnan(train))=0; target(isnan(target))=0; 
val(isnan(val))=0; val_target(isnan(val_target))=0;

save('processed_train.mat','-v7.3','train','target','val','val_target')
