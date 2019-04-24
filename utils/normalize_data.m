function [norm_data] = normalize_data(data)
%
% [norm_data] = normalize_data(data)
%
% Description:
%   Normalize T2 decay curve or T2 distribution
%   to fed into the network (ANN) as an input or a label.
%   For training, put the data after BET.
%   For test, put the whole brain data.
%
% Inputs:
%   data: original 4-D array data 
%         T2 decay curve (row, col, slice, echo numbers (32))
%         or T2 distribution (row, col, slice, log-scaled T2 times (120))
%
% Ouputs:
%   norm_data: normalized data for training or test
%
% Copyright @ Jieun Lee
% Laboratory for Imaging Science and Technology
% Seoul National University
% email: jjje0924@gmail.com
%

[y,x,z,n] = size(data);

% [T2 decay curve] Set the first echo to 1
if n==32  % echo numbers
    curve = data;
    cnorm = curve(:,:,:,:) ./ curve(:,:,:,1);
    c_temp = reshape(cnorm,y*x*z,n); 
    c_temp = permute(c_temp,[2 1]);
    c_temp(isnan(c_temp)) = 0; c_temp(isinf(c_temp)) = 0;
    norm_data = c_temp;

% [T2 distribution] Set the area to 15
elseif n==120 % log-scaled T2 times
    distribution = data;
    dsum = sum(distribution,4); dsum(isnan(dsum)) = 0;
    dnorm = distribution .* (15 ./ dsum);
    dnorm(isnan(dnorm)) = 0;
    d_temp = reshape(dnorm,y*x*z,n); d_temp = permute(d_temp,[2 1]); 
    norm_data = d_temp;
end

end
