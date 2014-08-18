function [cost,grad] = deepAutoencoder(theta, layersizes, data, top)
W1 = reshape(theta,layersizes(2),layersizes(1));
% handle tied-weight stuff
W2 = W1';
%% Forwards & Backwards Prop

h1 = W1 * data;

h2 = W2 * h1;


%% COMPUTE COST

diff = h2 - data;
M = size(data,2);
dd = data * diff';

lambda = 0.05; % Lambda trades off between sparsity and reconstruction
s = log(cosh(h1));
cost = 1/M * (sum(diff(:).^2) + lambda * sum(s(:)))

%% TODO: The sparsity gradient is only correct for layer 1!!!
Wgrad = 1/M * (2 * W1 * (dd + dd') + lambda * tanh(h1) * data');
grad = Wgrad(:);

end

