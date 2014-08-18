function [] = optimizeAutoencoderLBFGS()
% traindata = loadData('F:/RICA-master/MNIST');
path = fullfile('D:','algorithm_learning','sparseae_exercise','contrast','im_input_40by40.mat');
traindata =Im_Input(path);

layersizes = [size(traindata,1) 2500 900 36];

% Record the index that each layer starts at
indx = 1;
for i=1:length(layersizes)-1
    layerinds(i) = indx;
    indx = indx + layersizes(i) * layersizes(i+1);
end
layerinds(length(layersizes)) = indx;

% Weight Initialization
% TODO: May need to add biases back in
for i=1:length(layersizes)-1
    r  = sqrt(6) / sqrt(layersizes(i+1)+layersizes(i));   
    A = rand(layersizes(i+1), layersizes(i))*2*r - r; 
    theta{i} = A;
end
% for i=1:length(layersizes)-1
%     r  = sqrt(6) / sqrt(layersizes(i+1)+layersizes(i));   
%     A = rand(layersizes(i+1), layersizes(i))*2*r - r; 
%     theta{i}= A(:);
% end


% theta = theta';
%initial parameters
% layer1patchsize = 40;
% layer2patchsize =40;
% versionsize1 = size(traindata,1);
% hiddensize1 = 1024;
% versionsize2= hiddensize1*(layer2patchsize-layer1patchsize+1)^2;   
% hiddensize2 = 625 ;
% versionsize3 =  hiddensize2*(layer2patchsize-layer1patchsize+1)^2;
% hiddensize3 = 36;
% layersize = 3;
% theta1 = initializeParameters(hiddensize1,versionsize1);
% layersize_mat=[versionsize1 hiddensize1;versionsize2 hiddensize2;versionsize3 hiddensize3];
% for i=0:layersize-1
%     j=0;
%     theta{i} = initializeParameters(layersize_mat(i,j+1), layersize_mat(i,j));
% end




% Found at http://www.di.ens.fr/~mschmidt/Software/minFunc.html
addpath minFunc/;

options.Method = 'lbfgs'; 
options.maxIter = 20;
options.display = 'on';
options.TolX = 1e-3;

batchSize = 1000;
maxIter = 20;
for layer=1:length(layersizes)-1
    fprintf('Training Layer %i\n',layer);
    for i=1:1
        % Each iteration does a fresh batch looping when data runs out
        startIndex = mod((i-1) * batchSize, size(traindata,2)) + 1;
        fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
%         data = traindata(:, startIndex:startIndex + batchSize-1);
         data = traindata;

        %% Optionally Check the Gradient
%         fastDerivativeCheck(@deepAutoencoder, theta, 1, 2, layersizes, layerinds, data, layer);
        % exit;
%         [theta1,cost1]=deepAutoencoder(theta,layersizes, layerinds, data,layer);
       [theta,cost]= minFunc( @(p) deepAutoencoder(p, ...
                                   layersizes, layerinds, data,layer), ...
                              theta, options);

%         [theta,cost] = minFunc(@deepAutoencoder,theta, ...
%              layer,options);
            
    end
end
visualizeWeights(theta, layersizes, traindata)
%%=====================================
%% result process
fprintf('Training Layer %i\n',layer);
for i=1:1
    % Each iteration does a fresh batch looping when data runs out
    startIndex = mod((i-1) * batchSize, size(traindata,2)) + 1;
    fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
    %         data = traindata(:, startIndex:startIndex + batchSize-1);
    for i=1:10
        k=1;
        train_negative_sample{i} = traindata;
        for j=i:10:n
            train_sample{i}(:,k)=traindata(:,j);
            train_negative_sample{i}(:,k)=[];
            k = k+ 1;
        end
    end
    data = train_sample;
    
    %% Optionally Check the Gradient
    %         fastDerivativeCheck(@deepAutoencoder, theta, 1, 2, layersizes, layerinds, data, layer);
    % exit;
    %         [theta1,cost1]=deepAutoencoder(theta,layersizes, layerinds, data,layer);
    [theta,cost]= minFunc( @(p) deepAutoencoder(p, ...
        layersizes, layerinds, data,layer), ...
        theta, options);
    
    %         [theta,cost] = minFunc(@deepAutoencoder,theta, ...
    %              layer,options);
    
end



path_test = fullfile('D:','algorithm_learning','sparseae_exercise','contrast','im_input_40by40_test.mat');
patches_test = Im_Input(path_test);
display_network(patches_test(:,1:100),12);
patches2_test = W1*patches_test;
patches2_test = patches2_test+repmat(b1,1,size(patches2_test,2));
patches3_test = W2*patches2_test;
patches3_test = patches3_test+repmat(b2,1,size(patches3_test,2));
W3= reshape(opttheta3(1:hiddenSize3*visibleSize3), hiddenSize3, visibleSize3);
b3 = opttheta3(2*hiddenSize3*visibleSize3+1:2*hiddenSize3*visibleSize3+hiddenSize3);
training_data = W3*patches3_test;
training_data = training_data+repmat(b3,1,size(training_data,2));
[R,P,Q] = Precision(training_data);




end


