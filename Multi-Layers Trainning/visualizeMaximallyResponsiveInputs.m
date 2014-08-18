function [] = visualizeMaximallyResponsiveInputs(theta, layersizes, data)
%VISUALIZEMAXIMALLYRESPONSIVEINPUTS Shows the n inputs that create maximal
%responses from output neuron n
layersizes = [size(data,1) layersizes];
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    W{i} = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
end
% handle tied-weight stuff
j = 1;
for i=l:2*(l-1)
    W{i} = W{l - j}';
    j = j + 1;
end
assert(lnew == length(theta), 'Error: dimensions of theta and layersizes do not match\n')

%% FORWARD PROP
for i=1:2*(l-1)
    if i==1
        h{i} = W{i} * data;
    else
        h{i} = W{i} * h{i-1};
    end
end

n = 25;
for i=0:layersizes(2)-1
    [~, sortIndex] = sort(h{1}(i+1,:),'descend');
    s = zeros(28*sqrt(n));
    for j=0:n-1
        col = mod(j,sqrt(n));
        row = floor(j/sqrt(n));
        s(row*28+1:row*28+28,col*28+1:col*28+28) = reshape(data(:,sortIndex(j+1)),28,28);
    end
    filename = strcat('images/n',num2str(i+1),'MaxResp.png');
    imwrite(s,filename);
end

end

