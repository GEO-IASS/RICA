function [] = visualizeWeights(theta, layersizes, data)
%% VISUALIZEWEIGHTS This function shows the weights for each hidden neuron
l = length(layersizes);
lnew = 0;
transform = eye(layersizes(1));
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    W = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
    
    % Transform the weights back into the input space
    transform = W * transform;
    
    % Normalize the weights to [0,1]
    maxN = max(transform(:));
    minN = min(transform(:));
    normalized = 1-(transform - minN)./(maxN-minN);
    
    patches_size=12;
    toShow = zeros(patches_size+2, (patches_size+2)*layersizes(i+1));
    showedge1=zeros(1,patches_size);
    showedge2=zeros(1,patches_size+2);
    for j=0:layersizes(i+1)-1
        toShow(:, j*(patches_size+2)+1:(j+1)*(patches_size+2)) =[showedge2;[showedge1;reshape(normalized(j+1,:), patches_size, patches_size);showedge1]';showedge2];
    end
    showimages=zeros((patches_size+2)*sqrt(layersizes(i+1)));
    for j=0:sqrt(layersizes(i+1))-1
         for k=0:sqrt(layersizes(i+1))-1
            showimages(j*(patches_size+2)+1:(j+1)*(patches_size+2),:)=toShow(:,j*sqrt(layersizes(i+1))*(patches_size+2)+1:(j+1)*sqrt(layersizes(i+1))*(patches_size+2));
         end
    end
    filename = strcat('images/layer', num2str(i), 'optimal.png');
    mkdir('images') 
    imwrite(showimages,filename)
end

end

