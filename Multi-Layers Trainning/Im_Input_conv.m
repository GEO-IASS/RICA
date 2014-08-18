function patches =Im_Input_conv(path)
images = load(path);
[M,N]=size(images.im_input);%data cell 1*1500  
[m,n]=size(images.im_input{1,1});%data dimension 40*40
data_dim=m*n;
temp_data= zeros(25,(m-5+1)*(n-5+1))*N;
row_value =1;
for i=1:500
    for j=1:m-5+1
        for k=1:n-5+1
            temp = images.im_input{1,i}(j:j+4,k:k+4);
            if sum(temp(:))~=0
                temp_data(:,row_value)=temp(:);
                row_value = row_value+1;
            end
        end
    end
end
patches = temp_data(:,1:row_value-1);
% %πÈ“ªªØ
% max_value = max(max(patches));
% min_value = min(min(patches));
% patches = (patches-min_value)./(max_value - min_value);

end