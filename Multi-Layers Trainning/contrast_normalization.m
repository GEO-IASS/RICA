function result = contrast_normalization(data,dimension)
%dimension �������ɸ�˹�����ά�ȣ�Ҳ�ǿ��ƶԱȹ�һ�� ���ֵ�patch��С��
%data���ݸ�ʽi j k  ; iΪ���ݸ��� ,j*kΪ������Ԫ�����ݷ���
u = dimension/2;
[R,Q,M,N]=size(data);
dim_add =floor(dimension/2);
%��չdata��ά�ȣ�һʵ�ֶԱȹ�һ�������ά����ԭ��ά��һ����
data_temp = zeros(Q,M+dim_add*2,N+dim_add*2);
for i=1:Q
    for j=1:M
        for k=1:N
            data_temp(i,j+dim_add,k+dim_add) = data(1,i,j,k);
        end
    end
end
%���ɸ�˹���󣬲�����Լ����ʹ��ֵΪ1
[X,Y] = meshgrid(1:dimension,1:dimension);
W = exp(-(X-u).^2-(Y-u).^2);
W = W./sum(W(:));
%�����׼��ı���
temp_count=zeros(M,N);
for j=1:M
    for k=1:N
        temp =0;
        for i=1:Q        
            temp1 = data_temp(i,j:j+dimension-1,k:k+dimension-1);
            temp2=temp1(:);
            temp3 = reshape(temp2,dimension,dimension);
            temp=temp+sum(sum(W.*temp3));
        end
        temp_count(j,k)=temp;
    end
end
clear temp1 temp2 temp;
for i=1:Q
    for j=1:M
        for k=1:N
            temp =0;
            %subtractive
            data_temp(i,j+dim_add,k+dim_add) = data_temp(i,j+dim_add,k+dim_add)-temp_count(j,k);
        end
    end
end
for j=1:M
    for k=1:N
        temp =0;
        for i=1:Q     
            temp1 = data_temp(i,j:j+dimension-1,k:k+dimension-1).^2;
            temp2 =temp1(:);
            temp3 = reshape(temp2,dimension,dimension);
            temp=temp+sum(sum((W.*temp3)));
        end
        temp_count(j,k)=sqrt(temp);
    end
end
clear temp1 temp2 temp3 temp;
%�������ڿ��Ʒ�ĸ��ȫ�ֱ�׼��ľ�ֵ
mean_var = mean(temp_count(:));
for i=1:Q
    for j=1:M
        for k=1:N
            if max(mean_var, temp_count(j,k))==0
               data(i,j,k)=data_temp(i,j+dim_add,k+dim_add)/10^-10;
            else
                 data(i,j,k)=data_temp(i,j+dim_add,k+dim_add)/max(mean_var, temp_count(j,k));
        end
    end
end
result= data;

end