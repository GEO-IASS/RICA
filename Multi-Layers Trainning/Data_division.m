function result = Data_division(data,dim)
%��������ӦΪM*N*layernum
%�������ݸ�ʽΪ{i��j}iΪ�����ָ�������jΪ�������ķָ������
num = size(data,2);
[M,N,K]=size(data{1});
% result{num,floor((M-dim+1)/4)*floor((N-dim+1)/4)}=[];
for i=1:num
    num_flag =1;
    for j=1:4:M-dim+1
        for k=1:4:N-dim+1
            result{i,num_flag}=data{i}(j:j+dim-1,k:k+dim-1,:);
            num_flag =num_flag+1;
        end
    end
end

end