function [data_result,data_result_conv]=layer_conv_max_contrast(data,W_old,layer_i)
%������һ�ε���������Ԥ����----������Ԫ�����maxpooling��contrast normalization
%dataΪԭʼ����minist 40*40
%data������λ����������ά�ȣ�����λ��������W_old������λ����ά�ȣ�����Ϊ��Ԫ����
%ѵ�����ݵ�����ά��Ԥ����Ϊ5*5
patches_dim = 10;
%maxpooling�����С Ԥ����2*2
maxpool_size =4;
%contrast normalization dimension Ԥ����Ϊ5*5
contrast_dim =4;

if layer_i==1
    [m,n]=size(data);
    [M,N] =size(W_old);
    data_simple_dim = sqrt(m);
    W_dim = sqrt(N);
    %data��Ϊ���󣬴�Ϊ��ά
    data_temp = zeros(n,data_simple_dim,data_simple_dim);
    for i=1:n
        temp = data(:,i);
        data_temp(i,:,:)=reshape(temp,data_simple_dim,data_simple_dim);
    end
    clear temp;
    %W��Ϊ����
    W_temp = zeros(M,W_dim,W_dim);
    for i=1:M
        temp = W_old(i,:);
        W_temp(i,:,:)=reshape(temp,patches_dim,patches_dim);
    end
    clear temp;
    %�������
    %     data_result = zeros(n,M,floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1,floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1,patches_dim,patches_dim);
    temp_conv = zeros(data_simple_dim-patches_dim+1);
    temp_pool = zeros(floor((data_simple_dim-patches_dim+1)/maxpool_size));
    %     temp_contrast = zeros(floor((data_simple_dim-patches_dim+1)/maxpool_size));
    %     data_conv_temp = zeros(n,M,floor((data_simple_dim-patches_dim+1)/maxpool_size),floor((data_simple_dim-patches_dim+1)/maxpool_size));
    conv_temp = zeros(M,floor((data_simple_dim-patches_dim+1)/maxpool_size),floor((data_simple_dim-patches_dim+1)/maxpool_size));
    data_conv_temp{n}=[];
    for i=1:n
        for j=1:M
            %�������
            for k=1:data_simple_dim-patches_dim+1
                for l=1:data_simple_dim-patches_dim+1
                    temp_conv(k,l)=sum(sum(data_temp(i,k:k+patches_dim-1,l:l+patches_dim-1).*W_temp(j,:,:)));
                end
            end
            %maxpooling
            for k=1:floor((data_simple_dim-patches_dim+1)/maxpool_size)
                for l=1:floor((data_simple_dim-patches_dim+1)/maxpool_size)
                    temp = temp_conv((k-1)*maxpool_size+1:k*maxpool_size,(l-1)*maxpool_size+1:l*maxpool_size);
                    temp_pool(k,l) = max(temp(:));
                end
            end
            clear temp;
            conv_temp(j,:,:) = temp_pool;
        end
        data_conv_temp{i}=conv_temp;
    end
    data_result_conv{n}=[];
    for i=1:n
        %contrast normalization
        temp_contrast = contrast_normalization(data_conv_temp{i},contrast_dim);
        %ÿһ��cell����һ����������ά���� ��64*36*36
        data_result_conv{i} = temp_contrast;
    end
    %���ݷָ�
    %ûһ��data_result_conv cell ����һ���������ݵľ�����
    %ÿһ��cell����temp����һ���ָ��γ������ݼ� ��64*36*36�ķָ�(36-5+1)^2�� 64*5*5��С��ά���塣
    t=floor((floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1)*(floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1)/9);
    data_result{n,t}=[];
    %     cell_temp{1,(floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1)*(floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1)}=[];
    for i=1:n
        num_flag =1;
        for k=1:3:floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1
            for l=1:3:floor((data_simple_dim-patches_dim+1)/maxpool_size)-patches_dim+1
                data_result{i,num_flag}=data_result_conv{i}(:,k:k+patches_dim-1,l:l+patches_dim-1);
                num_flag = num_flag+1;
            end
        end
    end
    % i���������ݵ�������j����һ���˲�����������k*l�Ǿ�������γɵ�patch��С���ɷָ���������� �����λ�� �ָ����������ά��
end
if layer_i==2
    simple_num=size(data,2);
    [K,M,N]=size(data{1});%KΪ�˲���������M*Nÿ���˲�������õ���һ��Ĵ�С �� 64*36*36
    conv_num = size(W_old,2);
    temp_conv = zeros(M-patches_dim+1)*(N-patches_dim+1);
    temp_pool = zeros(floor((M-patches_dim+1)/maxpool_size),floor((N-patches_dim+1)/maxpool_size));
    %      data_conv_temp = zeros(simple_num,conv_num,floor((M-patches_dim+1)/maxpool_size),floor((N-patches_dim+1)/maxpool_size));
    conv_temp = zeros(conv_num,floor((M-patches_dim+1)/maxpool_size),floor((N-patches_dim+1)/maxpool_size));
    data_conv_temp{simple_num}=[];
    for i=1:simple_num
        i
        for j=1:conv_num
            %�������
            for k=1:M-patches_dim+1
                for l=1:N-patches_dim+1
                    temp =data{i}(:,k:k+patches_dim-1,l:l+patches_dim-1).*W_old{j};
                    temp_conv(k,l) = sum(temp(:));
                end
            end
            clear temp;
            %maxpooling
            for k=1:floor((M-patches_dim+1)/maxpool_size)
                for l=1:floor((N-patches_dim+1)/maxpool_size)
                    temp = temp_conv((k-1)*maxpool_size+1:k*maxpool_size,(l-1)*maxpool_size+1:l*maxpool_size);
                    temp_pool(k,l) = max(temp(:));
                end
            end
            clear temp;
            conv_temp(j,:,:) = temp_pool;
        end
        data_conv_temp{i} = conv_temp;
    end
    data_result_conv{simple_num}=[];
    for i=1:simple_num
        %contrast normalization
        temp_contrast = contrast_normalization(data_conv_temp{i},contrast_dim);
        %ÿһ��cell����һ����������ά���� ��64*36*36
        data_result_conv{i} = temp_contrast;
    end
    %���ݷָ�
    %ûһ��data_result_conv cell ����һ���������ݵľ�����
    %ÿһ��cell����temp����һ���ָ��γ������ݼ� ��64*36*36�ķָ�(36-5+1)^2�� 64*5*5��С��ά���塣
    data_result{simple_num,floor((floor((M-patches_dim+1)/maxpool_size)-patches_dim+1)*(floor((N-patches_dim+1)/maxpool_size)-patches_dim+1)/9)}=[];
    %     cell_temp{1,(floor((M-patches_dim+1)/maxpool_size)-patches_dim+1)*(floor((N-patches_dim+1)/maxpool_size)-patches_dim+1)/9}=[];
    for i=1:simple_num
        num_flag =1;
        for k=1:3:floor((M-patches_dim+1)/maxpool_size)-patches_dim+1
            for l=1:3:floor((N-patches_dim+1)/maxpool_size)-patches_dim+1
                data_result{i,num_flag}=data_result_conv{i}(:,k:k+patches_dim-1,l:l+patches_dim-1);
                num_flag=num_flag+1;
            end
        end
    end
    % i���������ݵ�������j����һ���˲�����������k*l�Ǿ�������γɵ�patch��С���ɷָ���������� �����λ�� �ָ����������ά��
end


if layer_i==3
    %ѵ�����ݵ�����ά��Ԥ����Ϊ5*5
    patches_dim = 10;
    %maxpooling�����С Ԥ����2*2
    maxpool_size =1;
    %contrast normalization dimension Ԥ����Ϊ5*5
    contrast_dim = 10;
    simple_num=size(data,2);
    [K,M,N]=size(data{1});%KΪ�˲���������M*Nÿ���˲�������õ���һ��Ĵ�С �� 64*36*36
    conv_num = size(W_old,2);
    temp_conv = zeros(M-patches_dim+1)*(N-patches_dim+1);
    temp_pool = zeros(floor((M-patches_dim+1)/maxpool_size),floor((N-patches_dim+1)/maxpool_size));
    %      data_conv_temp = zeros(simple_num,conv_num,floor((M-patches_dim+1)/maxpool_size),floor((N-patches_dim+1)/maxpool_size));
    conv_temp = zeros(conv_num,floor((M-patches_dim+1)/maxpool_size),floor((N-patches_dim+1)/maxpool_size));
    data_conv_temp{simple_num}=[];
    for i=1:simple_num
        for j=1:conv_num
            %�������
            for k=1:M-patches_dim+1
                for l=1:N-patches_dim+1
                    temp =data{i}(:,k:k+patches_dim-1,l:l+patches_dim-1).*W_old{j};
                    temp_conv(k,l) = sum(temp(:));
                end
            end
            clear temp;
            %maxpooling
            for k=1:floor((M-patches_dim+1)/maxpool_size)
                for l=1:floor((N-patches_dim+1)/maxpool_size)
                    temp = temp_conv((k-1)*maxpool_size+1:k*maxpool_size,(l-1)*maxpool_size+1:l*maxpool_size);
                    temp_pool(k,l) = max(temp(:));
                end
            end
            clear temp;
            conv_temp(j,:,:) = temp_pool;
        end
        data_conv_temp{i} = conv_temp;
    end
    data_result_conv{simple_num}=[];
    for i=1:simple_num
        %contrast normalization
        temp_contrast = contrast_normalization(data_conv_temp{i},contrast_dim);
        %ÿһ��cell����һ����������ά���� ��64*36*36
        data_result_conv{i} = temp_contrast;
    end
    %���ݷָ�
    %ûһ��data_result_conv cell ����һ���������ݵľ�����
    %ÿһ��cell����temp����һ���ָ��γ������ݼ� ��64*36*36�ķָ�(36-5+1)^2�� 64*5*5��С��ά���塣
    data_result{simple_num,floor((floor((M-patches_dim+1)/maxpool_size)-patches_dim+1)*(floor((N-patches_dim+1)/maxpool_size)-patches_dim+1)/9)}=[];
    %     cell_temp{1,(floor((M-patches_dim+1)/maxpool_size)-patches_dim+1)*(floor((N-patches_dim+1)/maxpool_size)-patches_dim+1)/9}=[];
    for i=1:simple_num
        num_flag =1;
        for k=1:3:floor((M-patches_dim+1)/maxpool_size)-patches_dim+1
            for l=1:3:floor((N-patches_dim+1)/maxpool_size)-patches_dim+1
                data_result{i,num_flag}=data_result_conv{i}(:,k:k+patches_dim-1,l:l+patches_dim-1);
                num_flag=num_flag+1;
            end
        end
    end
    % i���������ݵ�������j����һ���˲�����������k*l�Ǿ�������γɵ�patch��С���ɷָ���������� �����λ�� �ָ����������ά��
end

clearvars -except data_result data_result_conv ;



end