function newPopulation = addNoise(init_population, Nini, n, Problem)

VarMin=Problem.XLow;
VarMax=Problem.XUpp;
%nά��
%�Ե�ǰȺ��ĸ������Ӱ��������������˰������ĸ����ԭʼ������
    newPopNumber = ceil(Nini) - size(init_population,2);
    newPopulation = zeros(n,newPopNumber);
    for i = 1:newPopNumber
        
        index = randperm(size(init_population,2));
        index2 = randperm(n);
       
        for idx2 = 1:round(n*0.5)      %���ѡȡһ���λ�����ֲ��仯
            temp = init_population(index2(idx2),index(1));
            temp =cauchyrnd(temp,4);
            if temp < VarMin(index2(idx2))
                temp = VarMin(index2(idx2));
            end
            if temp > VarMax(index2(idx2))
                temp = VarMax(index2(idx2));
            end
            init_population(index2(idx2),index(1)) = temp;
        end
        
        newPopulation(:,i) = init_population(:,index(1));
    end

end