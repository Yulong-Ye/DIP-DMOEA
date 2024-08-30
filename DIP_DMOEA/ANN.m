function init_Pop = ANN(his_POS,curr_POS)

%%his_NDS: nxd curr_NDS: nxd


h=size(his_POS,2);  %input nodes 
i=5;                %Hidden nodes   
j=size(curr_POS,2); %Output nodes 
Alpha=0.1;          %The learning rate
Beta=0.1;           %The learning rate
Gamma=0.8;          %The constant determines effect of past weight changes
maxIteration=10;    %The max number of Iteration
trainNum=size(his_POS,1);


V=2*(rand(h,i)-0.5);    %The weights between input and hidden layers——[-1, +1]
W=2*(rand(i,j)-0.5);    %The weights between hidden and output layers——[-1, +1]
HNT=2*(rand(1,i)-0.5);  %The thresholds of hidden layer nodes
ONT=2*(rand(1,j)-0.5);  %The thresholds of output layer nodes
DeltaWOld(i,j)=0; %The amout of change for the weights  W
DeltaVOld(h,i)=0; %The amout of change for the weights  V
DeltaHNTOld(i)=0; %The amount of change for the thresholds HNT
DeltaONTOld(j)=0; %The amount of change for the thresholds ONT

Epoch=1;


% Normalize the data set

[inputn,inputs] = mapminmax(his_POS');
inputn = inputn';
[outputn,outputs] = mapminmax(curr_POS');
outputn = outputn';

while Epoch<maxIteration
    for k=1:trainNum
        
        a=inputn(k,:);
        
        ck=outputn(k,:);
        % Calcluate the value of activity of hidden layer FB
        for ki=1:i
            b(ki)=logsig(a*V(:,ki)+HNT(ki));
        end;
        %  Calcluate the value of activity of hidden layer FC
        for kj=1:j
            c(kj)=logsig(b*W(:,kj)+ONT(kj));
        end;
        % Calculate the errorRate of FC
        d=c.*(1-c).*(ck-c);
        % Calculate the errorRate of FB
        e=b.*(1-b).*(d*W');
        %Update the weights between FC and FB——Wij 
        for ki=1:i
            for kj=1:j
                DeltaW(ki,kj)=Alpha*b(ki)*d(kj)+Gamma*DeltaWOld(ki,kj);
            end
        end;
        W=W+DeltaW;
        DeltaWOld=DeltaW;
        %Update the weights between FA and FB——Vhj
        for kh=1:h
            for ki=1:i
                DeltaV(kh,ki)=Beta*a(kh)*e(ki);                               
            end
        end;
        V=V+DeltaV;                                                    
        DeltaVold=DeltaV;                                              
        % Update HNT and ONT
        DeltaHNT=Beta*e+Gamma*DeltaHNTOld;
        HNT=HNT+DeltaHNT;
        DeltaHNTOld=DeltaHNT;
        DeltaONT=Alpha*d+Gamma*DeltaONTOld;
        ONT=ONT+DeltaONT;
        DeltaTauold=DeltaONT;
    end 
    Epoch = Epoch +1; % update the iterate number
end



inputn = outputn;

for k=1:size(inputn,1)
    a=inputn(k,:); %get testSet
    
    %Calculate the value of activity of hidden layer FB
    for ki=1:i
        b(ki)=logsig(a*V(:,ki)+HNT(ki));
    end;
    %Calculate the result
    for kj=1:j
        c(kj)=logsig(b*W(:,kj)+ONT(kj));
    end;
    
    init_Pop(k,:)=c;

end
init_Pop = mapminmax('reverse',init_Pop',outputs);

end
