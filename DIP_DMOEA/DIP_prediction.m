function [init_solution,his_POS,curr_POS,pre_Pop] = DIP_prediction(curr_POS,curr_POF, his_POS,his_POF, Problem,popSize)

% Prediction via neural network in DIP-DMOEA for solving DMOP.
% curr_POS and his_POS denote N number of non-dominated solutions obtained in the current and previous time windows, respectively. 
% curr_POF and his_POF are the corresponding mapping from decision space to objective space
% Both in the form of N*d matrix. 
% N is the number of individual, 
% d is the variable dimension of the given DMOP
% output is the predicted initial solutions of the evolutionary search for new time window.



m = Problem.NObj;
N_curr = size(curr_POS,1);
N_his = size(his_POS,1);
d = size(curr_POS,2);
Sample_size = min(N_curr,N_his);


Fronts_his = ones(1,N_his);
Distances_his = CrowdDistances(his_POF, Fronts_his);
[~, index_his] = sort(Distances_his,'ascend'); 
his_POS = his_POS(index_his(1:Sample_size),:);

Fronts_curr = ones(1,N_curr);
Distances_curr = CrowdDistances(curr_POF, Fronts_curr);
[~, index_curr] = sort(Distances_curr,'ascend');
curr_POS = curr_POS(index_curr(1:Sample_size),:);

% while Sample_size < (popSize/2)
%     his_POS_noise = [addGaussianNoise_2(his_POS', size(his_POS,1), size(his_POS,2), Problem)]';
%     his_POS = [his_POS;his_POS_noise];
%     curr_POS_noise = [addGaussianNoise_2(curr_POS', size(curr_POS,1), size(curr_POS,2), Problem)]';
%     curr_POS = [curr_POS;curr_POS_noise];
%     Sample_size = size(curr_POS, 1);
% end

pre_Pop = ANN(his_POS,curr_POS);
init_Pop = (curr_POS')+rand*(pre_Pop-curr_POS')+rand*(curr_POS'-his_POS');

%init_Pop = getRandomPoints(Problem,size(his_POS,1));
%init_Pop = (curr_POS')+rand*(init_Pop-curr_POS')+rand*(curr_POS'-his_POS');
%pre_Pop = repair(pre_Pop,Problem);
init_Pop = repair(init_Pop,Problem);
init_solution = init_Pop;
end


function init_solution = repair(init_solution,Problem)
        VarMin=Problem.XLow;
        VarMax=Problem.XUpp;
        for j=1:size(init_solution,2)
            for i=1:size(init_solution,1)
                init_solution(i,j) = min(max(init_solution(i,j),VarMin(i)), VarMax(i));
            end
        end
end