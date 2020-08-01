% Varying number of Features;
%%%%%%%-------------数据读入---- gX 的每一列表示一个样本----------
clear; 
load(['./MSRA25_uni.mat']);        
gY=Y(:);
gX = NormalizeFea(full(X),0);



[Dim,N] = size(gX);
labelSet = unique(gY);    numClass = numel(labelSet);
%%%%%%%%%%%--------------------Parameters Setting---------------%%%%%%%%
Nround = 10; lambdaA = 0.01; lambdaI = 3;  n_size = 6;  ldc = 5;
feaRange = [5:5:50];


for iter1 = 1:length(feaRange)
    numFeat = feaRange(iter1);

    opt = optSetforS3C();
    [ind_lpboxUDFS2,scores,A,V] = lpboxUDFS_ver2(gX,lambdaA,lambdaI,numFeat,opt,gY);
    %% -----evaluation process  12 methods
    [nn_ac_lpboxUDFS2(iter1),nn_ac_std_lpboxUDFS2(iter1),clu_ac_lpboxUDFS2(iter1),clu_ac_std_lpboxUDFS2(iter1), clu_mhat_lpboxUDFS2(iter1),clu_mhat_std_lpboxUDFS2(iter1)] = evalute_Feature(gX',gY,numFeat,ind_lpboxUDFS2);
    
%   save(['./result_unsup_revise_tmp/' dataName '_' num2str(numFeat) '.mat'],'gX','gY','pc','ind_Lap','idx_SEFS21','ind_UDFS','ind_JELSR','ind_RUFS','ind_TRACK','ind_URAFS','ind_SCUFS','ind_MCFS_p','ind_SPFS','ind_lpboxUDFS2');
end



