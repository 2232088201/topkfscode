function [nn_acc_fs,nn_acc_std,clu_acc_mean,clu_acc_std, clu_mhat_mean,clu_mhat_std, rebunduncy] = evalute_Feature(X,L,feature_num,idx)
%% X: input  data matrix, each row is a sample, only denoted by selected features
%% Y: input label vector
%% rec_acc_fs: 1NN classification accuracy;  rec_acc_clu: kmeans clustering accuracy
%% rec_clu: MIhat values 
lab_val=unique(L);
mm=length(lab_val);
dat=[];
Lab=[];
for i=1:mm
      dat=[dat;X(L==lab_val(i),:)];
      Lab=[Lab;i*ones(sum(L==lab_val(i)),1)];
end
X=dat;
L=Lab;

indx=idx(1:feature_num);
newfea = NormalizeFea(X(:,indx));
data=[newfea,L];
% rec_acc_raw=crossvalidate([X,L],10,'KNN',1);
% disp(['raw recognition accuracy: ',num2str(rec_acc_raw)]);

[nn_acc_fs,nn_acc_std] = crossvalidate(data,10,'KNN',1);
% fprintf('num of feature: %5i, accuracy: %5.3f\n', feature_num, rec_acc_fs);

for i=1:20
    label = litekmeans(newfea,mm,'Replicates',10);
    [accu(i),  MIhat(i)] = xinBestMap(L,label);
end
clu_mhat_mean = mean(MIhat);
clu_mhat_std= std(MIhat);

clu_acc_mean = mean(accu);
clu_acc_std  = std(accu);

% fprintf('num of feature: %5i, NMI: %5.3f\n', feature_num, rec_clu);
% fprintf('num of feature: %5i, cluster: %5.3f\n', feature_num, rec_acc_clu);

rebunduncy = evalFSRedncy(newfea,indx,feature_num);
% fprintf('num of feature: %5i, redundancy: %5.3f\n', feature_num, rebunduncy);


fprintf('num of feature:%3i, accuracy: %5.3f, cluster acc: %5.3f,  NMI: %5.3f,  redundancy: %5.3f \n', feature_num, nn_acc_fs, clu_mhat_mean, clu_acc_mean,rebunduncy);