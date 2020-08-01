
clear; 
load(['./MSRA25_uni']);    
ldc = 1; 
gY=Y(:);
gX = NormalizeFea(full(X),0);



%%%%%%%%%%%--------------------Parameters Setting---------------%%%%%%%%
[Dim,N] = size(gX);    labelSet = unique(gY);    numClass = numel(labelSet);
%%%%%%%%%%%-----------------------------------------------------%%%%%%%%
Nround =10; lambdaA = 0.01; lambdaI = 3;  n_size = 6;  ldc = 3;


feaRange=[5:5:50];

for T = 1:Nround
    %% Random permutation of the data points
    rnperm = randperm(N);
    dataX = gX(:,rnperm);  labelY = gY(rnperm);
    [Dim,N] = size(dataX);
    %% index of each class
    Dind=cell(numClass,1);
    for iterC=1:numClass,
        Dind{iterC}=find(labelY==labelSet(iterC));
    end
    ind1 =[];  ind2=[];
    for c=1:numClass
        ind1 = [ind1; Dind{c}(1:ldc)];                              %%%  labeled index
        ind2 = [ind2; Dind{c}((1+ldc):end)];                        %%%  unlabeled index
    end
    xTr = dataX(:,ind1);    yTr = labelY(ind1);     nTr = numel(yTr);    %%%  labeled data
    xTe = dataX(:,ind2);    yTe = labelY(ind2);     nTe = numel(yTe);    %%%  unlabeled data
    vYSSL = [full(sparse(yTr,1:nTr,1)), zeros(numClass, nTe)];            %%%  vYSSL = [YL, YU];
    vYr = full(sparse(yTr,1:numel(yTr),1));
    
    [Wlap, Wlle] = DoubleWforMR([xTr, xTe], 'k', n_size);   %% Weight matrix
    L = diag(sum(Wlap)) - Wlap;
    
    for iter1 = 1:length(feaRange)
        numFeat = feaRange(iter1);
        %% -Structure regularized Discriminant Feature Selection----
        opt.nbcluster = numClass;  opt.gamma0 = 0.05;
       
        %%---------------Our Method--------------------
        [ind_SrSemiDFS] = lpboxSSFS(xTr,yTr,xTe,lambdaA,lambdaI,numFeat,opt,yTe);

        X_test_class = knnclassify(xTe(ind_SrSemiDFS(1:numFeat),:)',xTr(ind_SrSemiDFS(1:numFeat),:)', yTr);
        SrSemiDFSAcc1NN(T,iter1) = sum((X_test_class==yTe))*100/length(X_test_class);
    end

end



