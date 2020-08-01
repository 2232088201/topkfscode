function [indSF,scores,A,V] = lpboxUDFS_ver2(xTr,lambdaA,lambdaI,P,opt,yTr)
% Structure Regularized Unsupervised Discriminant Feature Selection
% yTr and yTe are label vectors, each data point associate with an label
% entry. 

if exist('opt','var')
    opt = optSetforS3C(opt);
else
    opt = optSetforS3C();
end

nbcluster = opt.nbcluster;
if exist('yTr','var')
    nbcluster = numel(unique(yTr));  
end

%% parameter settings:
% StrSSC specific parameters
iter_max =opt.iter_max; % iter_max =10
gamma0 =opt.gamma0;     % this is lambdaZ for structure regularization 
nu =opt.nu;             % nu=1; this is not important which changes in the optimization process

% parameters used in SSC
affine = opt.affine;  % affine space, no = 0
outliers = opt.outliers; % no = 0
alpha = opt.lambda;        
% r: the dimension of the target space when applying PCA or random projection
rho =opt.SSCrho; %rho = 1 be default;
N = size(xTr,2);
% parameters used in ADMM
maxIter =opt.maxIter; % in ADMM
%             admmopt.tol =1e-5;
%             admmopt.rho=1.1;
%             admmopt.maxIter =150;
%             admmopt.mu_max = 1e8;
%             admmopt.epsilon =1e-3;    

%% Initialization
C =zeros(size(xTr,2));  %% N * N
Theta_old =ones(size(xTr, 2)); %% N * N  %Theta_old =zeros(size(D, 2));
grps =0; 
iter =0; 
while (iter < iter_max)
    iter = iter +1;
    gamma1 =gamma0;
    if (iter <= 1)
        %% This is the standard SSC when iter <=1
        nu =1;
    else
        %% This is for re-weighted SSC
        nu = nu * 1.2;%1.1, 1.2, 1.5,
    end
    
    %% run ADMM to solve the re-weighted SSC problem
    C = ADMM_S3R(xTr, outliers, affine, alpha, Theta_old, gamma1, nu,  maxIter, C);
    %% Initialize Z with the previous optimal solution
    CKSym = BuildAdjacency(thrC(C,rho));   W = (abs(C)+abs(C'))/2;
    grps = SpectralClustering3C(CKSym, nbcluster, grps);
    if exist('yTr','var')
        [Acc] =  ClusteringMeasure(yTr, grps);
        disp(['iter = ',num2str(iter),', Clustering Acc = ', num2str( Acc)]);
    end
    Theta1 = 1 - form_structure_matrix(grps);
     %---------------- Feature Selection-----------------------------
     if ~exist('alphaTheta','var'),  alphaTheta = 0.95;  end 
    [indSF,scores,A,V] = ldaL2boxFS(xTr,grps,P,lambdaA, W, lambdaI);
    Yw = A*xTr;
    Theta2 = L2_distance(Yw(:,1:N),Yw(:,1:N));  Theta2 = (Theta2.^2)/2;
    Theta = alphaTheta * Theta1 + (1-alphaTheta)*Theta2; 
    %% Checking stop criterion
    tmp =Theta - Theta_old;
    maxabs = max(abs(tmp(:)));
    if (maxabs < 10^(-5))
        break; % if Theta didn't change, stop the iterations.
    end
    Theta_old =Theta;
end
%[indSF,scores, A1, A2, V,obj] = regressAndSelfExpress(xTr,yR,lambdaAFS,alphaFS,P);  %%% This place should be lpboxFS or lpboxFSHighDimTarget? Depends on further tests
end

function M = form_structure_matrix(idx,n)
if nargin<2
    n =size(idx,2);
end
M =zeros(n);
id =unique(idx);
for i =1:length(id)
    idx_i =find(idx == id(i));
    M(idx_i,idx_i)=ones(size(idx_i,2));
end
end