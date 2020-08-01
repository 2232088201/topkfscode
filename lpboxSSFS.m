function [indSF,scores,A1, A2, V,obj] = lpboxSSFS(xTr,yTr,xTe,lambdaA,lambdaI,P,opt,yTe)

% yTr and yTe are label vectors, each data point associate with an label
% entry. 
dataX = [xTr, xTe];  
[~, N] = size(dataX); 
%% parameter settings:
% StrSSC specific parameters
if exist('opt','var')
    opt = optSetforS3C(opt);
else
    opt = optSetforS3C();
end
%% Initialization
C =zeros(size(dataX,2));               % C is initialized as a zero matrix
Theta_old =ones(size(dataX, 2));       %Theta_old =zeros(size(D, 2));
eval_iter =zeros(1, opt.iter_max);     % iter_max =10  
iter =0; 
while (iter < opt.iter_max)       
    iter = iter +1;    
    if (iter <= 1)
        %% This is the standard SSC when iter <=1
        nu =1; 
    else
        %% This is for re-weighted SSC
        nu = nu * 1.2;%1.1, 1.2, 1.5,      
    end
    
    %% run ADMM to solve the re-weighted SSC problem
    C = ADMM_S3R(dataX, opt.outliers, opt.affine, opt.lambda, Theta_old, opt.gamma0, nu,  opt.maxIter, C);

    %% Initialize Z with the previous optimal solution
    % CKSym = BuildAdjacency(thrC(C,rho));   W = (abs(CKSym) + abs(CKSym)')/2;
     W = (abs(C) + abs(C)')/2;
    % --------------Semi-Supervised and Unsupervised label prediction------
    [~, grpsU, Acc] = my_grf(yTr, W, yTe);
    grps = [yTr(:); grpsU(:)];
    Theta1 = 1 - form_structure_matrix(grps);
    disp(['iter = ',num2str(iter),', Semi-Supervised Acc = ', num2str(Acc)]);
    eval_iter(iter) = Acc;
    [indSF,scores,A,V] = ldaL2boxFS(dataX,grps,P,lambdaA, W, lambdaI);
    Yw = A*dataX;  Theta2 = L2_distance(Yw(:,1:N),Yw(:,1:N));  Theta2 = (Theta2.^2)/2;
    if ~exist('alphaTheta','var'),  alphaTheta = 0.95;  end 
    Theta = alphaTheta * Theta1 + (1-alphaTheta)*Theta2; 
    %% Checking stop criterion
    tmp =Theta - Theta_old;
    if (max(abs(tmp(:))) < 10^(-9))
        disp('max error lower than threshold and code break!')
        break; % if Theta didn't change, stop the iterations.
    end
    Theta_old =Theta;
end
% yR = full(sparse(grps,1:numel(grps),1));
% [indSF,scores, A1, A2, V,obj] = regressAndSelfExpress(xTr,yR,lambdaAFS,alphaFS,P); %%% This place should be lpboxFS or lpboxFSHighDimTarget? Depends on further tests
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


function [preYu,labelU, Acc] = my_grf(yTr, W, yTe)

l = numel(yTr);   
n = size(W, 1); 
% total number of points
YTr = full(sparse(yTr,1:l,1));

% the graph Laplacian L=D-W
L = diag(sum(W)) - W;

% the harmonic function.
preYu = - YTr * L( 1:l,l+1:n)/L(l+1:n, l+1:n); %%%Matrix is close to singular or badly scaled.

% compute the CMN solution
q = sum(YTr,2)+1; % the unnormalized class proportion estimate from labeled data, with Laplace smoothing
preYu = preYu .* repmat(q./sum(preYu,2), 1, n-l);

[~,labelU] = max(preYu,[],1);
if exist('yTe','var')
    Acc = 100*sum(labelU(:) == yTe(:))/numel(yTe);
end
end