function [indSF,scores,A,V,obj] = ldaL2boxFS(xTr,yTr,P, lambdaA, W, lambdaI)
%% 21-norm loss with 21-norm regularization
% xTr is a Dim * N data matrix, each column is a data vector
% yTr is a numClass * N label matrix, each column is a label vector
% gnd is a label vector, each entry indicates the class of a data point
%% Problem
% 但是问题在于在有监督的时候还要用到无监督信息，要么考虑dark nowledge,即类和类，同类与异类样本之间的间隙信息。
% 否则感觉没有什么意义。
%  min_A  || XdVA - Y||_F^2 + lambdaA * ||A||_F^2 + lambdaI Tr( AdVXLXTdVAT)

rho = 1; 
mu = 1.1; 
MaxIter = 150; 

if ~exist('lambdaA','var'),  lambdaA = 0.1;   end

if exist('W','var')
    W = (abs(W) + abs(W)')/2;
    L = diag(sum(W)) - W; 
else
    L = 0;  lambdaI = 0; 
end

[Dim,N] = size(xTr); 
labelSet = unique(yTr); 
classNum = numel(labelSet); 

y = zeros(N,1);   %% Preprocess of the labels
for iter1 = 1:classNum
   ind = yTr == labelSet(iter1);
   y(ind) = iter1; 
end
yTr = full(sparse(y,1:N,1)); 

V = ones(Dim,1); V1 = zeros(size(V)); V2 = zeros(size(V));  
A = rand(classNum,Dim);   ones1 = ones(size(V)); 
y1 = 0;     y2 = zeros(size(V));  y3 = zeros(size(V)); 

XXT = xTr*xTr';   XLXT = xTr*L*xTr';  Id = eye(Dim);  obj =0; 
for iter1 = 1:MaxIter
    dV = diag(V);
    % update projection mapping A
    A = (yTr*xTr'*dV)/((dV*XXT*dV) + lambdaA*Id + lambdaI*(dV*XLXT*dV)); 
    % compute V
    PSI1 = A'*A;  THETA = xTr*yTr'*A; 
    V =(2*(XXT.*PSI1')+2*lambdaI*(PSI1.*XLXT) + rho*(ones1*ones1')+2*rho*Id)\...
        (2*diag(THETA) + rho*((P-(y1/rho))*ones1+ V1 - (y2/rho) + V2 - (y3/rho)));  
    % Projection on Sb and Sp
    V1 = PSb(V+(y2/rho)); 
    V2 = PSp(V+(y3/rho)); 
    % update the parameters
    y1 = y1 + rho*(V'*ones1 - P); 
    y2 = y2 + rho*(V-V1); 
    y3 = y3 + rho*(V-V2);
    rho = rho*mu;
    % compute the objective function 
    obj(iter1) = norm(A*diag(V)*xTr - yTr,'fro')^2 + lambdaA*norm(A,'fro')^2 ...
        + lambdaI*trace(A*dV*XLXT*dV*A');
end
scores = sum(A.*A,1);
[~,indSF] = sort(scores,'descend');
end

function X = PSb(X,a,b)
if ~exist('a','var')
    a = 0; b =1;
end
A = X(:); 
ind1 = A<a; ind2 = A>b;
A(ind1)=a; A(ind2) = b; 
X = reshape(A, size(X));
end

function X = PSp(X)
Dim = length(X); 
ones1 = ones(Dim,1);
t0 = sqrt(Dim)/(2*norm(X-(ones1/2),2)); 
X1 = (ones1/2) + t0*(X-(ones1/2)); 
X2 = (ones1/2) - t0*(X-(ones1/2)); 
d1 = norm(X-X1,2); d2 = norm(X-X2,2);
if d1>d2 
    X = X2;
else
    X = X1;
end
end
