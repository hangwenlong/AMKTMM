function pre_label = AMKTMM(traindata, trainlabel, testdata, C, tau, w_pre1, max_iter, rho)
%traindata: n*p*q;
%w_pre1: sources models obtained by LSSMM
%pre_label: prediction

if nargin < 11, rho = 1;          end
if nargin < 10,  max_iter = 100;    end

[n,p,q] = size(traindata);
d = p*q;
X = reshape(traindata,size(traindata,1),d);
y = trainlabel;

s_km1 = zeros(d, 1);
s_hatk = s_km1;
lambda_km1 = ones(d, 1);
lambda_hatk = lambda_km1;

H = X*X'+ eye(n)*(1 + rho)/C;
I = ones(n,1);
K = [H I; I' 0];
P = pinv(K);

for k=1: max_iter
    [beta, alpha, b] = source_beta (X, y', w_pre1, s_hatk, lambda_hatk, P, max_iter, rho); 
    [w_k, b, s_hatk, lambda_hatk] = fastADMM (X, y', p, q, C, tau, beta', alpha, b, w_pre1);     
    tr_w = w_k;
    pre_label = sign(testdata*tr_w+b);   
end

end

