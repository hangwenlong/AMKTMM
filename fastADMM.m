function [w_k, b, s_hatk,lambda_hatk, rk, stop_iter] = fastADMM (X, y, p, q, C, tau, beta,alpha,b, wk, max_iter, eps, rho, eta)

if nargin < 14, eta = 0.999;       end
if nargin < 13, rho = 1;          end
if nargin < 12,  eps = 1e-8;        end
if nargin < 11,  max_iter = 500;    end
if nargin < 10,  wk = zeros(size(X, 1),1);       end  


n = size(X, 1);
d = size(X, 2);

s_km1 = zeros(d, 1);
s_hatk = s_km1;
lambda_km1 = ones(d, 1);
lambda_hatk = lambda_km1;
t_k = 1;
c_km1 = 0;

recent_number = 50;
recent_idx = 0;
obj_recent = zeros(recent_number, 1);

for k=1: max_iter
    W_S=[];
    for j_K=1:length(wk)
        W_S=[W_S wk{:,j_K}];
    end  
    w_k = (W_S * beta + lambda_hatk + rho * s_hatk + X'*(alpha)) / (rho + 1);

    W_k = reshape(w_k, p, q);  
    Lambda_k = reshape(lambda_hatk, p, q); 
    S = shrinkage(rho*W_k - Lambda_k, tau) / rho;
    s_k = reshape(S, d, 1); 
    
    lambda_k = lambda_hatk - rho * (w_k - s_k);   
    c_k = (lambda_k - lambda_hatk)' * (lambda_k - lambda_hatk) / rho + rho * (s_k - s_hatk)' * (s_k - s_hatk);
    
    if (c_k < eta * c_km1)
        t_kp1 = 0.5 * (1 + sqrt(1 + 4*t_k*t_k));
        s_hatkp1 = s_k + (t_k-1) / t_kp1 * (s_k - s_km1);
        lambda_hatkp1 = lambda_k + (t_k-1) / t_kp1 * (lambda_k - lambda_km1);
        restart = false;
    else
        t_kp1 = 1;
        s_hatkp1 = s_km1;
        lambda_hatkp1 = lambda_km1;
        c_k = c_km1 / eta;
        restart = true;
    end
    
    s_hatk = s_hatkp1;
    lambda_hatk = lambda_hatkp1;
    c_km1 = c_k;
    s_km1 = s_k;
    lambda_km1 = lambda_k;
    t_k = t_kp1;
    
    obj_k = objective_value(w_k, p, q, b, X, y, C, tau, wk, beta);
    recent_idx = recent_idx + 1;
    obj_recent(recent_idx) = obj_k;
    if (recent_idx == recent_number)
        recent_idx = 0;
    end
    if mod(k, 1000) == 0
        rk = sum(svd(reshape(w_k, p, q))>1e-6);
        fprintf('k=%d, obj=%f, restart=%d, rank=%d\n', k, obj_k, restart, rk);
    end
    
    if (abs(obj_k - mean(obj_recent)) / abs(mean(obj_recent)) < eps && k > recent_number)
        break;
    end
end

stop_iter = k;
rk = sum(svd(reshape(w_k, p, q))>1e-6);
    function obj = objective_value(w, p, q, b, X, y, C, tau, wk,beta)
        W_SK=[];
        for j_KS=1:length(wk)
            W_SK=[W_SK wk{:,j_KS}];
        end  
        ww= w - W_SK * beta;
        obj = 0.5 * (ww') * ww + 0.5 *C * sum( (y - X * w - b).^2) + tau * norm_nuc(reshape(w,p,q),p,q);
    end
end