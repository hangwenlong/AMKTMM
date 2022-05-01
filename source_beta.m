function [beta,alpha,b] = source_beta (X, y, W, s, lambda, P, max_iter, rho)

if nargin < 8,  rho = 1;          end
if nargin < 7,  max_iter = 1000;    end

beta = zeros(1,numel(W));

index_p = find(y==1);
l_p = numel(index_p);
index_m = find(y==-1);
l_m = numel(index_m);
zita = zeros(1,numel(y));
zita(index_p) = numel(y)/(2*l_p);
zita(index_m) = numel(y)/(2*l_m);

m = diag(P);
m = m(1:end-1);
Q_1 = [(1+rho).*y - X * (lambda + rho * s);0];
S1 = P*Q_1;
alpha_1 = S1(1:end-1);
for j = 1:numel(W)
    Q_j = [X * W{j}; 0];
    Q{j} = Q_j;
    alpha_var = P*Q_j;
    alpha_2{j} = alpha_var(1:end-1);
    term_prev2{j} = alpha_2{j}./(m*(1+rho));
end

term_prev1 = alpha_1./(m*(1+rho)); 
term_prev2_mat = zeros(numel(W),numel(y));
for idx_modello = 1:numel(W)
    term_prev2_mat(idx_modello,:) = term_prev2{idx_modello}';
end
for k = 1:max_iter   
    S = beta*term_prev2_mat; 
    yh = term_prev1 - S'; 
    part = zita'.*(y.*yh);
    deriv = - ((part'>0).*zita.*y')*term_prev2_mat';
    beta_one = beta-deriv/(sqrt(k)*numel(y));    
    beta_two = beta_one;
    index_NEG = find(beta_one<0);
    beta_two(index_NEG) = 0;   
    if(norm(beta_two)) > 1
        beta_two = beta_two/(norm(beta_two));
    end
    beta = beta_two;   
end

inter = zeros(numel(y)+1,1);
for k = 1:numel(W)  
    inter = inter + beta(k).* Q{k};  
end
M = Q_1 - inter; 
alpha_inter = P * M;
alpha = alpha_inter(1:end-1);
b = alpha_inter(end)/(1+rho);

