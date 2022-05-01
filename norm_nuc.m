function [ val ] = norm_nuc( W,p,q )

if p~=size(W,1) 
    error('dimension does not mach!');
end
if q~=size(W,2) 
    error('dimension does not mach!');
end

[U, S, V] = svd(W);
 s = max(0, S);
 val = sum(diag(s));


end

