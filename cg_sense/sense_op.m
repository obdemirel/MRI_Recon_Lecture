function [res] = sense_op(coils_ind,mb_kspace,m,n,coil_n,acq_p)
kk1 = reshape(mb_kspace,m,n);


%% sens multiplication
ev1 = coils_ind.*repmat(kk1,[1 1 coil_n]);

%% go back to k-space
im_to_kspace = @(x) fft2c(x);% / sqrt(size(x,1) * size(x,2));

ev2= im_to_kspace(ev1);

%% taking the acq points from k-space
res = ev2(acq_p);


end

