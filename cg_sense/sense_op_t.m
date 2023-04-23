function [res] = sense_op_t(coils_ind,x,loc_mask,m,n,no_c)

%% put the appr. locations of the conc. kspace with the incoming data
kk1 = zeros(m,n,no_c,'single');
kk1(loc_mask) = x;

%% go back to image domain

kspace_to_im = @(x) ifft2c(x);%* sqrt(size(x,1) * size(x,2));
% kk2 = kspace_to_im(kk1);

kk2 = kspace_to_im(kk1);

%% conjugate sens multiplication
ev1 = conj(coils_ind).*kk2;
res= sum(ev1,3);
res = res(:);    

end




