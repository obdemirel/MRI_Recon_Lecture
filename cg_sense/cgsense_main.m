function [res,res_kspace] = cgsense_main(kspace_in,sense_maps_in,iter,gui_on)
tic
kspace = kspace_in;
sense_maps = sense_maps_in;



[m,n,no_c] = size(kspace);

non_acq_p = kspace==0;
acq_p = ones(m,n,no_c,'single')-non_acq_p;

non_acq_p = logical(non_acq_p);
loc_mask = logical(acq_p);


y = kspace(loc_mask);
cc =sense_maps(:,:,:,:); %% loading sens maps


E = @(x) sense_op(cc,x,m,n,no_c,loc_mask);
ET = @(x) sense_op_t(cc,x,loc_mask,m,n,no_c);

ATA = @(x) ET(E(x));
ATb = ET(y);

inp = reshape(ATb,[m,n]);


[imm,error] = conjgrad(iter,ATA, ATb, inp(:),gui_on);
final_image = reshape(imm,[m n]);
res = final_image;

toc
end

