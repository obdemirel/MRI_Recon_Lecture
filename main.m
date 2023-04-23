clear all
%close all
clc

%some usefull functions
rssq = @(x) squeeze(sum(abs(x).^2,3)).^(1/2); %root-sum-squares
addpath cg_sense

load brain_kspace.mat
load brain_maps.mat
kspace = kspace./max(abs(kspace(:)));
DATA = kspace; 
cmaps = maps;
cmaps=double(cmaps);
[nx,ny,nc]=size(DATA);


rssq_kspace = rssq(ifft2c(kspace));
figure, imshow(rssq_kspace,[]), title('Data in Image Domain') %data in image-domain

figure,
for ii = 1:size(kspace,3)
    subplot(4,5,ii),imshow(abs(maps(:,:,ii)),[]), title(strcat('Channel ',num2str(ii),' Maps')) %senstivity maps
end
figure,
for ii = 1:size(kspace,3)
    subplot(4,5,ii),imshow(log(kspace(:,:,ii)),[-10 0]),title(strcat('Channel ',num2str(ii),' K-space')) %data in k-space
end

%sampling the k-space
load mask_R3.mat
mask(:,128-12:128+12-1) = 1;
kspace_r3 = kspace.*repmat(mask,[1 1 size(kspace,3)]);
figure,
imshow(abs(reshape(rssq(ifft2c(kspace_r3)),[size(kspace,1) size(kspace,2)])),[0 max(max(rssq(ifft2c(kspace))))/1]), title('Undersampled Data in Image Domain') %data in image-domain
figure,
for ii = 1:size(kspace,3)
    subplot(4,5,ii),imshow(log(kspace_r3(:,:,ii)),[-10 0]),title(strcat('Channel ',num2str(ii),' K-space Undersampled')) %data in k-space
end

%%SENSE Reconstruction
[cgsense_res] = cgsense_main(kspace_r3,maps,15,0);
figure, imshow(abs(cgsense_res),[0 max(abs(cgsense_res(:)))])


%%% Sparse Reconstruction

% Multicoil combination using sum of square
Img=sum(abs(ifft2c_mri(DATA).^2),3).^(1/2);


load mask_R3.mat
mask(:,128-12:128+12-1) = 1;
kdata=double((DATA).*mask);

% Multicoil Encoding Operator
E=Emat(mask,cmaps);
Efull=Emat(ones(size(mask)),cmaps);
Img_full = sum(conj(cmaps).*ifft2c(DATA),3);

% Wavelet transform 
W=Wavelet('Daubechies',4,4);

% Regularization Parameter
lambda=0.004;
%lambda=0.004*3;

% Number of iterations
nite=25;

% Get undersampled images
Img_u=E'*kdata;

% From image domain to sparse domain 
Img_sparse=W*Img_u;

% Iterative reconstruction
figure, set(gcf, 'Position', get(0, 'Screensize'));
kin = fft2c(Img_u);
subplot(2,3,1), imshow(log(kin(9:256-8,9:256-8)),[]),title(strcat('K-space at Iteration: ',num2str(0))), drawnow
subplot(2,3,4), imshow(abs(Img_u),[0 max(max(abs(W'*Img_sparse)))]), title('Forward Encoding'),drawnow
loss = [];

for ite=1:nite
    
    pause(1)
    kin = fft2c(W'*Img_sparse);
    subplot(2,3,1), imshow(log(kin(9:256-8,9:256-8)),[-10 0]),title(strcat('K-space at Iteration: ',num2str(ite))), drawnow
    subplot(2,3,4), imshow(abs(W'*Img_sparse),[0 max(max(abs(W'*Img_sparse)))]), title('Forward Encoding'),drawnow
    Img_tmp=Img_sparse;

    %Soft-thresholding, as shown in Equation 8.16
    Img_sparse=(abs(Img_sparse)-lambda).*Img_sparse./abs(Img_sparse).*(abs(Img_sparse)>lambda);
    Img_sparse(isnan(Img_sparse))=0;
    subplot(2,3,5), imshow(abs(Img_sparse),[0 max(abs(Img_sparse(:)))/20]), title('Sparse Domain Denoising') ,drawnow

	% Image update as shown in Equation 8.15 (data consistency)
	Img_sparse=Img_sparse-W*(E'*(E*(W'*Img_sparse)-kdata));
    subplot(2,3,6), imshow(abs(W'*Img_sparse),[0 max(max(abs(W'*Img_sparse)))]), title('Denoised Image'),drawnow
    psnr_cal = psnr(abs(W'*Img_sparse),abs(Img_full),max(abs(Img_full(:))));
    subplot(2,3,3), imshow(abs(abs(W'*Img_sparse)-abs(Img_full)),[0 max(abs(Img_full(:)))/5]), title(strcat('Difference to Reference and PSNR: ',num2str(psnr_cal))),drawnow
    
    %Recon information
    loss = [loss norm(Img_sparse(:)-Img_tmp(:))/norm(Img_tmp(:))]; 
    subplot(2,3,2), plot(loss), title('Relative Error'),drawnow
    
    fprintf(' ite: %d, update: %f3\n', ite,norm(Img_sparse(:)-Img_tmp(:))/norm(Img_tmp(:)));
end

%From sparse domain to image domain
Img_recon=W'*Img_sparse;

% Display images (left to right: Fullysampled Image,Undersampled Image,Reconstructed Image)
%figure,imshow(abs(cat(2,Img,Img_u,Img_recon)),[0 max(abs(Img(:)))/1]),title('Fullysampled Image,     Undersampled Image,     Reconstructed Image')

Img = abs(Img_full);
Img_u = abs(Img_u);
Img_recon = abs(Img_recon);
cgsense_res = abs(cgsense_res);
figure,imshow(abs(cat(2,Img,Img_u,Img_recon,cgsense_res)),[0 max(abs(Img(:)))/1]),title('Fullysampled Image,     Undersampled Image,     CS Reconstructed Image, CG-SENSE Reconstructed Image')
figure,imshow(abs(cat(2,abs(Img-Img),abs(Img-Img_u),abs(Img-Img_recon),abs(Img-cgsense_res))),[0 max(abs(Img(:)))/5]),title('Difference to Fullysampled Image,     Undersampled Image,     CS Reconstructed Image, CG-SENSE Reconstructed Image')
