function res = mtimes(a,b)
if a.adjoint
    % generate multi-coil data in image domain 
    x_array=ifft2c_mri(b.*a.mask);

    % combining multicoil images
    res=sum(squeeze(x_array).*conj(a.b1),3)./sum(abs((a.b1)).^2,3);
else
    % multi-coil image from combined image
    for ch=1:size(a.b1,3),
        x_array(:,:,ch)=b.*a.b1(:,:,ch);
    end
    % multi-coil image to k-space domain
    res=fft2c_mri(x_array).*a.mask;
end