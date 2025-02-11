% Visualize Nifti file
V = niftiread('nii.gz');  % path to nii.gz file
c = V(:,:,100);  % n = axial slice number 
imshow(c,[])
