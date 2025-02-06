% Visualize Nifti file
V = niftiread('real_B_1BA054.nii.gz');  % path to nii.gz file
c = V(:,:,100);  % n = axial slice number 
imshow(c,[])
