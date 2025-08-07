import os

import nibabel as nib
import numpy as np

from scipy.ndimage import binary_erosion, binary_fill_holes, binary_closing
from skimage.morphology import ball, cube
from skimage.filters import threshold_otsu
from skimage.measure import label


def _get_largest_connected_commponent(
        mask,
    ):
    
    labels = label(mask)
    assert(labels.max() != 0) # assume at least 1 CC
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    
    return largest_cc


def binary_segmentation(
        input_img_path,
        output_mask_path, 
    ):
    
    # Load nifti
    img_nii = nib.load(input_img_path)
    img = img_nii.get_fdata()
    
    # Extract header and affine
    header = img_nii.header
    affine = img_nii.affine
    
    # Otsu thresholding
    thresh = threshold_otsu(img)
    mask = img > thresh
    
    # Select largest connected set
    mask = _get_largest_connected_commponent(mask)
   
    # Fill holes
    mask = binary_fill_holes(mask)
    
    # Apply morphological closing
    closing_radius = 3
    mask = np.pad(mask, closing_radius)
    mask = binary_closing(mask, structure=ball(closing_radius))
    mask = mask[closing_radius:-closing_radius,
                closing_radius:-closing_radius,
                closing_radius:-closing_radius]
    
    # Initialize output directory
    output_dir = os.path.dirname(output_mask_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mask
    mask_nii = nib.Nifti1Image(mask.astype(np.float32), affine, header)
    nib.save(mask_nii, output_mask_path)


def erode_mask(
        input_mask_path,
        output_mask_path,
        eros_rad=0.0,
        eros_strc="ball",
        **kwargs,
    ):    
    
    # Load mask
    mask_nii = nib.load(input_mask_path)
    mask = mask_nii.get_fdata()
    
    # Resolution
    vox = mask_nii.header["pixdim"][1:4]
    
    if eros_rad > 0:
        
        # Define structure element
        if eros_strc == "ball":
            erosion_radius = np.max(np.ceil(eros_rad / vox)) # from mm to voxel
            structure = ball(erosion_radius)
        elif eros_strc == "cube":
            erosion_radius = np.max(np.ceil(eros_rad / vox)) # from mm to voxel
            structure = cube(erosion_radius)
        elif eros_strc == "ellipsoid":
            # Init ellipsoidal structural element
            center = np.ceil(eros_rad / vox)
            sz = np.array(2*center + 1, dtype=int)
            distance = np.zeros(sz)
            for i in range(sz[0]):
                for j in range(sz[1]):
                    for k in range(sz[2]):
                        distance[i,j,k] = np.linalg.norm(([i, j, k] - center) * vox)
            structure = np.ones(sz) * (distance <= max(center*vox))
            
        # 4D case
        if np.ndim(mask) == 4:
            structure = structure[..., None]
               
        # Perform binary erosion
        mask = binary_erosion(mask, structure=structure)
               
    # Initialize output directory
    output_dir = os.path.dirname(output_mask_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Save mask
    output_mask_nii = nib.Nifti1Image(mask.astype(np.float32), mask_nii.affine, mask_nii.header)
    nib.save(output_mask_nii, output_mask_path)