import os
import datetime
import tempfile
import shutil

import nibabel as nib
import numpy as np

from scipy.special import erfcinv
from scipy.ndimage import binary_dilation
from skimage.morphology import ball
from scipy.interpolate import RBFInterpolator

from utils.tools.mirtk import mirtk_average_images
from utils.tools.mrtrix import mrtrix_multiply


def _create_mask_artefact(
        pha,
        mask,
        input_mask_artefact=None,
        dil_size=1,
        **kwargs,
    ):
    
    # Estimate mask artefact using MAD
    # Compute Median Absolute Deviation
    c = -1/(np.sqrt(2)*erfcinv(3/2))
    MED = np.median(pha[mask])
    MAD = c*np.median(np.abs(pha[mask] - MED))
    
    # Create artefact mask
    mask_artefact = np.zeros(np.shape(pha), dtype=bool)
    mask_artefact[np.abs(pha-MED) > 3*MAD] = True
    mask_artefact *= mask
    
    if input_mask_artefact is not None:
        mask_artefact = np.logical_or(mask_artefact, input_mask_artefact)
        
    # Apply dilation
    if dil_size > 0:
        mask_artefact = binary_dilation(mask_artefact, structure=ball(dil_size))
    mask_artefact *= mask
    
    return mask_artefact


def _b_spline_phase_artefact_correction(
        pha,
        mask,
        mask_artefact,
        vox=[1.0,1.0,1.0],
        bsp_n=100,
        bsp_axes=[0,1,2],
        **kwargs,
    ):
    
    # Image dimension and resolution
    nx, ny, nz = np.shape(pha)
    dx, dy, dz = vox
    
    # 3D case
    if bsp_axes == [0,1,2]:
        
        # Sample mask and phase
        mask_f = mask.flatten()
        mask_artefact_f = mask_artefact.flatten()
        mask_f = np.logical_xor(mask_f, mask_artefact_f)
        pha_f = pha.flatten()
        
        # Create grid
        grid = np.mgrid[0:nx, 0:ny, 0:nz]
        grid = grid.reshape(3, -1).T.astype(np.float64)
        grid[:,0] *= dx
        grid[:,1] *= dy
        grid[:,2] *= dz    
        
        # Apply mask on sampled grid and phase
        grid_artefact = grid[mask_artefact_f, :]
        grid = grid[mask_f, :]
        pha_f = pha_f[mask_f]
        
        # B-spline correction
        try:
            interp = RBFInterpolator(
                grid,
                pha_f, 
                neighbors=bsp_n, 
                kernel="thin_plate_spline",
            )
            pha[mask_artefact] = interp(grid_artefact)
            
        except:
            print("RBF interpolation error!")
    
    # 2D case
    elif bsp_axes == [0,1]:
        for z in range(nz):
            
            # Extract slice
            pha_z = pha[..., z]
            mask_z = mask[..., z]
            mask_artefact_z = mask_artefact[..., z]
            
            # No artefact in slice
            if not np.sum(mask_artefact_z): 
                continue
                
            pha_z_f = pha_z.flatten()
            mask_z_f = mask_z.flatten()
            mask_artefact_z_f = mask_artefact_z.flatten()
            mask_z_f = np.logical_xor(mask_z_f, mask_artefact_z_f)
            
            # Create grid
            grid_z = np.mgrid[0:nx, 0:ny]
            grid_z = grid_z.reshape(2, -1).T.astype(np.float64)
            grid_z[:,0] *= dx
            grid_z[:,1] *= dy
            
            # Apply mask on sampled grid and phase
            grid_artefact_z = grid_z[mask_artefact_z_f, :]
            grid_z = grid_z[mask_z_f, :]
            pha_z_f = pha_z_f[mask_z_f]
            
            # B spline correction
            try:
                interp = RBFInterpolator(
                    grid_z,
                    pha_z_f, 
                    neighbors=bsp_n, 
                    kernel="thin_plate_spline",
                )
                pha_z[mask_artefact_z] = interp(grid_artefact_z)

                # Update slice
                pha[..., z] = pha_z
                
            except:
                print("RBF interpolation error!")
    
    # Apply mask
    pha *= mask
    
    return pha


def average_phase_artefact_mask(
        input_mask_artefact_paths,
        input_ref_mag_path,
        input_ref_mask_path,
        input_dof_paths,
        output_mask_artefact_path,
    ):
    
    mirtk_average_images(
        input_paths=input_mask_artefact_paths,
        input_dof_paths=input_dof_paths,
        input_ref_path=input_ref_mag_path,
        output_path=output_mask_artefact_path,
        label=True,
    )
    
    mrtrix_multiply(
        operand1=output_mask_artefact_path,
        operand2=input_ref_mask_path,
        output_path=output_mask_artefact_path,
    )
    

def correct_phase_artefact(
        input_pha_path, 
        input_mask_path,
        output_pha_path,
        input_mask_artefact_path=None,
        output_mask_artefact_path=None,
        verbose=False,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
        
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()
    if output_mask_artefact_path is None:
        output_mask_artefact_path = os.path.join(temp_dir.name, "artefact.nii.gz")
        
    # Load niftis
    pha_nii = nib.load(input_pha_path)
    pha = pha_nii.get_fdata()
    
    mask_nii = nib.load(input_mask_path)
    mask = mask_nii.get_fdata().astype(dtype=bool)
    
    if input_mask_artefact_path is not None:
        mask_artefact_nii = nib.load(input_mask_artefact_path)
        mask_artefact = mask_artefact_nii.get_fdata().astype(dtype=bool)
    else:
        mask_artefact = None
    
    # Get header and affine
    header = pha_nii.header
    affine = pha_nii.affine
    
    # Get image resolution
    vox = header["pixdim"][1:4]
    
    # Create artefact mask
    mask_artefact = _create_mask_artefact(
        pha=pha,
        mask=mask,
        input_mask_artefact=mask_artefact,
        **kwargs,
    )
    
    # B-spline phase artefact correction
    pha = _b_spline_phase_artefact_correction(
        pha=pha,
        mask=mask,
        mask_artefact=mask_artefact,
        vox=vox,
        **kwargs,
    )
    
    # Initialize output directory
    output_dir = os.path.dirname(output_pha_path)
    os.makedirs(output_dir, exist_ok=True)
       
    # Save
    pha_nii = nib.Nifti1Image(pha, affine, header)
    nib.save(pha_nii, output_pha_path)
    
    # Save
    mask_artefact_nii = nib.Nifti1Image(mask_artefact, affine, header)
    nib.save(mask_artefact_nii, output_mask_artefact_path)
    
    # Delete temp folder    
    shutil.rmtree(temp_dir.name) 
    
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Phase correction run time: {elapsed_time}")
