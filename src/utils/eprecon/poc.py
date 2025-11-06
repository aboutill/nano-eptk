import os 
import datetime
import tempfile
import shutil

import nibabel as nib
import numpy as np


from scipy.ndimage import convolve, gaussian_filter
from utils.eprecon.fd_kernels import laplacian_kernel
from utils.tools.mirtk import mirtk_average_images
from utils.tools.mrtrix import mrtrix_multiply
from utils.mask import erode_mask
from utils.metrics import extract_ep_metrics


def _gaussian_filter(
        img,
        mask,
        gs_sigma=1.0,
        vox=[1.0,1.0,1.0],
        gs_axes=[0,1,2],
        **kwargs,
    ):
    
    # Image dim
    d = np.ndim(img)
    
    # Apply Gaussian smoothing
    if gs_sigma > 0.0:
        
        # Set sigma
        sigma = [0.0]*d
        for ax in gs_axes:
            sigma[ax] = gs_sigma / vox[ax] # in voxels
            
        img = gaussian_filter(img, sigma)

        # Filter correction at border
        mask_filtered = gaussian_filter(mask, sigma)
        mask_filtered[mask_filtered == 0] = 1 # Prevent NaN
    
        # Apply mask
        img /= mask_filtered
        img *= mask
        
    return img


def _poc_solver(
        pha,
        vox=[1.0,1.0,1.0],
        f0=128e6, # 3 Tesla
        pde_axes=[0,1,2],
        **kwargs
    ):
    
    # Physical constants
    w0 = 2*np.pi*f0
    mu0 = 4*np.pi*1e-7
    
    # Load Laplacian kernels
    k_del2_x, k_del2_y, k_del2_z = laplacian_kernel(vox=vox)    
    
    # Apply Central Finite Difference method to compute Laplacian
    # All axes combination are not implemented
    if pde_axes == [0,1,2]:
        del2 = convolve(pha, k_del2_x) + convolve(pha, k_del2_y) + convolve(pha, k_del2_z)
    elif pde_axes == [0,1]:
        del2 = 1.5 * (convolve(pha, k_del2_x) + convolve(pha, k_del2_y))
    
    # Compute conductivity
    sig = del2 / (w0*mu0)
        
    return sig


def _poc_reconstruction(
        input_pha_path,
        input_mask_path,
        output_sig_path,
        **kwargs,
    ):  
    
    # Load phase
    pha_nii = nib.load(input_pha_path)
    pha = pha_nii.get_fdata()

    # Load mask
    mask_nii = nib.load(input_mask_path)
    mask = mask_nii.get_fdata()
    
    # Get affine and header
    affine = pha_nii.affine
    header = pha_nii.header
    
    # Get resolution
    vox = pha_nii.header["pixdim"][1:4] # in mm

    # Apply mask on phase
    pha *= mask
    
    # Apply TPA
    pha /= 2
    
    # Apply Gaussian smoothing
    pha = _gaussian_filter(img=pha, mask=mask, vox=vox, **kwargs)
                
    # Solve POC
    vox = vox * 1e-3 # in m
    sig = _poc_solver(pha, vox=vox, **kwargs)
    
    # Initialize output directory
    output_dir = os.path.dirname(output_sig_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save sig
    sig_nii = nib.Nifti1Image(sig, affine, header)
    nib.save(sig_nii, output_sig_path)
    

def poc_pipeline(
        input_pha_path,
        input_mask_path,
        output_sig_path=None,
        output_ep_metrics_path=None,
        output_mask_eroded_path=None,
        output_sig_eroded_path=None,
        verbose=False,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
    
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()
    if output_sig_path is None:
        output_sig_path = os.path.join(temp_dir.name, "sig.nii.gz")
    if output_ep_metrics_path is None:
        output_ep_metrics_path = os.path.join(temp_dir.name, "ep_metrics.json")
    if output_mask_eroded_path is None:
        output_mask_eroded_path = os.path.join(temp_dir.name, "mask_eroded.nii.gz")
    if output_sig_eroded_path is None:
        output_sig_eroded_path = os.path.join(temp_dir.name, "sig_eroded.nii.gz")
    
    # POC reconstruction
    _poc_reconstruction(
        input_pha_path=input_pha_path,
        input_mask_path=input_mask_path,
        output_sig_path=output_sig_path,
        **kwargs,
    )
        
    # Erode mask
    erode_mask(
        input_mask_path=input_mask_path,
        output_mask_path=output_mask_eroded_path,
        **kwargs,
    )
    
    # Apply mask
    mrtrix_multiply(
        operand1=output_sig_path,
        operand2=output_mask_eroded_path,
        output_path=output_sig_eroded_path,
    )
        
    # Measure EP
    extract_ep_metrics(
        input_sig_path=output_sig_path,
        input_mask_eroded_path=output_mask_eroded_path,
        output_ep_metrics_path=output_ep_metrics_path,
        **kwargs,
    )
    
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"POC reconstruction pipeline run time: {elapsed_time}")
        
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)
    

def mspoc_pipeline(
        input_pha_paths,
        input_mask_paths,
        input_dof_paths,
        output_sig_path=None,
        input_ref_path=None,
        input_dhcp_labels9_paths=None,
        output_sig_paths=None,
        output_mask_eroded_paths=None,
        output_ep_metrics_paths=None,
        output_sig_eroded_paths=None,
        output_mask_path=None,
        output_mask_eroded_path=None,
        output_ep_metrics_path=None,
        output_sig_eroded_path=None,
        output_dhcp_labels9_path=None,
        verbose=False,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
    
    # Set length
    n = len(input_pha_paths)
    
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()
    if output_sig_path is None:
        output_sig_path = os.path.join(temp_dir.name, "sig.nii.gz")
    if output_sig_paths is None:
        output_sig_paths = [
            os.path.join(temp_dir.name, f"sig-{i}.nii.gz")
            for i in range(n)
        ]
    if output_mask_eroded_paths is None:
            output_mask_eroded_paths = [
                os.path.join(temp_dir.name, f"mask_eroded-{i}.nii.gz")
                for i in range(n)
            ]
    if output_ep_metrics_paths is None:
        output_ep_metrics_paths = [
            os.path.join(temp_dir.name, f"ep_metrics-{i}.json")
            for i in range(n)
        ]
    if output_sig_eroded_paths is None:
        output_sig_eroded_paths = [
            os.path.join(temp_dir.name, f"sig_eroded-{i}.nii.gz")
            for i in range(n)
        ]
    if output_mask_path is None:
        output_mask_path = os.path.join(temp_dir.name, "mask.nii.gz")
    if output_mask_eroded_path is None:
        output_mask_eroded_path = os.path.join(temp_dir.name, "mask_eroded.nii.gz")
    if output_ep_metrics_path is None:
        output_ep_metrics_path = os.path.join(temp_dir.name, "ep_metrics.json")
    if output_sig_eroded_path is None:
        output_sig_eroded_path = os.path.join(temp_dir.name, "sig_eroded.nii.gz")
    if input_dhcp_labels9_paths is not None and output_dhcp_labels9_path is None:
        output_dhcp_labels9_path = os.path.join(temp_dir.name, "dhcp_labels9.nii.gz")
    if input_dhcp_labels9_paths is None:
        input_dhcp_labels9_paths = n*[None]
    
    # Iter over stacks
    for i in range(n):
        
        # POC reconstruction
        _poc_reconstruction(
            input_pha_path=input_pha_paths[i],
            input_mask_path=input_mask_paths[i],
            output_sig_path=output_sig_paths[i],
            **kwargs,
        )
        
        # Erode mask
        erode_mask(
            input_mask_path=input_mask_paths[i],
            output_mask_path=output_mask_eroded_paths[i],
            **kwargs,
        )
        
        # Apply mask
        mrtrix_multiply(
            operand1=output_sig_paths[i],
            operand2=output_mask_eroded_paths[i],
            output_path=output_sig_eroded_paths[i],
        )
            
        # Measure EP
        extract_ep_metrics(
            input_sig_path=output_sig_paths[i],
            input_mask_eroded_path=output_mask_eroded_paths[i],
            output_ep_metrics_path=output_ep_metrics_paths[i],
            input_dhcp_labels9_path=input_dhcp_labels9_paths[i],
            **kwargs,
        )
    
    # Compute average mask
    mirtk_average_images(
        input_paths=input_mask_paths,
        input_dof_paths=input_dof_paths,
        input_ref_path=input_ref_path,
        output_path=output_mask_path,
        label=True,
    )
    
    # Compute average labels
    if output_dhcp_labels9_path is not None:
        mirtk_average_images(
            input_paths=input_dhcp_labels9_paths,
            input_dof_paths=input_dof_paths,
            input_ref_path=input_ref_path,
            output_path=output_dhcp_labels9_path,
            label=True,
        )
    
    # Compute average conductivity
    mirtk_average_images(
        input_paths=output_sig_paths,
        input_dof_paths=input_dof_paths,
        input_ref_path=input_ref_path,
        output_path=output_sig_path,
    )
    
    # Apply mask
    mrtrix_multiply(
        operand1=output_sig_path,
        operand2=output_mask_path,
        output_path=output_sig_path,
    )
    
    # Erode mask
    erode_mask(
        input_mask_path=output_mask_path,
        output_mask_path=output_mask_eroded_path,
        **kwargs,
    )
    
    # Apply mask
    mrtrix_multiply(
        operand1=output_sig_path,
        operand2=output_mask_eroded_path,
        output_path=output_sig_eroded_path,
    )
        
    # Measure EP
    extract_ep_metrics(
        input_sig_path=output_sig_path,
        input_mask_path=output_mask_path,
        input_mask_eroded_path=output_mask_eroded_path,
        output_ep_metrics_path=output_ep_metrics_path,
        input_dhcp_labels9_path=output_dhcp_labels9_path,
        **kwargs,
    )
        
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"MSPOC pipeline run time: {elapsed_time}")
        
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)