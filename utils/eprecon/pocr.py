import os 
import datetime
import json
import tempfile
import shutil

import nibabel as nib
import numpy as np

from scipy.ndimage import (
    convolve,
    binary_erosion, 
    binary_dilation, 
    gaussian_filter,
)
from scipy.sparse import spdiags
from scipy.sparse.linalg import gmres
from scipy.special import erfcinv
from scipy.interpolate import RBFInterpolator
from skimage.morphology import ball

from utils.eprecon.fd_kernels import gradient_kernel, laplacian_kernel
from utils.tools.mirtk import mirtk_average_images
from utils.tools.mrtrix import mrtrix_multiply
from utils.mask import erode_mask
from utils.metrics import extract_ep_metrics, extract_outlier_metrics


def _gaussian_filter(
        img,
        mask,
        gs_sigma=1.0,
        vox=[1.0,1.0,1.0],
        gs_axes=[0,1,2],
        **kwargs,
    ):
    
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


def _pocr_solver(
        pha, 
        mask, 
        vox=[1.0,1.0,1.0],
        f0=128e6, # 3 Tesla 
        pde_axes=[0,1,2],
        diff_reg_w=0.05,
        diff_reg_axes=[0,1,2],
        gmres_atol=1e-4,
        gmres_restart=30,
        gmres_maxiter=50,
        **kwargs,
    ):
    
    # Physical constants
    w0 = 2*np.pi*f0
    mu0 = 4*np.pi*1e-7
    c = diff_reg_w # Shorter variable name...
    
    # Get resolution and shape
    dx, dy, dz = vox
    nx, ny, nz = np.shape(pha)
    
    # Remove border voxels
    mask = binary_erosion(mask)
    mask[0,:,:] = 0
    mask[-1,:,:] = 0
    mask[:,0,:] = 0
    mask[:,-1,:] = 0
    mask[:,:,0] = 0
    mask[:,:,-1] = 0
    
    # Load Gradient and Laplacian kernels
    k_grad_x, k_grad_y, k_grad_z = gradient_kernel(vox)
    k_del2_x, k_del2_y, k_del2_z = laplacian_kernel(vox)  
    
    # Compute gradients and Laplacian
    grad_x = convolve(pha, k_grad_x)
    grad_y = convolve(pha, k_grad_y)
    grad_z = convolve(pha, k_grad_z)
    del2_x = convolve(pha, k_del2_x)
    del2_y = convolve(pha, k_del2_y)
    del2_z = convolve(pha, k_del2_z)

    # Number of voxels in 3D volume
    n = nx*ny*nz
    
    # Number of voxels inside ROI
    m = np.count_nonzero(mask)
    
    # Flatten matrices (Row major)
    grad_x = grad_x.flatten()
    grad_y = grad_y.flatten()
    grad_z = grad_z.flatten()
    del2_x = del2_x.flatten()
    del2_y = del2_y.flatten()
    del2_z = del2_z.flatten()
    mask = mask.flatten()

    # Build equations # All axes combination are not implemented
    t = grad_x/(2*dx) - c/(dx**2)
    u = grad_y/(2*dy) - c/(dy**2)
    if pde_axes == [0,1,2]:
        v = grad_z/(2*dz) - c/(dz**2)
        w = del2_x + del2_y + del2_z + 2*c/(dx**2) + 2*c/(dy**2) + 2*c/(dz**2)
        x = -grad_z/(2*dz) - c/(dz**2)
    elif pde_axes == [0,1]:
        if diff_reg_axes == [0,1,2]:
            v = - c/(dz**2) * np.ones_like(grad_z)
            w = del2_x + del2_y + 2*c/(dx**2) + 2*c/(dy**2) + 2*c/(dz**2)
            x = - c/(dz**2) * np.ones_like(grad_z)
        elif diff_reg_axes == [0,1]:
            w = del2_x + del2_y + 2*c/(dx**2) + 2*c/(dy**2)
    y = -grad_y/(2*dy) - c/(dy**2)
    z = -grad_x/(2*dx) - c/(dx**2)
    
    # Build sparse matrix
    if pde_axes == [0,1,2]:
        data = [t, u, v, w, x, y, z]
        diags = [-nz*ny, -nz, -1, 0, 1, nz, nz*ny]
        
    elif pde_axes == [0,1]:
        if diff_reg_axes == [0,1,2]:
            data = [t, u, v, w, x, y, z]
            diags =  [-nz*ny, -nz, -1, 0, 1, nz, nz*ny]
        elif diff_reg_axes == [0,1]:
            data = [t, u, w, y, z]
            diags = [-nz*ny, -nz, 0, nz, nz*ny]
        
    A = spdiags(data, diags, n, n, "csr")
    A = np.transpose(A)
    
    # Only ROI
    A = A[mask,:]
    A = A[:,mask]

    # Build vector b
    b = 2*w0*mu0*np.ones(m)
    
    # Apply solver
    rho = gmres(A, b, atol=gmres_atol, restart=gmres_restart, maxiter=gmres_maxiter)[0]
    
    # Get conductivity inside ROI
    sig = np.zeros(n)
    sig[mask] = np.divide(np.ones(m), rho)
    
    # 2D PDE correction
    if pde_axes == [0,1]:
        sig *= 1.5
    
    # Reshape to 3D matrix
    sig = np.reshape(sig, (nx, ny, nz))
    mask = np.reshape(mask, (nx, ny, nz))
    
    return sig, mask


def _create_mask_outlier(
        sig,
        mask,
        dil_size=1,
        **kwargs,
    ):
    
    # Compute Median Absolute Deviation
    c = -1/(np.sqrt(2)*erfcinv(3/2))
    MED = np.median(sig[mask])
    MAD = c*np.median(np.abs(sig[mask] - MED))
    
    # Create artefact mask
    mask_outlier = np.zeros(np.shape(sig), dtype=bool)
    mask_outlier[np.abs(sig-MED) > 3*MAD] = True
    mask_outlier *= mask
    
    # Dilate outlier mask
    if dil_size > 0:
        mask_outlier = binary_dilation(mask_outlier, structure=ball(dil_size))
    mask_outlier *= mask #  Prevent singular matrix error
    
    return mask_outlier

    
def _b_spline_post_processing(
        sig,
        mask,
        mask_outlier,
        vox=[1.0,1.0,1.0],
        bsp_n=100,
        bsp_axes=[0,1,2],
        sig_thr=[0.0,2.5],
        **kwargs,
    ):
    
    # Image dimension and resolution
    nx, ny, nz = np.shape(sig)
    dx, dy, dz = vox
    
    # 3D case
    if bsp_axes == [0,1,2]:
        # Sample mask and conductivity
        mask_f = mask.flatten()
        mask_outlier_f = mask_outlier.flatten()
        mask_f = np.logical_xor(mask_f, mask_outlier_f)
        sig_f = sig.flatten()
        
        # Create grid
        grid = np.mgrid[0:nx, 0:ny, 0:nz]
        grid = grid.reshape(3, -1).T.astype(np.float64)
        grid[:,0] *= dx
        grid[:,1] *= dy
        grid[:,2] *= dz    
        
        # Apply mask on sampled grid and conductivity
        grid_outlier = grid[mask_outlier_f, :]
        grid = grid[mask_f, :]
        sig_f = sig_f[mask_f]
        
        # B-spline correction
        try:
            interp = RBFInterpolator(
                grid, 
                sig_f, 
                neighbors=bsp_n, 
                kernel="thin_plate_spline",
            )
            sig[mask_outlier] = interp(grid_outlier)
            
        except:
            print("RBF interpolator error!")
        
    # 2D case
    if bsp_axes == [0,1]:
        for z in range(nz):
            
            # Extract slice
            sig_z = sig[..., z]
            mask_z = mask[..., z]
            mask_outlier_z = mask_outlier[..., z]
            
            # No artefact in slice
            if not np.sum(mask_outlier_z): 
                continue
               
            sig_z_f = sig_z.flatten()
            mask_z_f = mask_z.flatten()
            mask_outlier_z_f = mask_outlier_z.flatten()
            mask_z_f = np.logical_xor(mask_z_f, mask_outlier_z_f)
            
            # Create grid
            grid_z = np.mgrid[0:nx, 0:ny]
            grid_z = grid_z.reshape(2, -1).T.astype(np.float64)
            grid_z[:,0] *= dx
            grid_z[:,1] *= dy
            
            # Apply mask on sampled grid and conductivity
            grid_outlier_z = grid_z[mask_outlier_z_f, :]
            grid_z = grid_z[mask_z_f, :]
            sig_z_f = sig_z_f[mask_z_f]
            
            # B spline correction
            try:
                interp = RBFInterpolator(
                    grid_z,
                    sig_z_f, 
                    neighbors=bsp_n, 
                    kernel="thin_plate_spline",
                )
                sig_z[mask_outlier_z] = interp(grid_outlier_z)
    
                # Update slice
                sig[..., z] = sig_z
                
            except:
                print("RBF interpolator error!")
    
    # Threshold conductivity
    mask_outlier[sig<sig_thr[0]] = True
    mask_outlier[sig>sig_thr[1]] = True
    
    sig[sig<sig_thr[0]] = sig_thr[0]
    sig[sig>sig_thr[1]] = sig_thr[1]
    
    # Apply mask
    sig *= mask
    mask_outlier *= mask
    
    return sig, mask_outlier

     
def _outlier_stack_computation(
        input_sig_paths,
        input_mask_paths,
        input_mask_outlier_paths,
        output_outlier_metrics_paths,
    ):

    # Init array
    n = len(input_sig_paths)
    
    # Iter over stacks
    for i in range(n):
        # EP measure on whole mask
        extract_outlier_metrics(
            input_mask_path=input_mask_paths[i],
            input_mask_outlier_path=input_mask_outlier_paths[i],
            output_outlier_metrics_path=output_outlier_metrics_paths[i],
        )
        

def _outlier_stack_rejection(
        input_outlier_metrics_paths,
        rel_vol_threshold=0.3,
        **kwargs,
    ):
    
    # Init array
    n = len(input_outlier_metrics_paths)
    mask_outlier_rel_vols = []
    
    # Iter over stacks
    for i in range(n):

        # Load mask outlier relative volume
        outlier_metrics = json.load(open(input_outlier_metrics_paths[i]))
        mask_outlier_rel_vols += [outlier_metrics["REL_VOL"]]

    # Inlier array
    inlier = [mask_outlier_rel_vol <= rel_vol_threshold
               for mask_outlier_rel_vol in mask_outlier_rel_vols]

    # Init arrays
    inlier_index = []
    
    # Iter over stacks
    for i in range(n):
        if inlier[i]:
            inlier_index += [i]
    
    # If no inlier, no rejection
    if len(inlier_index) == 0:
        inlier_index = [i for i in range(n)]

    # If only one stack, make it double
    elif len(inlier_index) == 1:
        inlier_index += [inlier_index[0]]
            
    return inlier_index


def _pocr_reconstruction(
        input_pha_path,
        input_mask_path,
        output_sig_path,
        output_mask_outlier_path=None,
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
    
    # Apply Gaussian smoothing
    pha = _gaussian_filter(img=pha, mask=mask, vox=vox, **kwargs)
                
    # Solve POCR
    vox = vox * 1e-3 # in m
    sig, mask = _pocr_solver(pha=pha, mask=mask, vox=vox, **kwargs)
    
    # Create outlier mask
    mask_outlier = _create_mask_outlier(sig=sig, mask=mask, **kwargs)
    
    # Apply post-processing
    sig, mask_outlier = _b_spline_post_processing(
        sig=sig,
        mask=mask,
        mask_outlier=mask_outlier,
        vox=vox,
    )
    
    # Initialize output directory
    output_dir = os.path.dirname(output_sig_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Save sig
    sig_nii = nib.Nifti1Image(sig, affine, header)
    nib.save(sig_nii, output_sig_path)
    
    if output_mask_outlier_path is not None:
        # Initialize output directory
        output_dir = os.path.dirname(output_mask_outlier_path)
        os.makedirs(output_dir, exist_ok=True)
            
        # Save mask outlier
        mask_outlier_nii = nib.Nifti1Image(mask_outlier.astype(np.float32), affine, header)
        nib.save(mask_outlier_nii, output_mask_outlier_path)
    

def pocr_pipeline(
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
        
    # POCR reconstruction
    _pocr_reconstruction(
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
        print(f"POCR pipeline run time: {elapsed_time}")
        
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)
    
        
def mspocr_pipeline(
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
    output_mask_outlier_paths = [
        os.path.join(temp_dir.name, f"mask_outlier-{i}.nii.gz")
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
    output_outlier_metrics_paths = [
        os.path.join(temp_dir.name, f"outlier_metrics-{i}.json")
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
        
        # POCR reconstruction
        _pocr_reconstruction(
            input_pha_path=input_pha_paths[i],
            input_mask_path=input_mask_paths[i],
            output_sig_path=output_sig_paths[i],
            output_mask_outlier_path=output_mask_outlier_paths[i],
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
            input_dhcp_labels9_path=input_dhcp_labels9_paths[i],
            output_ep_metrics_path=output_ep_metrics_paths[i],
            **kwargs,
        )
            
    # Stack outlier computation
    _outlier_stack_computation(
        input_sig_paths=output_sig_paths,
        input_mask_paths=input_mask_paths,
        input_mask_outlier_paths=output_mask_outlier_paths,
        output_outlier_metrics_paths=output_outlier_metrics_paths,
    )
    
    # Stack outlier rejection
    inlier_index = _outlier_stack_rejection(
        input_outlier_metrics_paths=output_outlier_metrics_paths,
        **kwargs,
    )
    input_dof_paths = [input_dof_paths[idx] for idx in inlier_index]
    output_sig_paths = [output_sig_paths[idx] for idx in inlier_index]
    input_mask_paths = [input_mask_paths[idx] for idx in inlier_index]
    output_mask_outlier_paths = [output_mask_outlier_paths[idx] for idx in inlier_index]
    input_dhcp_labels9_paths = [input_dhcp_labels9_paths[idx] for idx in inlier_index]
        
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
        input_dhcp_labels9_path=output_dhcp_labels9_path,
        output_ep_metrics_path=output_ep_metrics_path,
    )
        
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"MSPOCR pipeline run time: {elapsed_time}")
    
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)
    

