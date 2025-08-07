import os 
import datetime
import tempfile
import shutil

import nibabel as nib
import numpy as np

from scipy.ndimage import convolve, gaussian_filter
from scipy.linalg import svd

from utils.eprecon.fd_kernels import gradient_kernel, laplacian_kernel
from utils.tools.mrtrix import mrtrix_multiply
from utils.mask import erode_mask
from utils.metrics import extract_ep_metrics


def _magnitude_least_square(
        A, 
        b, 
        mls_max_iter=200, 
        mls_tol=1e-7,
        **kwargs,
    ):
    
    # Initial values
    b = np.abs(b.astype(np.float64))
    norm_target = np.linalg.norm(b)
    err = [1, 1]
    err_change = 1
    
    # Initial phase
    np.random.seed(42)
    z = np.exp(1j*2*np.pi*np.random.uniform(size=np.shape(b)))
    
    # Loop
    i = 0
    while i < mls_max_iter and err_change > mls_tol:
        
        # Solve system
        x = np.linalg.lstsq(A, b*z, rcond=None)[0]
        
        # Error
        e = np.linalg.norm(np.abs(A @ x) - b) / norm_target
        err[0] = err[1]
        err[1] = e
        err_change = err[0] - err[1]
        
        # Update phase
        z = np.exp(1j*np.angle(A @ x))
        i += 1
   
    # Normalization
    x /= np.linalg.norm(x)
        
    return A @ x


def _multi_coil_normalization(
        cmpl, 
        mask=None,
        coils_idx=None,
        ref_coil_idx=None,
        **kwargs,
    ):
    
    # Initialize mask
    if mask is None:
        mask = np.abs(np.mean(cmpl, axis=-1)) > 0
        
    # Data dimension
    (nx, ny, nz, nc) = np.shape(cmpl)

    # Compute reference coil
    if ref_coil_idx is None:
        
        # Extract coils
        if coils_idx is not None:
            cmpl = cmpl[..., coils_idx]
        
        # Create system
        A = cmpl[mask]
        b = mask[mask]

        # Solve system
        ref = np.zeros((nx, ny, nz), dtype=np.complex128)
        ref[mask] = _magnitude_least_square(A=A, b=b, **kwargs)
        
    else:
        
        # Extract ref coil
        ref = cmpl[..., ref_coil_idx]
        
        # Extract coils
        if coils_idx is None:
            coils_idx = [i for i in range(nc) if i != ref_coil_idx]
        cmpl = cmpl[..., coils_idx]

    # Divide by reference
    ref[ref == 0] = 1 # prevent NaN
    cmpl /= ref[..., None]
    
    # Apply mask
    cmpl *= mask[..., None]

    return cmpl


def _gaussian_filter(
        cmpl, #4D
        mask, #3D or 4D
        gs_sigma=1.0,
        vox=[1.0,1.0,1.0],
        gs_axes=[0,1,2],
        **kwargs,
    ):
    
    # Data dims
    d = np.ndim(cmpl) 
    if np.ndim(mask) == d-1:
        mask = mask[..., None]
    
    if gs_sigma > 0.0:
        
        # Set sigma
        sigma = [0.0]*d
        for ax in gs_axes:
            sigma[ax] = gs_sigma / vox[ax] # in voxels
            
        cmpl = (gaussian_filter(np.real(cmpl), sigma) 
                + 1j * gaussian_filter(np.imag(cmpl), sigma))

        # Filter correction at border
        mask_filtered = gaussian_filter(mask, sigma)
        mask_filtered[mask_filtered == 0] = 1 # Prevent NaN
    
        # Apply mask
        cmpl /= mask_filtered
        cmpl *= mask
        
    return cmpl


def _svd(
        cmpl, 
        n_svd=None,
        **kwargs,
    ):
    
    # Get image dim
    nx, ny, nz, nc = np.shape(cmpl)
    
    # Apply SVD
    if n_svd is not None and n_svd >= 3:
        cmpl = np.reshape(cmpl, [nx*ny*nz, nc]) # flatten each
        u, _ ,_ = svd(cmpl, full_matrices=False)
        cmpl = np.reshape(u, [nx, ny, nz, nc])
        cmpl = cmpl[:,:,:,:n_svd] # Keep largest modes
        
    return cmpl


def _saep_solver(
        cmpl, 
        f0=128e6, # 3 Tesla
        vox=[1.0,1.0,1.0],
        **kwargs,
    ):
    
    # Physical constants
    w0 = 2*np.pi*f0
    mu0 = 4*np.pi*1e-7
    eps0 = 8.85e-12
    
    # Data dimensions
    dims = list(np.shape(cmpl))
    nx, ny, nz, nc = dims
    
    # Load Gradient and Laplacian kernels
    k_grad_x, k_grad_y, k_grad_z = gradient_kernel(vox=vox)
    k_del2_x, k_del2_y, k_del2_z = laplacian_kernel(vox=vox)  
    
    # Apply Central Finite Difference method to compute gradient over channels
    grad = np.zeros(dims+[3], dtype=np.cdouble)
    for c in range(nc):
        grad[..., c, 0] = convolve(cmpl[..., c], k_grad_x) 
        grad[..., c, 1] = convolve(cmpl[..., c], k_grad_y)
        grad[..., c, 2] = convolve(cmpl[..., c], k_grad_z)
        
    # Apply Central Finite Difference method to compute Laplacian over channels
    del2 = np.zeros(dims, dtype=np.cdouble)
    for c in range(nc):
        del2[..., c] = (convolve(cmpl[..., c], k_del2_x)
                        + convolve(cmpl[..., c], k_del2_y) 
                        + convolve(cmpl[..., c], k_del2_z))
    
    # Build equations
    b = np.reshape(del2, [nx*ny*nz, nc])
    b = np.transpose(b, [1, 0])
    A = np.reshape(grad, [nx*ny*nz, nc, 3])
    A = -2*np.transpose(A, [1, 2, 0])
    X = np.zeros([nx*ny*nz, 3], dtype=np.cdouble)
    
    # Solve equation at each voxel location
    for i in range(nx*ny*nz):
        b_i = b[..., i] # ncx1
        A_i = A[..., i] # ncx3
        if not np.all(A_i) or not np.all(b_i):
            continue
        X[i, ...] = np.linalg.lstsq(A_i, b_i, rcond=None)[0] # 3x1

    # Reshape to 3D
    X = np.reshape(X, [nx, ny, nz, 3])
    
    # Apply Central Finite Difference method to compute divergence
    div = (convolve(X[..., 0], k_grad_x) 
           + convolve(X[..., 1], k_grad_y) 
           + convolve(X[..., 2], k_grad_z))
    
    # Build final equation
    EP = np.sum(X * X, axis=-1) + div
    
    # Extract EP maps
    eps = -np.real(EP)/(mu0*w0**2)/eps0
    sig = np.imag(EP)/(mu0*w0)
    
    return sig, eps


def _saep_reconstruction(
        input_mag_path,
        input_pha_path,
        input_mask_path,
        output_sig_path,
        output_eps_path,
        **kwargs
    ):
    
    # Load niftis
    mag_nii = nib.load(input_mag_path)
    pha_nii = nib.load(input_pha_path)
    mask_nii = nib.load(input_mask_path)
    
    # Extract header and affine
    header = mag_nii.header
    affine = mag_nii.affine

    # Load data
    mag = mag_nii.get_fdata()
    pha = pha_nii.get_fdata()
    mask = mask_nii.get_fdata().astype(bool)
    
    # Get resolution and dimmension
    vox = header["pixdim"][1:4].astype(float) # in mm
    
    # To complex
    cmpl = mag * np.exp(1j * pha)
    
    # Apply mask
    cmpl *= mask[..., None]
    
    # Multi coil normalization
    cmpl = _multi_coil_normalization(cmpl=cmpl, mask=mask, **kwargs)
    
    # Apply Gaussian smoothing
    cmpl = _gaussian_filter(cmpl=cmpl, mask=mask, vox=vox, **kwargs)

    # SVD
    cmpl = _svd(cmpl=cmpl, **kwargs)
        
    # Apply mask
    cmpl *= mask[..., None]
    
    # SAEP solver
    vox = vox * 1e-3 # in m
    sig, eps = _saep_solver(cmpl=cmpl, vox=vox, **kwargs)
    
    # Apply mask
    sig *= mask
    eps *= mask

    # Initialize output directory
    output_dir = os.path.dirname(output_sig_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Save conductivity
    sig_nii = nib.Nifti1Image(sig, affine, header)
    nib.save(sig_nii, output_sig_path)

    # Initialize output directory
    output_dir = os.path.dirname(output_eps_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Save permittivity
    eps_nii = nib.Nifti1Image(eps, affine, header)
    nib.save(eps_nii, output_eps_path)
    

def saep_pipeline(
        input_mag_path,
        input_pha_path,
        input_mask_path,
        output_sig_path=None,
        output_eps_path=None,
        output_ep_metrics_path=None,
        output_mask_eroded_path=None,
        output_sig_eroded_path=None,
        output_eps_eroded_path=None,
        verbose=None,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
    
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()
    if output_sig_path is None:
        output_sig_path = os.path.join(temp_dir.name, "sig.nii.gz")
    if output_eps_path is None:
        output_eps_path = os.path.join(temp_dir.name, "eps.nii.gz")
    if output_ep_metrics_path is None:
        output_ep_metrics_path = os.path.join(temp_dir.name, "ep_metrics.json")
    if output_mask_eroded_path is None:
        output_mask_eroded_path = os.path.join(temp_dir.name, "mask_eroded.nii.gz")
    if output_sig_eroded_path is None:
        output_sig_eroded_path = os.path.join(temp_dir.name, "sig_eroded.nii.gz")
    if output_eps_eroded_path is None:
        output_eps_eroded_path = os.path.join(temp_dir.name, "eps_eroded.nii.gz")
        
    # SAEP reconstruction
    _saep_reconstruction(
        input_mag_path=input_mag_path,
        input_pha_path=input_pha_path,
        input_mask_path=input_mask_path,
        output_sig_path=output_sig_path,
        output_eps_path=output_eps_path,
        **kwargs,
    )
    
    # Erode mask
    erode_mask(
        input_mask_path=input_mask_path,
        output_mask_path=output_mask_eroded_path,
        **kwargs,
    )
    
    # Apply mask on conductivity
    mrtrix_multiply(
        operand1=output_sig_path,
        operand2=output_mask_eroded_path,
        output_path=output_sig_eroded_path,
    )
    
    # Apply mask on permittivity
    mrtrix_multiply(
        operand1=output_eps_path,
        operand2=output_mask_eroded_path,
        output_path=output_eps_eroded_path,
    )
   
    # Compute EP
    extract_ep_metrics(
        input_sig_path=output_sig_path,
        input_eps_path=output_eps_path,
        input_mask_eroded_path=output_mask_eroded_path,
        output_ep_metrics_path=output_ep_metrics_path,
        **kwargs,
    )
        
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"SAEP pipeline run time: {elapsed_time}")
    
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)