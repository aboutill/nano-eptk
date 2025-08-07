import os
import datetime
import tempfile
import shutil

import nibabel as nib
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from nibabel.affines import rescale_affine

from utils.tools.mrtrix import (
    mrtrix_degibbs,
    mrtrix_polar, 
    mrtrix_real,
    mrtrix_imag, 
    mrtrix_complex,
    mrtrix_abs,
    mrtrix_phase,
    mrtrix_mean,
    mrtrix_extract,
)


def degibbs(
        input_mag_path, 
        input_pha_path,
        output_mag_path, 
        output_pha_path,
        k_window=None,
        verbose=False,
        axes=[0,1,2],
    ):

    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
    
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()

    # Temporary files
    real_ext = "real.nii.gz"
    imag_ext = "imag.nii.gz"
    cmpl_ext = "cmpl.nii.gz"
    
    # No k-space truncation
    if k_window is None:
        
        # To complex
        cmpl_path = os.path.join(temp_dir.name, cmpl_ext)
        mrtrix_polar(
            mag_path=input_mag_path, 
            pha_path=input_pha_path, 
            cmpl_path=cmpl_path,
        )
        
        # mrtrix mrdegibbs
        mrtrix_degibbs(
            input_path=cmpl_path,
            output_path=cmpl_path,
            axes=axes,
        )
         
        # Output degibbs magnitude
        mrtrix_abs(
            cmpl_path=cmpl_path,
            abs_path=output_mag_path,
        )
        
        # Output degibbs phase  
        mrtrix_phase(
            cmpl_path=cmpl_path,
            phase_path=output_pha_path,
        )
        
    # k-space truncation
    else:
        
        # Load magnitude and phase
        mag_nii = nib.load(input_mag_path)
        pha_nii = nib.load(input_pha_path)
        
        # Extract header and affine
        header = mag_nii.header
        affine = mag_nii.affine
        
        # Get resolution and dimension
        dx, dy, dz = header["pixdim"][1:4]
        nx, ny, nz, nc = header["dim"][1:5]
        
        # Extract images
        mag = np.array(mag_nii.get_fdata())
        pha = np.array(pha_nii.get_fdata())
        
        # Transform image representation to k-space
        cmpl = mag*np.exp(1j*pha)
        cmpl_kspace = np.fft.fftn(cmpl, axes=(0,1,2))
        cmpl_kspace = np.fft.fftshift(cmpl_kspace, axes=(0,1,2))
        
        # Truncate k-space
        cmpl_kspace = cmpl_kspace[
            k_window[0]:k_window[1],
            k_window[2]:k_window[3],
            k_window[4]:k_window[5],
        ]
            
        # Reconstruct image
        cmpl_kspace = np.fft.ifftshift(cmpl_kspace, axes=(0,1,2))
        cmpl = np.fft.ifftn(cmpl_kspace, axes=(0,1,2))
        
        # Real and imaginary parts
        real = np.real(cmpl)
        imag = np.imag(cmpl)
        
        # Modify header and affine
        header_acquired = header.copy()
        header_acquired["dim"][1:4] = np.shape(cmpl)[:3]
        header_acquired["pixdim"][1:4] = [
            header_acquired["pixdim"][i] * header["dim"][i]
            / header_acquired["dim"][i] for i in range(1,4)
        ]
        
        affine_acquired = rescale_affine(
            affine, 
            header["dim"][1:4], 
            header_acquired["pixdim"][1:4], 
            header_acquired["dim"][1:4],
        )
    
        # Save real part
        real_nii = nib.Nifti1Image(real, affine_acquired, header_acquired)
        real_path = os.path.join(temp_dir.name, real_ext)
        nib.save(real_nii, real_path)
    
        # Save imaginary part
        imag_nii = nib.Nifti1Image(imag, affine_acquired, header_acquired)
        imag_path = os.path.join(temp_dir.name, imag_ext)
        nib.save(imag_nii, imag_path)
        
        # To complex
        cmpl_path = os.path.join(temp_dir.name, cmpl_ext)
        mrtrix_complex(
            real_path=real_path,
            imag_path=imag_path,
            cmpl_path=cmpl_path,
        )
    
        # mrtrix mrdegibbs
        mrtrix_degibbs(
            input_path=cmpl_path,
            output_path=cmpl_path,
            axes=axes,
        )
        
        # To real and imaginary parts
        mrtrix_real(
            cmpl_path=cmpl_path, 
            real_path=real_path,
        )
        mrtrix_imag(
            cmpl_path=cmpl_path, 
            imag_path=imag_path,
        )
        
        # Load degibbs real and imaginary parts
        real_nii = nib.load(real_path)
        imag_nii = nib.load(imag_path)
    
        # Load matrix
        real = np.array(real_nii.get_fdata())
        imag = np.array(imag_nii.get_fdata())
        
        # Resampling
        # Build image grid
        xi = np.linspace(0, nx-1, nx)*dx
        yi = np.linspace(0, ny-1, ny)*dy
        zi = np.linspace(0, nz-1, nz)*dz
        
        # Dimension and resolution of reconstructed image grid
        (nxx, nyy, nzz) = header_acquired['dim'][1:4]
        (dxx, dyy, dzz) = header_acquired['pixdim'][1:4]
        
        # Build image grid
        xj = np.linspace(0, nxx-1, nxx)*dxx
        yj = np.linspace(0, nyy-1, nyy)*dyy
        zj = np.linspace(0, nzz-1, nzz)*dzz
    
        # Prevent error at borders
        xj[0], xj[-1] = xi[0], xi[-1]
        yj[0], yj[-1] = yi[0], yi[-1]
        zj[0], zj[-1] = zi[0], zi[-1] 
    
        # Build meshgrid
        (xii, yii, zii) = np.meshgrid(xi, yi, zi, indexing='ij')   
       
        if nc > 1:
            # Interpolate to original grid
            real_temp = np.zeros((nx, ny, nz, nc))
            imag_temp = np.zeros((nx, ny, nz, nc))
            
            # Iter over coils
            for i in range(nc):
                # Resample real part
                interp = RegularGridInterpolator(
                    (xj, yj, zj), 
                    real[:,:,:,i], 
                    method="linear",
                )
                real_temp[:,:,:,i] = interp((xii, yii, zii))
                
                # Resample imaginary part
                interp = RegularGridInterpolator(
                    (xj, yj, zj), 
                    imag[:,:,:,i],
                    method="linear",
                )
                imag_temp[:,:,:,i] = interp((xii, yii, zii))
        else:
            # Interpolate to original grid
            real_temp = np.zeros((nx, ny, nz))
            imag_temp = np.zeros((nx, ny, nz))
            
            # Resample real part
            interp = RegularGridInterpolator(
                (xj, yj, zj), 
                real, 
                method="linear",
            )
            real_temp = interp((xii, yii, zii))
            
            # Resample imaginary part
            interp = RegularGridInterpolator(
                (xj, yj, zj), 
                imag,
                method="linear",
            )
            imag_temp = interp((xii, yii, zii))
        
        # Complex image
        cmpl = real_temp + 1j * imag_temp
        
        # Extract magnitude and phase
        mag = np.abs(cmpl)
        pha = np.angle(cmpl)
        
        # Initialize output directory
        output_dir = os.path.dirname(output_mag_path)
        os.makedirs(output_dir, exist_ok=True)
            
        # Save degibbs magnitude
        mag_nii = nib.Nifti1Image(mag, affine, header)
        nib.save(mag_nii, output_mag_path)
    
        # Initialize output directory
        output_dir = os.path.dirname(output_pha_path)
        os.makedirs(output_dir, exist_ok=True)
            
        # Save degibbs phase
        pha_nii = nib.Nifti1Image(pha, affine, header)
        nib.save(pha_nii, output_pha_path)
        
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Degibbs run time: {elapsed_time}")
        
    # Delete temp folder    
    shutil.rmtree(temp_dir.name)
    
    
def extract_reference_coil(
        input_mag_path,
        output_mag_path,
        ref_coil_idx=None,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_mag_path)
    os.makedirs(output_dir, exist_ok=True)
        
    if ref_coil_idx is None:
        
        # Extract mean over channels
        mrtrix_mean(
            input_path=input_mag_path,
            output_path=output_mag_path,
        )
    
    else:
        
        # Extract channel
        mrtrix_extract(
            input_path=input_mag_path,
            output_path=output_mag_path,
            idx=ref_coil_idx,
        )