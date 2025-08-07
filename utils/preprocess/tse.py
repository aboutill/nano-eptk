import os
import datetime
import tempfile
import json
import shutil

import nibabel as nib
import numpy as np

from utils.tools.mrtrix import (
    mrtrix_polar, 
    mrtrix_real,
    mrtrix_imag, 
    mrtrix_complex,
    mrtrix_abs,
    mrtrix_phase,
    mrtrix_multiply,
)
from utils.tools.mirtk import mirtk_average_images


def _select_stacks(
        input_mag_paths,
        input_pha_paths,
        input_mask_paths,
        input_dof_paths,
        output_info_path,
    ):
    
    # Init dict
    info = {}
    
    # Init dicts
    axial_dict = None
    sagittal_dict = None
    coronal_dict = None
    
    # Number of stacks
    n = len(input_mag_paths)
    
    # Iterate over stacks
    for i in range(n):
        
        # Set path
        input_mag_path = input_mag_paths[i]
        input_pha_path = input_pha_paths[i]
        input_mask_path = input_mask_paths[i]
        input_dof_path = input_dof_paths[i]
        
        # Load magnitude
        mag_nii = nib.load(input_mag_path)
        mag = np.array(mag_nii.get_fdata())
        
        # Affine 
        affine = mag_nii.affine
        
        # Get slice direction
        # This assumes axis 2 is slice direction
        axcodes = nib.orientations.aff2axcodes(affine)
        axcodes = "".join(axcodes).lower()
        if axcodes.endswith("s") or axcodes.endswith("i"):
            # Transverse/axial slice direction, 3rd axis is superior-inferior
            slice_direction = "axial"
        elif axcodes.endswith("p") or axcodes.endswith("a"):
            # Coronal slice direction, 3rd axis is posterior-anterior
            slice_direction = "coronal"
        elif axcodes.endswith("r") or axcodes.endswith("l"):
            # Sagittal slice direction, 3rd axis is right-left
            slice_direction = "sagittal"
        else:
            continue
        
        # Extract dimension
        dim = mag_nii.header["dim"]
        
        # Compute slice-by-slice smoothness in z-direction 
        z_smooth = np.std([
            np.mean(np.abs(mag[:,:,z] - mag[:,:,z+1])) 
            for z in range(0, dim[3]-1)
        ]) 
        
        # Stack info
        stack_info = {
            "mag": input_mag_path,
            "pha": input_pha_path,
            "mask": input_mask_path,
            "z_smooth": z_smooth,
            "transform": input_dof_path,
            "slice_direction": slice_direction,
        }
        
        # Update info
        info[i] = stack_info
        
        # Update axial dir
        if slice_direction == "axial":
            if axial_dict is None:
                
                # Initialize axial dict
                axial_dict = stack_info
            
            else:
                # Update axial dict if lower z-smooth in new stack
                if z_smooth < axial_dict["z_smooth"]:
                    axial_dict = stack_info
                    
        elif slice_direction == "sagittal":
            if sagittal_dict is None:
                
                # Initialize sagittal dict
                sagittal_dict = stack_info
            
            else:
                # Update sagittal dict if lower z-smooth in new stack
                if z_smooth < sagittal_dict["z_smooth"]:
                    sagittal_dict = stack_info
                    
        elif slice_direction == "coronal":
            if coronal_dict is None:
                
                # Initialize sagittal dict
                coronal_dict = stack_info
            
            else:
                # Update sagittal dict if lower z-smooth in new stack
                if z_smooth < coronal_dict["z_smooth"]:
                    coronal_dict = stack_info
    
    # Initialize output directory
    output_dir = os.path.dirname(output_info_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Write output json file
    if axial_dict:
        info["axial"] = axial_dict
    if sagittal_dict:
        info["sagittal"] = sagittal_dict
    if coronal_dict:
        info["coronal"] = coronal_dict
    with open(output_info_path, "w") as f:
        json.dump(info, f, indent=4)
        

def average_stacks(
        input_mag_paths,
        input_pha_paths,
        input_mask_paths,
        input_dof_paths,
        output_mag_path,
        output_pha_path,
        output_mask_path,
        output_info_path=None,
        verbose=False,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
        
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()
    if output_info_path is None:
        output_info_path = os.path.join(temp_dir.name, "info.json")
        
    # Temporary file extension
    cmpl_ext = "cmpl.nii.gz"
    real_ext = "real.nii.gz"
    imag_ext = "imag.nii.gz"
    
    #
    ornts = ["axial", "sagittal", "coronal"]
    
    # Stack selection
    _select_stacks(
        input_mag_paths=input_mag_paths,
        input_pha_paths=input_pha_paths,
        input_mask_paths=input_mask_paths,
        input_dof_paths=input_dof_paths,
        output_info_path=output_info_path,
    )
    
    #
    real_paths = []
    imag_paths = []
    mask_paths = []
    dof_paths = []
    
    # Load stack selection
    info = json.load(open(output_info_path))
    
    #
    for ornt in ornts:
        if ornt in info:
            ornt_info = info[ornt]
        else:
            continue
        
        # Set paths
        ornt_mag_path = ornt_info["mag"]
        ornt_pha_path = ornt_info["pha"]
        ornt_mask_path = ornt_info["mask"]
        ornt_dof_path = ornt_info["transform"]
        
        # To complex
        ornt_cmpl_path = os.path.join(temp_dir.name, f"{ornt}_{cmpl_ext}")
        mrtrix_polar(
            mag_path=ornt_mag_path,
            pha_path=ornt_pha_path,
            cmpl_path=ornt_cmpl_path,
        )
        
        # Extract real part
        ornt_real_path = os.path.join(temp_dir.name, f"{ornt}_{real_ext}")
        mrtrix_real(
            cmpl_path=ornt_cmpl_path,
            real_path=ornt_real_path,
        )
        
        # Extract imag part
        ornt_imag_path = os.path.join(temp_dir.name, f"{ornt}_{imag_ext}")
        mrtrix_imag(
            cmpl_path=ornt_cmpl_path,
            imag_path=ornt_imag_path,
        )
        
        #
        real_paths += [ornt_real_path]
        imag_paths += [ornt_imag_path]
        mask_paths += [ornt_mask_path]
        dof_paths += [ornt_dof_path]
        
    # Average real parts
    real_path = os.path.join(temp_dir.name, real_ext)
    mirtk_average_images(
        input_paths=real_paths,
        input_dof_paths=dof_paths,
        output_path=real_path,
    )
    
    # Average image parts
    imag_path = os.path.join(temp_dir.name, imag_ext)
    mirtk_average_images(
        input_paths=imag_paths,
        input_dof_paths=dof_paths,
        output_path=imag_path,
    )
    
    # To complex
    cmpl_path = os.path.join(temp_dir.name, cmpl_ext)
    mrtrix_complex(
        real_path=real_path,
        imag_path=imag_path,
        cmpl_path=cmpl_path,
    )
    
    # Average mask
    mirtk_average_images(
        input_paths=mask_paths,
        input_dof_paths=dof_paths,
        output_path=output_mask_path,
    )
        
    # Output magnitude
    mrtrix_abs(
        cmpl_path=cmpl_path,
        abs_path=output_mag_path,
    )
    
    # Aply mask
    mrtrix_multiply(
        operand1=output_mag_path,
        operand2=output_mask_path,
        output_path=output_mag_path,
    )
        
    # Output phase
    mrtrix_phase(
        cmpl_path=cmpl_path,
        phase_path=output_pha_path,
    )
    
    # Apply mask
    mrtrix_multiply(
        operand1=output_pha_path,
        operand2=output_mask_path,
        output_path=output_pha_path,
    )
        
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Average stack run time: {elapsed_time}")
        
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)
