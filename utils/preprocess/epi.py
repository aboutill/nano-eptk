import os
import json
import shutil
import datetime
import tempfile

import nibabel as nib
import numpy as np

from utils.tools.mrtrix import (
    mrtrix_polar, 
    mrtrix_real,
    mrtrix_imag, 
    mrtrix_complex,
    mrtrix_abs,
    mrtrix_phase,
    mrtrix_extract,
    mrtrix_cat,
)
from utils.tools.fsl import fsl_topup, fsl_apply_topup


def _select_PE_directions(
        input_mag_path,
        input_pha_path,
        output_info_path,
        pe_dirs=[],
        rd_time=1.0,
    ):
    
    # Load magnitude
    mag_nii = nib.load(input_mag_path)
    mag = np.array(mag_nii.get_fdata())
    
    # Load dimension
    dim = mag_nii.header["dim"]
    
    # Load up/down indexes
    up_idx = [
        ii for ii, p in enumerate(pe_dirs) 
        if p == 1
    ]
    down_idx = [
        ii for ii, p in enumerate(pe_dirs) 
        if p == -1
    ]
     
    # Empty lists
    if not up_idx or not down_idx:
        return
    if dim[4] != len(up_idx+down_idx):
        return
    
    # Compute slice-by-slice smoothness in z-direction 
    # for all PE directions
    z_smooth = [
        np.std([np.mean(np.abs(mag[:,:,z,pe] - mag[:,:,z+1,pe])) 
                for z in range(0, dim[3]-1)]) 
        for pe in range(dim[4])
    ]
     
    # Minimum z-smooth in up/down EP direction
    up_min_z_smooth = np.min([z_smooth[ii] for ii in up_idx])
    down_min_z_smooth = np.min([z_smooth[ii] for ii in down_idx])
    
    # Indexes of up/down minimum
    up_idx = z_smooth.index(up_min_z_smooth)
    down_idx = z_smooth.index(down_min_z_smooth)
    
    # Set info
    up_info = {
        "mag": input_mag_path,
        "pha": input_pha_path,
        "pe_idx": up_idx,
        "z_smooth": up_min_z_smooth, 
        "rd_time": rd_time,
    }
    
    down_info = {
        "mag": input_mag_path,
        "pha": input_pha_path,
        "pe_idx": down_idx, 
        "z_smooth": down_min_z_smooth,
        "rd_time": rd_time,
    }
    
    # Initialize output directory
    output_dir = os.path.dirname(output_info_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Write output json file
    info = {
        "up": up_info,
        "down": down_info,
    }
    with open(output_info_path, "w") as f:
        json.dump(info, f, indent=4)
        
        
def topup(
        input_mag_path, 
        input_pha_path,
        output_mag_path, 
        output_pha_path,
        output_info_path=None,
        pe_axis=1,
        verbose=False,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
        
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()
    if output_info_path is None:
        output_info_path =  os.path.join(temp_dir.name, "info.json")
    
    # Temporary file extension
    mag_ext = "mag.nii.gz"
    pha_ext = "pha.nii.gz"
    cmpl_ext = "cmpl.nii.gz"
    real_ext = "real.nii.gz"
    imag_ext = "imag.nii.gz"
    info_ext = "info.txt"
    
    # PE directions selection
    _select_PE_directions(
        input_mag_path,
        input_pha_path,
        output_info_path=output_info_path,
        **kwargs,
    )
    
    # Load PE directions selection
    info = json.load(open(output_info_path))
    up_info = info["up"]
    down_info  = info["down"]
   
    # Empty dict
    if up_info is None and down_info is None:
        # Delete temp dir    
        shutil.rmtree(temp_dir.name)
        return
        
    # Create info.txt for topup
    info_path = os.path.join(temp_dir.name, info_ext)
    info_file = open(info_path, "w")
    if pe_axis == 0:
        info_file.write(f"1 0 0 {up_info['rd_time']}\n") 
        info_file.write(f"-1 0 0 {down_info['rd_time']}\n")
    elif pe_axis == 1:
        info_file.write(f"0 1 0 {up_info['rd_time']}\n") # Specific to dHCP SE EPI data
        info_file.write(f"0 -1 0 {down_info['rd_time']}\n")
    info_file.close()
    
    # Iterate over up/down dict
    for pe_dir, pe_dir_dict in zip(["up", "down"], [up_info, down_info]):
        # Extract dict info
        pe_idx = pe_dir_dict["pe_idx"]
        input_mag_path = pe_dir_dict["mag"]
        input_pha_path = pe_dir_dict["pha"]
        
        # Extract magnitude with best z-smooth
        temp_mag_path = os.path.join(temp_dir.name, f"{pe_dir}_{mag_ext}")
        mrtrix_extract(
            input_path=input_mag_path,
            output_path=temp_mag_path,
            idx=pe_idx,
        )
        
        # Extract phase with best z-smooth
        temp_pha_path = os.path.join(temp_dir.name, f"{pe_dir}_{pha_ext}")
        mrtrix_extract(
            input_path=input_pha_path,
            output_path=temp_pha_path,
            idx=pe_idx,
        )
        
        # To complex
        temp_cmpl_path = os.path.join(temp_dir.name, f"{pe_dir}_{cmpl_ext}")
        mrtrix_polar(
            mag_path=temp_mag_path,
            pha_path=temp_pha_path,
            cmpl_path=temp_cmpl_path,
        )
    
        # Extract real part
        temp_real_path = os.path.join(temp_dir.name, f"{pe_dir}_{real_ext}")
        mrtrix_real(
            cmpl_path=temp_cmpl_path,
            real_path=temp_real_path,
        )
        
        # Extract imag part
        temp_imag_path = os.path.join(temp_dir.name, f"{pe_dir}_{imag_ext}")
        mrtrix_imag(
            cmpl_path=temp_cmpl_path,
            imag_path=temp_imag_path,
        )

        # Append dict info
        pe_dir_dict["temp_mag"] = temp_mag_path
        pe_dir_dict["real"] = temp_real_path
        pe_dir_dict["imag"] = temp_imag_path

    # Merge up/down magnitude
    temp_mag_path = os.path.join(temp_dir.name, mag_ext)
    mrtrix_cat(
        input_paths=[up_info["temp_mag"], down_info["temp_mag"]],
        output_path=temp_mag_path,
    )
        
    # Perform topup
    topup_dir = os.path.join(temp_dir.name, "out")
    fsl_topup(
        input_img_path=temp_mag_path,
        input_info_path=info_path,
        output_topup_dir=topup_dir,
    )
    
    # If previous topup command fails
    if (not os.path.exists(topup_dir + "_movpar.txt") and 
        not os.path.exists(topup_dir + "_fieldcoef.nii.gz")):
        fsl_topup(
            input_img_path=temp_mag_path,
            input_info_path=info_path,
            output_topup_dir=topup_dir,
            scale=1,
        )
    
    # If fail again
    if (not os.path.exists(topup_dir + "_movpar.txt") and 
        not os.path.exists(topup_dir + "_fieldcoef.nii.gz")):
        # Delete temp dir    
        shutil.rmtree(temp_dir.name)
        return
     
    # Apply topup on up/down real parts
    temp_real_path = os.path.join(temp_dir.name, real_ext)
    fsl_apply_topup(
        input_up_path=up_info["real"],
        input_down_path=down_info["real"],
        input_info_path=info_path,
        input_topup_dir=topup_dir,
        output_topup_img_path=temp_real_path,
    )
    
    # Apply topup on up/down imag parts
    temp_imag_path = os.path.join(temp_dir.name, imag_ext)
    fsl_apply_topup(
        input_up_path=up_info["imag"],
        input_down_path=down_info["imag"],
        input_info_path=info_path,
        input_topup_dir=topup_dir,
        output_topup_img_path=temp_imag_path,
    )
    
    if (not os.path.exists(temp_real_path)
        or not os.path.exists(temp_imag_path)):
        # Delete temp dir    
        shutil.rmtree(temp_dir.name)
        return
     
    # To complex
    temp_cmpl_path = os.path.join(temp_dir.name, cmpl_ext)
    mrtrix_complex(
        real_path=temp_real_path,
        imag_path=temp_imag_path,
        cmpl_path=temp_cmpl_path,
    )
     
    # Output corrected magnitude
    mrtrix_abs(
        cmpl_path=temp_cmpl_path,
        abs_path=output_mag_path,
    )
    
    # Output corrected phase
    mrtrix_phase(
        cmpl_path=temp_cmpl_path,
        phase_path=output_pha_path,
    )

    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Topup run time: {elapsed_time}")
        
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)