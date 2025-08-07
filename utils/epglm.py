import os
import json
import datetime
import tempfile
import shutil

import numpy as np

from utils.tools.fsl import fsl_randomise, fsl_threshold
from utils.tools.mrtrix import (
    mrtrix_cat,
    mrtrix_mean,
    mrtrix_subtract,
    mrtrix_add,
    mrtrix_multiply,
)

    
def ep_glm(
        input_ep_paths,
        input_mask_path,
        output_dir,
        input_ev_path=None,
        input_vox_ev_path=None,
        n_perm=10000,
        alpha=0.05,
        verbose=False,
    ):
    
    if len(input_ep_paths) < 2:
        return
    
    # Initialize timer
    if verbose:
       start_time = datetime.datetime.now()
        
    # Utils for command
    arg_str = lambda l: ' '.join(str(x) for x in l)
    fsl_arg_str = lambda l: ','.join(str(x) for x in l)
    
    #
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory()
    
    #
    n = len(input_ep_paths)
    
    # Concatenate EP
    ep_4D_path = os.path.join(temp_dir.name, "ep_4D.nii.gz")
    mrtrix_cat(
        input_paths=input_ep_paths,
        output_path=ep_4D_path,
    )

    # Demean image
    ep_mean_path = os.path.join(temp_dir.name, "ep_mean.nii.gz")
    mrtrix_mean(
        input_path=ep_4D_path,
        output_path=ep_mean_path,
    )
    mrtrix_subtract(
        operand1=ep_4D_path,
        operand2=ep_mean_path,
        output_path=ep_4D_path,
    )
    
    #
    ev = json.load(open(input_ev_path))
    ev_labels = list(ev)
    
    #
    vox_ev = json.load(open(input_ev_path))
    vox_ev_labels = list(vox_ev)
            
    # Demean EVs
    for ev_label in ev_labels:
        ev[ev_label] -= np.mean(ev[ev_label])

    #    
    vox_ev_4D_paths = []
    for vox_ev_label in vox_ev_labels:
        
        # Concatenate EV
        vox_ev_4D_path = os.path.join(temp_dir.name, f"{ev}_4D.nii.gz")
        mrtrix_cat(
            input_paths=vox_ev[vox_ev_label],
            output_path=vox_ev_4D_path,
        )
        
        # Demean image
        vox_ev_mean_path = os.path.join(temp_dir.name, f"{ev}_mean.nii.gz")
        mrtrix_mean(
            input_path=vox_ev_4D_path,
            output_path=vox_ev_mean_path,
        )
        mrtrix_subtract(
            operand1=vox_ev_4D_path,
            operand2=vox_ev_mean_path,
            output_path=vox_ev_4D_path,
        )
        
        #
        vox_ev_4D_paths += [vox_ev_4D_path]
    
    # Initialize design matrix
    design_path = os.path.join(temp_dir.name, "design.txt")
    design_file = open(design_path, "w")

    for i in range(n):
        
        design = []
        for ev_label in ev_labels:
            design += [ev[ev_label][i]]
            
        for vox_ev_label in vox_ev_labels:
            design += [1]
        
        # Update design matrix
        design_file.write(f"{arg_str(design)}\n")
        
    # Close design file
    design_file.close()

    # Design matrix to .mat file
    design_path_mat = os.path.join(temp_dir.name, "design.mat")
    os.system(f"Text2Vest {design_path} {design_path_mat}")
    
    # Apply mask on EP
    mrtrix_multiply(
        operand1=ep_4D_path,
        operand2=input_mask_path,
        output_path=ep_4D_path,
    )
    
    # Aplly mask on voxel EVs
    for vox_ev_4D_path in vox_ev_4D_paths:
        
        mrtrix_multiply(
            operand1=vox_ev_4D_path,
            operand2=input_mask_path,
            output_path=vox_ev_4D_path,
        )
        
    # Contrast
    all_ev_labels = ev_labels + vox_ev_labels
    if len(vox_ev_labels) > 0:
        
        vxl = fsl_arg_str([i for i in range(len(ev_labels)+1, len(all_ev_labels)+1)])
        vxf = fsl_arg_str(vox_ev_4D_paths)
        
    for i, ev in enumerate(all_ev_labels):
        
        # Create contrast
        contrast_path = os.path.join(temp_dir.name, f"contrast_{ev}.txt")
        contrast_file = open(contrast_path, "w")
        contrast = [0] * len(all_ev_labels)
        contrast[i] = 1 # Positive correlation
        contrast_file.write(f"{arg_str(contrast)}\n")
        contrast[i] = -1 # Negative correlation
        contrast_file.write(f"{arg_str(contrast)}\n")
        
        # Close contrast file
        contrast_file.close()
        
        # Contrast matrix to .con file
        contrast_path_con = os.path.join(temp_dir.name, f"contrast_{ev}.con")
        os.system(f"Text2Vest {contrast_path} {contrast_path_con}")
        
        #
        output_dir_ev = os.path.join(output_dir, ev)
        os.makedirs(output_dir_ev, exist_ok=True)
            
        #
        output_path_ev = os.path.join(output_dir_ev, ev)
        
        fsl_randomise(
            imgs_path=ep_4D_path,
            output_path=output_path_ev,
            design_path=design_path_mat,
            contrast_path=contrast_path_con, 
            n_perm=n_perm,
            mask_path=input_mask_path,
            vxf=vxf,
            vxl=vxl,
        )
        
        #
        n = 2
        tstat_paths = [os.path.join(output_dir, f"{ev}_tstat{i+1}.nii.gz") for i in range(n)]
        tfce_corrp_tstat_paths = [os.path.join(output_dir, f"{ev}_tfce_corrp_tstat{i+1}.nii.gz") for i in range(n)]
        tfce_corrp_tstat_tresh95_paths = [os.path.join(output_dir, f"{ev}_tfce_corrp_tstat{i+1}_thresh95.nii.gz") for i in range(n)]
        tfce_corrp_tstat_mask_path = os.path.join(output_dir, f"{ev}_tfce_corrp_tstat_mask.nii.gz")
        tstat_thresh_tfce_corrp_path = os.path.join(output_dir, f"{ev}_tstat_thresh_tfce_corrp.nii.gz")
        
        # Trehsold at significance level
        for i in range(n):
            fsl_threshold(
                input_path=tfce_corrp_tstat_paths[i],
                output_path=tfce_corrp_tstat_tresh95_paths[i],
                threshold=(1-alpha/n),
            )
    
        # Add mask
        mrtrix_add(
            operand1=tfce_corrp_tstat_tresh95_paths[0],
            operand2=tfce_corrp_tstat_tresh95_paths[1],
            output_path=tfce_corrp_tstat_mask_path
        )
    
        # Aplly mask
        mrtrix_multiply(
            operand1=tstat_paths[0],
            operand2=tfce_corrp_tstat_mask_path,
            output_path=tstat_thresh_tfce_corrp_path,
        ) 
            
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"EP voxel-wise GLM run time: {elapsed_time}")
        
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)