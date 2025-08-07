import os
import datetime
import tempfile
import shutil

import numpy as np

from utils.tools.mrtrix import (
    mrtrix_multiply,
    mrtrix_cat,
    mrtrix_sum,
)
from utils.tools.fsl import (
    fsl_apply_warp,
)


def construct_ep_atlas(
        input_ep_paths,
        input_ages,
        input_anat_atlas_paths,
        input_intra_atlas_warp_paths,
        input_atlas_ages,
        output_ep_atlas_paths,
        age_sigma=1, 
        verbose=False,
    ):
    
    if len(input_ep_paths) < 2:
        return
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
    
    # Set length
    n = len(input_ep_paths)
    m = len(input_anat_atlas_paths)
    
    # Temporary dir
    temp_dir = tempfile.TemporaryDirectory() 
        
    # Initialize arrays
    ep_paths = [[] for j in range(m)]
    weights = [[] for j in range(m)]
    
    # Iter over EP images
    for i in range(n):
        
        if verbose:
            print(f"EP {i+1}/{n} to atlas space")
        
        #
        ep_path = input_ep_paths[i]
        
        for j in range(m):
            
            # 
            age_delta = input_atlas_ages[j] - input_ages[i]
            
            # Skip 
            if abs(age_delta) > 3 * age_sigma:
                continue
            
            #
            if verbose:
                print(f"EP {i+1}/{n} to atlas space {j+1}/{m}")
            
            #
            anat_atlas_path = input_anat_atlas_paths[j]
            warp_path = input_intra_atlas_warp_paths[j]
            temp_ep_age_path = os.path.join(temp_dir.name, f"ep_{i}_{j}.nii.gz")
            
            #
            if  warp_path is not None:
            
                # FSL apply warp to atlas
                fsl_apply_warp(
                    input_img_path=ep_path,
                    input_target_path=anat_atlas_path,
                    input_warp_path=warp_path,
                    output_img_path=temp_ep_age_path,
                )
                
            else:
                # Copy files
                os.system(f"cp {ep_path} {temp_ep_age_path}")
                
            # Gaussian temporal weights
            weight = np.exp(- age_delta**2 / (2 * age_sigma**2)) # exp(- delta**2 / (2*sigma**2))
            weight /= (age_sigma * np.sqrt(2 * np.pi)) # 1 / (sigma * sqrt(2 * pi))
                    
            # Apply weight on EP image
            mrtrix_multiply(
                operand1=temp_ep_age_path,
                operand2=weight,
                output_path=temp_ep_age_path,
            )
           
            # Update arrays
            ep_paths[j] += [temp_ep_age_path]
            weights[j] += [weight]
            
    # Build atlas
    for j in range(m):
        
        # Update number of EPs
        n_ep = len(ep_paths[j])
        
        if verbose:
            print(f"Building atlas {j+1}/{m} ({n_ep} EPs)")
            
        if n_ep < 2:
            continue
        
        #
        ep_atlas_path = output_ep_atlas_paths[j]
        
        # Concatenate all EPs weighted
        mrtrix_cat(
            input_paths=ep_paths[j],
            output_path=ep_atlas_path,
        )
        
        # Compute weighted sum of EPs
        mrtrix_sum(
            input_path=ep_atlas_path,
            output_path=ep_atlas_path,
        )
        
        # Compute weighted mean of EPs
        weights_sum = sum(weights[j]) # sum(w_i)
        mrtrix_multiply(
            operand1=ep_atlas_path,
            operand2=1/weights_sum,
            output_path=ep_atlas_path,
        )
    
    #
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"EP atlas construction run time: {elapsed_time}")
        
    # Delete temp dir    
    shutil.rmtree(temp_dir.name)