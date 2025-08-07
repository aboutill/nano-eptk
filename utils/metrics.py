import os
import json

import nibabel as nib
import numpy as np

dhcp_labels9_str = [
    "CSF",
    "cGM",
    "WM",
    "intracranial-background",
    "ventricle",
    "cerebellum",
    "dGM",
    "brainstem",
    "hippocampus",
]

brain_tissue_ids = [
    dhcp_labels9_str.index("cGM"),
    dhcp_labels9_str.index("WM"),
    dhcp_labels9_str.index("cerebellum"),
    dhcp_labels9_str.index("dGM"),
    dhcp_labels9_str.index("brainstem"),
    dhcp_labels9_str.index("hippocampus"),
]

tissue_ids = [
    dhcp_labels9_str.index("WM"),
    dhcp_labels9_str.index("dGM"),
]


def extract_ep_metrics(
        output_ep_metrics_path,
        input_sig_path=None,
        input_eps_path=None,
        input_mask_eroded_path=None,
        input_mask_outlier_path=None,
        input_dhcp_labels9_path=None,
        **kwargs
    ):

    # Initalize meta
    ep_metrics = {}
    
    # Load niftis
    if input_sig_path is not None:
        sig_nii = nib.load(input_sig_path)
        sig = sig_nii.get_fdata()
        
    if input_eps_path is not None:
        eps_nii = nib.load(input_eps_path)
        eps = eps_nii.get_fdata()
        
    if input_mask_eroded_path is not None:
        mask_eroded_nii = nib.load(input_mask_eroded_path)
        mask_eroded = mask_eroded_nii.get_fdata().astype(bool)
        
    if input_dhcp_labels9_path is not None:
        dhcp_labels9_nii = nib.load(input_dhcp_labels9_path)
        dhcp_labels9 = dhcp_labels9_nii.get_fdata()
        
    # Save results (computed in eroded mask)
    if input_mask_eroded_path is not None:
        if input_sig_path is not None:
            ep_metrics["SIG"] = np.median(sig[mask_eroded])
            ep_metrics["SIG_std"] = np.std(sig[mask_eroded])
            
            if input_dhcp_labels9_path is not None:
                # Apply eroded mask
                dhcp_labels9 *= mask_eroded
                
                # WN and dGM
                for tissue_id in tissue_ids:
                    tissue_str = dhcp_labels9_str[tissue_id]
                    tissue_sig = sig[dhcp_labels9 == tissue_id + 1]
                    ep_metrics[f"{tissue_str}_SIG"] = np.median(tissue_sig)
                    ep_metrics[f"{tissue_str}_SIG_std"] = np.std(tissue_sig)
                
                # All brain tissues (excluding CSF and ventricle))
                tissues_sig = np.concatenate(
                    [sig[dhcp_labels9 == tissue_id + 1]
                     for tissue_id in brain_tissue_ids], 
                    axis=-1,
                )
                ep_metrics["brain_SIG"] = np.median(tissues_sig)
                ep_metrics["brain_SIG_std"] = np.std(tissues_sig)
    
        if input_eps_path is not None:
            ep_metrics["EPS"] = np.median(eps[mask_eroded])
            ep_metrics["EPS_std"] = np.std(eps[mask_eroded])
            
            if input_dhcp_labels9_path is not None:
                # Apply eroded mask
                dhcp_labels9 *= mask_eroded
                
                # WN and dGM
                for tissue_id in tissue_ids:
                    tissue_str = dhcp_labels9_str[tissue_id]
                    tissue_eps = eps[dhcp_labels9 == tissue_id + 1]
                    ep_metrics[f"{tissue_str}_EPS"] = np.median(tissue_eps)
                    ep_metrics[f"{tissue_str}_EPS_std"] = np.std(tissue_eps)
                
                # All brain tissues (excluding CSF and ventricle))
                tissues_eps = np.concatenate(
                    [eps[dhcp_labels9 == tissue_id + 1]
                     for tissue_id in brain_tissue_ids], 
                    axis=-1,
                )
                ep_metrics["brain_EPS"] = np.median(tissues_eps)
                ep_metrics["brain_EPS_std"] = np.std(tissues_eps)
                
    # Round results
    for key, value in ep_metrics.items():
        ep_metrics[key] = round(ep_metrics[key], 4)
                
    # Initialize output directory
    output_dir = os.path.dirname(output_ep_metrics_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Save metadata
    with open(output_ep_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(ep_metrics, f, ensure_ascii=False, indent=4)
        
        
def extract_dhcp_volume_metrics(
        input_dhcp_labels9_path,
        output_volume_metrics_path,
    ):
    
    # Initialize metadata
    volume_metrics = {}
        
    # Load nifti
    dhcp_labels9_nii = nib.load(input_dhcp_labels9_path)
    dhcp_labels9 = dhcp_labels9_nii.get_fdata()
    
    # Remove intracranial background
    dhcp_labels9[dhcp_labels9 == dhcp_labels9_str.index("intracranial-background") + 1] = 0
    
    # Include CSF and ventricles
    brain_vol = np.sum(dhcp_labels9 >= 1)
    
    # Compute relative volume for all tissue
    ventricle_id = dhcp_labels9_str.index("ventricle")
    tissue_vol = np.sum(dhcp_labels9 == ventricle_id+1)
    volume_metrics["RVV"] = tissue_vol / brain_vol * 1e2
    
    # Round results
    for key, value in volume_metrics.items():
        volume_metrics[key] = round(volume_metrics[key], 4)
    
    # Initialize output directory
    output_dir = os.path.dirname(output_volume_metrics_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Save metadata
    with open(output_volume_metrics_path, "w", encoding="utf-8") as f:
        json.dump(volume_metrics, f, ensure_ascii=False, indent=4)
    

def extract_dhcp_mean_diffusivity_metrics(
        input_mean_diffusivity_path,
        input_dhcp_labels9_path,
        output_md_metrics_path,
    ):
    
    # Initialize metadata
    md_metrics = {}
        
    # Load nifti
    md_nii = nib.load(input_mean_diffusivity_path)
    dhcp_labels9_nii = nib.load(input_dhcp_labels9_path)
    
    #
    md = md_nii.get_fdata() * 1e3
    dhcp_labels9 = dhcp_labels9_nii.get_fdata()
    
    # WN and dGM
    for tissue_id in tissue_ids:
        tissue_str = dhcp_labels9_str[tissue_id]
        tissue_md = md[dhcp_labels9 == tissue_id + 1]
        md_metrics[f"{tissue_str}_MD"] = np.median(tissue_md)
    
    # All brain tissues (excluding CSF and ventricle)
    tissues_md = np.concatenate(
        [md[dhcp_labels9 == tissue_id + 1]
         for tissue_id in brain_tissue_ids], 
        axis=-1,
    )
    md_metrics["brain_MD"] = np.median(tissues_md)
    
    # Round results
    for key, value in md_metrics.items():
        md_metrics[key] = round(md_metrics[key], 4)
    
    # Initialize output directory
    output_dir = os.path.dirname(output_md_metrics_path)
    os.makedirs(output_dir, exist_ok=True)
        
    # Save metadata
    with open(output_md_metrics_path, "w", encoding="utf-8") as f:
        json.dump(md_metrics, f, ensure_ascii=False, indent=4)
        

def extract_outlier_metrics(
        input_mask_path,
        input_mask_outlier_path,
        output_outlier_metrics_path,
    ):
    
    # Initalize meta
    outlier_metrics = {}
    
    # Load data
    mask_nii = nib.load(input_mask_path)
    mask = mask_nii.get_fdata().astype(bool)
    
    mask_outlier_nii = nib.load(input_mask_outlier_path)
    mask_outlier = mask_outlier_nii.get_fdata().astype(bool)
    
    # Save mask outlier volume
    outlier_metrics["REL_VOL"] = int(np.sum(mask_outlier)) / int(np.sum(mask))
            
    # Initialize output directory
    output_dir = os.path.dirname(output_outlier_metrics_path)
    os.makedirs(output_dir, exist_ok=True)
        
    with open(output_outlier_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(outlier_metrics, f, ensure_ascii=False, indent=4)