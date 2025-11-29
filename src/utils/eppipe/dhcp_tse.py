import os 
import json
import yaml
import shutil
import tempfile
import datetime

from utils.preprocess.tse import average_stacks
from utils.preprocess.pha_corr import correct_phase_artefact
from utils.tools.mirtk import mirtk_register, mirtk_transform_image
from utils.eprecon.poc import mspoc
from utils.eprecon.pocr import mspocr


def _dhcp_tse(
        input_mag_paths,
        input_pha_paths,
        input_ref_path,
        input_mask_path,
        output_sig_path,
        input_dhcp_labels9_path=None,
        cfg_path=None,
        verbose=False,
        debug=False,
        eprecon=None,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
    
    # Load configuration
    cfg = yaml.safe_load(open(cfg_path))
    
    # Default values
    for step in ["topup", "pha_corr", eprecon]:
        if step not in cfg:
            cfg[step] = {}
    
    
    # Intermediate directory
    if debug:
        temp_dir = os.path.dirname(output_sig_path)
        temp_dir = os.path.join(temp_dir, "temporary-files")
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        
    # Set file path
    n = len(input_mag_paths)
    stack_dhcp_labels9_paths = [os.path.join(temp_dir, f"stack{i}_dhcp_labels9.nii.gz") for i in range(n)]
    stack_mask_paths = [os.path.join(temp_dir, f"stack{i}_mask.nii.gz") for i in range(n)]
    stack_dof_paths = [os.path.join(temp_dir, f"stack{i}_vol.dof") for i in range(n)]
    
    # Iter over stacks
    for i in range(n):
        # Register stack to volume
        mirtk_register(
            input_img1_path=input_ref_path,
            input_img2_path=input_mag_paths[i],
            output_dof_path=stack_dof_paths[i],
        )
        
        # Apply transform to mask and labels
        mirtk_transform_image(
            input_path=input_mask_path,
            input_target_path=input_mag_paths[i],
            input_invdof_path=stack_dof_paths[i],
            output_path=stack_mask_paths[i],
            label=True,
        )
        if input_dhcp_labels9_path:
            mirtk_transform_image(
                input_path=input_dhcp_labels9_path,
                input_target_path=input_mag_paths[i],
                input_invdof_path=stack_dof_paths[i],
                output_path=stack_dhcp_labels9_paths[i],
                label=True,
            )
        else:
            stack_dhcp_labels9_paths[i] = None
        
    # Set path
    avg_mag_path = os.path.join(temp_dir, "avg_mag.nii.gz")
    avg_pha_path = os.path.join(temp_dir, "avg_pha.nii.gz")
    avg_mask_path = os.path.join(temp_dir, "avg_mask.nii.gz")
    avg_info_path = os.path.join(temp_dir, "avg_info.json")
    
    # Average stacks
    average_stacks(
        input_mag_paths=input_mag_paths,
        input_pha_paths=input_pha_paths,
        input_mask_paths=stack_mask_paths,
        input_dof_paths=stack_dof_paths,
        output_mag_path=avg_mag_path,
        output_pha_path=avg_pha_path,
        output_mask_path=avg_mask_path,
        output_info_path=avg_info_path,
    )
    
    # Load stack selection information
    info = json.load(open(avg_info_path))
    
    # Indexes of selected stacks
    index = [input_mag_paths.index(info[ornt]['mag']) for ornt in ["axial", "sagittal"]]
    
    # Set file path
    stack_pha_paths = [input_pha_paths[i] for i in index]
    stack_mask_paths = [stack_mask_paths[i] for i in index]
    stack_dhcp_labels9_paths = [stack_dhcp_labels9_paths[i] for i in index]
    stack_dof_paths = [stack_dof_paths[i] for i in index]
    stack_pha_corr_paths = [os.path.join(temp_dir, f"stack{i}_pha_corr.nii.gz") for i in index]
    stack_mask_artefact_paths = [os.path.join(temp_dir, f"stack{i}_mask_artefact.nii.gz") for i in index]
    
    # Iter over stacks
    for i in index:
        # Apply phase artefact correction
        if cfg["pha_corr"]:
            correct_phase_artefact(
                input_pha_path=stack_pha_paths[i], 
                input_mask_path=stack_mask_paths[i], 
                output_pha_path=stack_pha_corr_paths[i],
                output_mask_artefact_path=stack_mask_artefact_paths[i],
                **cfg["pha_corr"],
                **kwargs,
            )
        else:
            shutil.copyfile(stack_pha_paths[i], stack_pha_corr_paths[i])
        
    # Run MSPOC/MSPOCR
    if eprecon == "mspoc":
        mspoc(
            input_pha_paths=stack_pha_corr_paths,
            input_mask_paths=stack_mask_paths,
            input_dhcp_labels9_paths=stack_dhcp_labels9_paths,
            input_dof_paths=stack_dof_paths,
            input_ref_path=avg_mag_path,
            output_sig_path=output_sig_path,
            verbose=verbose,
            debug=debug,
            **cfg["mspoc"], 
            **kwargs,
        )
    elif eprecon == "mspocr":
        # Run MSPOCR
        mspocr(
            input_pha_paths=stack_pha_corr_paths,
            input_mask_paths=stack_mask_paths,
            input_dhcp_labels9_paths=stack_dhcp_labels9_paths,
            input_dof_paths=stack_dof_paths,
            input_ref_path=avg_mag_path,
            output_sig_path=output_sig_path,
            verbose=verbose,
            debug=debug,
            **cfg["mspocr"],
            **kwargs,
        )
                
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"TSE {eprecon.upper()} run time: {elapsed_time}")
    
    # Delete temp dir 
    if not debug:
        shutil.rmtree(temp_dir)
        

def dhcp_tse_mspoc(**kwargs):
    
    _dhcp_tse(eprecon="mspoc", **kwargs)
        
    
def dhcp_tse_mspocr(**kwargs):
    
    _dhcp_tse(eprecon="mspocr", **kwargs)
        