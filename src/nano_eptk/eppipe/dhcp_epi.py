import os 
import yaml
import shutil
import tempfile
import datetime

from nano_eptk.preprocess.epi import topup
from nano_eptk.preprocess.pha_corr import correct_phase_artefact
from nano_eptk.eprecon.poc import poc
from nano_eptk.eprecon.pocr import pocr


def _dhcp_epi(
        input_mag_path,
        input_pha_path,
        input_mask_path,
        output_sig_path,
        cfg_path=None,
        verbose=False,
        debug=False,
        eprecon=None,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
        
    # Chek method
    if not (eprecon == "poc" or eprecon == "pocr"):
        return
    
    # Load configuration
    if cfg_path is not None:
        cfg = yaml.safe_load(open(cfg_path))
    else:
        cfg = {}
    
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
        
    # Set paths
    mag_topup_path = os.path.join(temp_dir, "mag_topup.nii.gz")
    pha_topup_path = os.path.join(temp_dir, "pha_topup.nii.gz")
    topup_info_path = os.path.join(temp_dir, "info_topup.nii.gz")
    
    
    # Run TOPUP
    if cfg["topup"]:
        topup(
            input_mag_path=input_mag_path, 
            input_pha_path=input_pha_path,
            output_mag_path=mag_topup_path, 
            output_pha_path=pha_topup_path,
            output_info_path=topup_info_path,
            **cfg["topup"],
            **kwargs,
        )
    else:
        shutil.copyfile(input_mag_path, mag_topup_path)
        shutil.copyfile(input_pha_path, pha_topup_path)
    
    # Set paths
    pha_corr_path = os.path.join(temp_dir, "pha_corr.nii.gz")
    mask_artefact_path = os.path.join(temp_dir, "mask_artefact.nii.gz")
    
    # Apply phase artefact correction
    if cfg["pha_corr"]:
        correct_phase_artefact(
            input_pha_path=pha_topup_path, 
            input_mask_path=input_mask_path,
            output_pha_path=pha_corr_path,
            output_mask_artefact_path=mask_artefact_path,
            **cfg["pha_corr"],
            **kwargs,
        )
    else:
        shutil.copyfile(pha_topup_path, pha_corr_path)
        
    # Run POC/POCR
    if eprecon == "poc":
        poc(
            input_pha_path=pha_corr_path,
            input_mask_path=input_mask_path,
            output_sig_path=output_sig_path,
            verbose=verbose,
            debug=debug,
            **cfg["poc"]
            **kwargs,
        )
    elif eprecon == "pocr":
        pocr(
            input_pha_path=pha_corr_path,
            input_mask_path=input_mask_path,
            output_sig_path=output_sig_path,
            verbose=verbose,
            debug=debug,
            **cfg["pocr"],
            **kwargs,
        )
    
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"EPI {eprecon.upper()} run time: {elapsed_time}")
    
    # Delete temp dir 
    if not debug:
        shutil.rmtree(temp_dir)
        

def dhcp_epi_poc(**kwargs):
    
    _dhcp_epi(eprecon="poc", **kwargs)
        
    
def dhcp_epi_pocr(**kwargs):
    
    _dhcp_epi(eprecon="pocr", **kwargs)
        