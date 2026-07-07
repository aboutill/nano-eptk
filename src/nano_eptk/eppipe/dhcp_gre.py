import os
import yaml
import shutil
import tempfile
import datetime

from nano_eptk.eprecon.saep import saep
from nano_eptk.preprocess.gre import degibbs


def dhcp_gre_saep(
        input_mag_path,
        input_pha_path,
        input_mask_path,
        output_sig_path,
        output_eps_path,
        cfg_path=None,
        verbose=False,
        debug=False,
        **kwargs,
    ):
    
    # Initialize timer
    if verbose:
        start_time = datetime.datetime.now()
    
    # Load configuration
    if cfg_path is not None:
        cfg = yaml.safe_load(open(cfg_path))
    else:
        cfg = {}
    
    # Default values
    for step in ["degibbs", "saep"]:
        if step not in cfg:
            cfg[step] = {}
    
    # Intermediate directory
    if debug:
        temp_dir = os.path.commonpath([output_sig_path, output_eps_path])
        temp_dir = os.path.join(temp_dir, "temporary-files")
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        
    # Set paths
    mag_degibbs_path = os.path.join(temp_dir, "mag_degibbs.nii.gz")
    pha_degibbs_path = os.path.join(temp_dir, "pha_degibbs.nii.gz")
    
    # Run DeGibbs correction
    if cfg["degibbs"]:
        degibbs(
            input_mag_path=input_mag_path, 
            input_pha_path=input_pha_path,
            output_mag_path=mag_degibbs_path, 
            output_pha_path=pha_degibbs_path,
            verbose=verbose,
            **cfg["degibbs"],
            **kwargs,
        )
    else:
        shutil.copyfile(input_mag_path, mag_degibbs_path)
        shutil.copyfile(input_pha_path, pha_degibbs_path)
    
    # Run SAEP
    saep(
        input_mag_path=mag_degibbs_path,
        input_pha_path=pha_degibbs_path,
        input_mask_path=input_mask_path,
        output_sig_path=output_sig_path,
        output_eps_path=output_eps_path,
        verbose=verbose,
        debug=debug,
        **cfg["saep"],
        **kwargs,
    )
    
    # Print timer
    if verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print(f"GRE SAEP run time: {elapsed_time}")
    
    # Delete temp dir 
    if not debug:
        shutil.rmtree(temp_dir)