import os
import subprocess


def fsl_topup(
        input_img_path,
        input_info_path,
        output_topup_dir,
        scale=None,
    ):
    
    # Initialize output directory
    os.makedirs(output_topup_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "topup",
        f"--imain={input_img_path}",
        f"--datain={input_info_path}",
        f"--out={output_topup_dir}",
    ]
    if scale:
        cmd += [f'--scale={scale}']

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FSL topup failed: {e}")


def fsl_apply_topup(
        input_up_path,
        input_down_path,
        input_info_path,
        input_topup_dir,
        output_topup_img_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_topup_img_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "applytopup",
        f"--imain={input_up_path},{input_down_path}",
        "--inindex=1,2", # This is hardcoded
        f"--datain={input_info_path}",
        f"--topup={input_topup_dir}",
        f"--out={output_topup_img_path}",
    ]
        
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FSL applytopup failed: {e}")


def fsl_apply_warp(
        input_img_path,
        input_target_path,
        input_warp_path,
        output_img_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_img_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "applywarp",
        "-i", input_img_path,
        "-r", input_target_path,
        "-w", input_warp_path,
        "-o", output_img_path,
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FSL applywarp failed: {e}")
    
    
def fsl_fill_holes(
        input_mask_path,
        output_mask_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_mask_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "fslmaths",
        input_mask_path,
        "-fillh26",
        output_mask_path,
    ]
        
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FSL fillholes failed: {e}")
    

def fsl_randomise(
        img_path,
        output_path,
        design_path,
        contrast_path, 
        n_perm,
        mask_path,
        vxf=None,
        vxl=None,
    ):
    
    # Initialize output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Set and run command
    cmd = [
        "randomise",
        "-i", img_path,
        "-o", output_path,
        "-d", design_path,
        "-t", contrast_path,
        "-n", n_perm,
        "T",
        "-m", mask_path,
        "-R",
    ]
    if vxf and vxl:
        cmd += [
            f"--vxl={vxl}",
            f"--vxf={vxf}",
        ]
        
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FSL randomise failed: {e}")
    
    
def fsl_threshold(
        input_path,
        output_path,
        threshold,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "fslmaths",
        input_path,
        "-thr", str(threshold),
        "-bin",
        output_path,
    ]
    
    try:
         subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FSL threshold failed: {e}")