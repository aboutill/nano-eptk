import os
import re

from utils.tools.fsl import fsl_threshold
from utils.tools.mrtrix import mrtrix_multiply, mrtrix_add


def mirtk_average_images(
        input_paths,
        input_dof_paths,
        output_path,
        input_ref_path=None,
        label=False,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "average-images",
        output_path,
        " ".join([f"-image {img} -dof_i {dof}" for img, dof 
                  in zip(input_paths, input_dof_paths)]),
    ]
    if input_ref_path:
        cmd += ["-reference", input_ref_path]
    if label:
        cmd += ["-label"]
    cmd = " ".join(cmd)
    
    os.system(cmd)
        
    # Merge labels
    if label:
        # Get filename
        base = os.path.basename(output_path).split(".nii.gz")[0]
        files = [file for file in os.listdir(output_dir)
                 if re.match(rf"{re.escape(base)}_\d+", file)]
        n = len(files)
        label_1_path = output_path.replace(".nii.gz", "_1.nii.gz")
        
        # Iter over labels
        for i in range(n):
            # Threshold
            label_i_path = output_path.replace(".nii.gz", f"_{i+1}.nii.gz")
            fsl_threshold(
                input_path=label_i_path,
                output_path=label_i_path,
                threshold=0.5,
            )
            
            if i > 0:
                # Add labels
                mrtrix_multiply(
                    operand1=label_i_path,
                    operand2=i+1,
                    output_path=label_i_path,
                )
                mrtrix_add(
                    operand1=label_i_path,
                    operand2=label_1_path,
                    output_path=label_1_path,
                )
                
                os.remove(label_i_path)
        os.rename(label_1_path, output_path)
    
        
def mirtk_transform_image(
        input_path,
        output_path,
        input_target_path=None,
        input_dof_path=None,
        input_invdof_path=None,
        label=False,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "transform-image",
        input_path,
        output_path,
    ]
    
    if input_target_path:
        cmd += ["-target", input_target_path]
    if input_dof_path:
        cmd += ["-dofin", input_dof_path]
    if input_invdof_path:
        cmd += ["-invdof", input_invdof_path]
    if label:
        cmd += ["-labels"]
    cmd = " ".join(cmd)
    
    os.system(cmd)
    

def mirtk_register(
        input_img1_path,
        input_img2_path,
        output_dof_path,
        model="Rigid",
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_dof_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "register",
        "-image", input_img1_path,
        "-image", input_img2_path,
        "-dofout", output_dof_path,
        "-model", model,
    ]
    cmd += ["-v 0"]
    cmd = " ".join(cmd)
    
    os.system(cmd)
    

def mirtk_resample_image(
        input_path,
        output_path,
        size=[1.0, 1.0, 1.0],
        interp="Linear with padding",
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set and run command
    cmd = [
        "resample-image",
        input_path,
        output_path,
        "-size", " ".join([str(s) for s in size]),
        "-interp", f"'{interp}'",
    ]
    cmd = " ".join(cmd)
    
    os.system(cmd)