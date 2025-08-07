import os
import subprocess

args = ["-force", "-quiet"]


def mrtrix_degibbs(
        input_path,
        output_path,
        axes=[0,1,2],
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrdegibbs",
        input_path,
        output_path,
        "-axes", ",".join([str(ax) for ax in axes]),
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix mrdegibbs failed: {e}")
    
    
def mrtrix_polar(
        mag_path,
        pha_path,
        cmpl_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(cmpl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        mag_path,
        pha_path,
        "-polar",
        cmpl_path,
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix polar failed: {e}")
    
        
def mrtrix_real(
        cmpl_path,
        real_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(real_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        cmpl_path,
        "-real",
        real_path,
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix real failed: {e}")
    

def mrtrix_imag(
        cmpl_path,
        imag_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(imag_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        cmpl_path,
        "-imag",
        imag_path,
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix imag failed: {e}")
    
    
def mrtrix_complex(
        real_path,
        imag_path,
        cmpl_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(cmpl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        real_path,
        imag_path,
        "-complex",
        cmpl_path,
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix complex failed: {e}")
    
    
def mrtrix_abs(
        cmpl_path,
        abs_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(abs_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        cmpl_path,
        "-abs",
        abs_path
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix abs failed: {e}")

    
def mrtrix_phase(
        cmpl_path,
        phase_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(phase_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        cmpl_path,
        "-phase",
        phase_path,
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix phase failed: {e}")
    
    
def mrtrix_mean(
        input_path,
        output_path,
        axis=3,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrmath",
        input_path,
        "mean",
        output_path,
        "-axis", str(axis),
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix mean failed: {e}")
    
        
def mrtrix_extract(
        input_path,
        output_path,
        axis=3,
        idx=0,
        axes=[0,1,2],
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
        
    if idx == -1:
        idx = "end" # mrtrix format
        
    cmd = [
        "mrconvert",
        input_path,
        "-coord", str(axis), str(idx),
        "-axes", ",".join([str(ax) for ax in axes]),
        output_path,
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix extract failed: {e}")
    

def mrtrix_cat(
        input_paths,
        output_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = ["mrcat"] + input_paths + [output_path]
    cmd += args
        
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix cat failed: {e}")
    

def mrtrix_multiply(
        operand1,
        operand2,
        output_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        str(operand1),
        str(operand2),
        "-multiply",
        output_path,
    ]
    cmd += args
        
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix multiply failed: {e}")
    
        
def mrtrix_sum(
        input_path,
        output_path,
        axis=3,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrmath",
        input_path,
        "sum",
        output_path,
        "-axis", axis,
    ]
    cmd += args
        
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix sum failed: {e}")
    

def mrtrix_subtract(
        operand1,
        operand2,
        output_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        str(operand1),
        str(operand2),
        "-subtract",
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix subtract failed: {e}")
    

def mrtrix_add(
        operand1,
        operand2,
        output_path,
    ):
    
    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "mrcalc",
        str(operand1),
        str(operand2),
        "-add",
        output_path,
    ]
    cmd += args
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix add failed: {e}")