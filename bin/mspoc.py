import os
import sys
import argparse
import pathlib
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.eprecon.poc import mspoc_pipeline


def parse_args():
    
    # Initialize parser
    parser = argparse.ArgumentParser(
        prog="nano_eptk_mspoc",
        description=(
            "mspoc: part of nano eptk package.\n"
            "\n"
            "Reconstruct conductivity from phase stacks using the multi-stack\n"
            "phase-only conductivity (MS-POC) method.\n"
            "\n"    
            "Required arguments:\n"
            " - Input stacks phase.\n"
            " - Input stacks mask.\n"
            " - Input stacks rigid transformation.\n"
        ),
        epilog="Arnaud Boutillon (arnaud.boutillon@kcl.ac.uk)",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=6),
    )
    
    # Initialize arguments
    # Required arguments
    # Input stacks phase
    parser.add_argument(
        "--phas", 
        type=pathlib.Path,
        nargs="*",
        help="Input stacks phase.",
        required=True,
        metavar="\b",
    )
    
    # Input stacks mask
    parser.add_argument(
        "--masks", 
        type=pathlib.Path,
        nargs="*",
        help="Input stacks mask.",
        required=True,
        metavar="\b",
    )
    
    # Input stacks rigid transformation
    parser.add_argument(
        "--dofs", 
        type=pathlib.Path,
        nargs="*",
        help="Input stacks rigid transformation.",
        required=True,
        metavar="\b",
    )
    
    # Optional arguments
    # Reference image
    parser.add_argument(
        "--ref",
        type=pathlib.Path,
        help="Input reference image.",
        default=None,
        metavar="\b",
    )
    
    # Output conductivity
    parser.add_argument(
        "--sig",
        type=pathlib.Path,
        help="Output conductivity.",
        default=None,
        metavar="\b",
    )
    
    # Output stacks conductivity
    parser.add_argument(
        "--sigs",
        type=pathlib.Path,
        nargs="*",
        help="Output stacks conductivity.",
        default=None,
        metavar="\b",
    )
    
    # Output EP metrics
    parser.add_argument(
        "--ep_metric",
        type=pathlib.Path,
        help="Output EP metrics.",
        default=None,
        metavar="\b",
    )
    
    # Output stacks EP metrics
    parser.add_argument(
        "--ep_metrics",
        type=pathlib.Path,
        help="Output stacks EP metrics.",
        default=None,
        metavar="\b",
    )
    
    # Output eroded conductivity
    parser.add_argument(
        "--sig_eroded",
        type=pathlib.Path,
        help="Output eroded conductivity.",
        default=None,
        metavar="\b",
    )
    
    # Output stacks eroded conductivity
    parser.add_argument(
        "--sigs_eroded",
        type=pathlib.Path,
        nargs="*",
        help="Output stacks eroded conductivity.",
        default=None,
        metavar="\b",
    )
    
    # Output eroded mask
    parser.add_argument(
        "--mask_eroded",
        type=pathlib.Path,
        help="Output eroded mask.",
        default=None,
        metavar="\b",
    )
    
    # Output stacks eroded mask
    parser.add_argument(
        "--masks_eroded",
        type=pathlib.Path,
        nargs="*",
        help="Output stacks eroded mask.",
        default=None,
        metavar="\b",
    )
    
    # Gaussian filter sigma
    parser.add_argument(
        "--gs_sigma",
        type=float,
        nargs="*",
        help="Kernel STD of Gaussian filter in mm. Default: [1.0,1.0,1.0]",
        default=[1.0,1.0,1.0],
        metavar="\b",
    ) 
    
    # Gaussian filter axes
    parser.add_argument(
        "--gs_axes",
        type=int,
        nargs="*",
        help="Axes of Gaussian filter. Default: [0,1,2].",
        default=[0,1,2],
        metavar="\b",
    ) 
    
    # Frequency of RF field
    parser.add_argument(
        "--f0",
        type=float,
        help="Frequency of RF field in Hz. Default: 128Mhz (i.e. 3T field strength).",
        default=128e6,
        metavar="\b",
    ) 
    
    # PDE axes
    parser.add_argument(
        "--pde_axes",
        type=int,
        nargs="*",
        help="Axes of PDE. Default: [0,1,2].",
        default=[0,1,2],
        metavar="\b",
    ) 
    
    # Erosion radius
    parser.add_argument(
        "--eros_rad",
        type=float,
        nargs="*",
        help="Radius of post-procesing erosion in mm. Default: [1.0,1.0,1.0].",
        default=[1.0,1.0,1.0],
        metavar="\b",
    ) 
    
    # Erosion structuring element
    parser.add_argument(
        "--eros_strc",
        type=str,
        help="Structuring element of post-procesing erosion. Default: ball",
        choices=["ball", "cube", "ellipsoid"],
        default="ball",
        metavar="\b",
    ) 
    
    # Misc
    parser.add_argument(
        "-v", 
        "--verbose",
        action="store_true",
        help="Increase verbosity.",
    ) 
    
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="Configuration file.",
        default=None,
        metavar="\b",
    ) 
    
    # Parse arguments
    args = parser.parse_args()

    return args


def execute_mspoc_with_config(
        config_path=None,
        **kwargs,
    ):
    
    #
    if config_path:
        cfg = yaml.safe_load(open(config_path))
        kwargs.update(cfg)
    #
    mspoc_pipeline(**kwargs)
        

def main():
    
    # Parse input arguments
    args = parse_args()

    # Main function call
    execute_mspoc_with_config(
        input_pha_paths=args.phas,
        input_mask_paths=args.masks,
        input_dof_paths=args.dofs,
        input_ref_path=args.ref,
        output_sig_path=args.sig,
        output_sig_path=args.sigs,
        output_ep_metrics_path=args.ep_metric,
        output_ep_metrics_path=args.ep_metrics,
        output_sig_eroded_path=args.sig_eroded,
        output_sig_eroded_paths=args.sigs_eroded,
        output_mask_eroded_path=args.mask_eroded,
        output_mask_eroded_paths=args.masks_eroded,
        gs_sigma=args.gs_sigma,
        gs_axes=args.gs_axes,
        f0=args.f0,
        pde_axes=args.pde_axes,
        eros_rad=args.eros_rad,
        eros_strc=args.eros_strc,
        verbose=args.verbose,
        config=args.config,
    )
    

if __name__ == "__main__":
    
    main()