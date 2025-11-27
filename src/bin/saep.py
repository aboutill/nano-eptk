#!/usr/bin/env python

import os
import sys
import argparse
import pathlib
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.eprecon.seap import saep


def parse_args():
    
    # Initialize parser
    parser = argparse.ArgumentParser(
        prog="nano_eptk_saep",
        description=(
            "saep: part of nano eptk package.\n"
            "\n"
            "Reconstruct conductivity and permittivity from complex multi-coil image\n"
            "using the single-acquisition electrical properties (SAEP) method.\n"
            "\n"    
            "Required arguments:\n"
            " - Input multi-coil magnitude.\n"
            " - Input multi-coil phase.\n"
            " - Input mask.\n"
            " - Output conductivity.\n"
            " - Output permittivity.\n"
        ),
        epilog="Arnaud Boutillon (arnaud.boutillon@kcl.ac.uk)",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=6),
    )
    
    # Initialize arguments
    # Required arguments
    # Input magnitude
    parser.add_argument(
        "--mag", 
        type=pathlib.Path,
        help="Input multi-coil magnitude.",
        required=True,
        metavar="\b",
    )
    
    # Input phase
    parser.add_argument(
        "--pha", 
        type=pathlib.Path,
        help="Input multi-coil phase.",
        required=True,
        metavar="\b",
    )
    
    # Input mask
    parser.add_argument(
        "--mask", 
        type=pathlib.Path,
        help="Input mask.",
        required=True,
        metavar="\b",
    )
    
    # Output conductivity
    parser.add_argument(
        "--sig",
        type=pathlib.Path,
        help="Output conductivity.",
        required=True,
        metavar="\b",
    )
    
    # Output permittivity
    parser.add_argument(
        "--eps",
        type=pathlib.Path,
        help="Output permittivity.",
        required=True,
        metavar="\b",
    )
    
    # Optional arguments
    # Output EP metrics
    parser.add_argument(
        "--ep_metric",
        type=pathlib.Path,
        help="Output EP metrics.",
        default=None,
        metavar="\b",
    )
    
    # Output eroded conductivity
    parser.add_argument(
        "--sig_eroded",
        type=pathlib.Path,
        help="Output erdoded conductivity.",
        default=None,
        metavar="\b",
    )
    
    # Output eroded permittivity
    parser.add_argument(
        "--eps_eroded",
        type=pathlib.Path,
        help="Output eroded permittivity.",
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
    
    # Method parameters
    # Coils index
    parser.add_argument(
        "--coils_idx",
        type=int,
        nargs="*",
        help="Coils indexes. Default: all coils.",
        default=None,
        metavar="\b",
    ) 
    
    # Reference coil index
    parser.add_argument(
        "--ref_coil_idx",
        type=int,
        help="Reference coil index. Default: use MLS to create reference.",
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
    
    # SVD modes
    parser.add_argument(
        "--n_svd",
        type=int,
        help="Number of SVD modes to truncate. Default: no SVD truncation.",
        default=None,
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
    # Verbosity
    parser.add_argument(
        "-v", 
        "--verbose",
        action="store_true",
        help="Increase verbosity.",
    ) 
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate files.",
    ) 
    
    # Configuration file
    parser.add_argument(
        "--cfg",
        type=pathlib.Path,
        help="Configuration file.",
        default=None,
        metavar="\b",
    ) 
    
    # Parse arguments
    args = parser.parse_args()

    return args


def execute_saep_with_cfg(
        cfg_path=None,
        **kwargs,
    ):
    
    # Load configuration
    if cfg_path:
        cfg = yaml.safe_load(open(cfg_path))
        kwargs.update(cfg)
   
    # Run SAEP
    saep(**kwargs)
        

def main():
    
    # Parse input arguments
    args = parse_args()

    # Main function call
    execute_saep_with_cfg(
        input_mag_path=args.mag,
        input_pha_path=args.pha,
        input_mask_path=args.mask,
        output_sig_path=args.sig,
        output_eps_path=args.eps,
        output_ep_metrics_path=args.ep_metric,
        output_sig_eroded_path=args.sig_eroded,
        output_eps_eroded_path=args.eps_eroded,
        output_mask_eroded_path=args.mask_eroded,
        coils_idx=args.coils_idx,
        ref_coil_idx=args.ref_coil_idx,
        gs_sigma=args.gs_sigma,
        gs_axes=args.gsr_axes,
        n_svd=args.n_svd,
        f0=args.f0,
        eros_rad=args.eros_rad,
        eros_strc=args.eros_strc,
        verbose=args.verbose,
        debug=args.debug,
        cfg_path=args.cfg,
    )
    

if __name__ == "__main__":
    
    main()
