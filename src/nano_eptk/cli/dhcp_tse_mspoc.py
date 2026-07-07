#!/usr/bin/env python

import argparse
import pathlib
import json

from nano_eptk.eppipe.dhcp_tse import dhcp_tse_mspoc


def parse_args():
    
    # Initialize parser
    parser = argparse.ArgumentParser(
        prog="dhcp_tse_mspoc",
        description=(
            "dhcp_tse_mspoc: part of nano eptk package.\n"
            "\n"
            "Apply the dHCP TSE-MSPOC pipeline on input data.\n"
            "\n"    
            "Required arguments:\n"
            " - Input stacks magnitude.\n"
            " - Input stacks phase.\n"
            " - Input reference volume.\n"
            " - Input mask.\n"
            " - Output conductivity.\n"
            " OR\n"
            " - Input datalist.\n"
        ),
        epilog="Arnaud Boutillon (arnaud.boutillon@kcl.ac.uk)",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=6),
    )
    
    # Initialize arguments
    # Required arguments
    # Input stacks magnitude
    parser.add_argument(
        "--mags", 
        type=pathlib.Path,
        nargs="*",
        help="Input stacks magnitude.",
        default=None,
        metavar="\b",
    )
    
    # Input stacks phase
    parser.add_argument(
        "--phas", 
        type=pathlib.Path,
        nargs="*",
        help="Input stacks phase.",
        default=None,
        metavar="\b",
    )
    
    # Input reference volume
    parser.add_argument(
        "--ref",
        type=pathlib.Path,
        help="Input reference volume.",
        default=None,
        metavar="\b",
    )
    
    # Input stacks mask
    parser.add_argument(
        "--mask", 
        type=pathlib.Path,
        help="Input reference mask.",
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
    
    # OR
    # Input datalist
    parser.add_argument(
        "--datalist",
        type=pathlib.Path,
        help="Input datalist.",
        default=None,
        metavar="\b",
    )

    # Optional arguments
    # Input dhcp labels9
    parser.add_argument(
        "--dhcp_labels9",
        type=pathlib.Path,
        help="Input dHCP labels9.",
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


def execute_dhcp_tse_mspoc_with_datalist(
        input_mag_paths=None,
        input_pha_paths=None,
        input_ref_path=None,
        input_mask_path=None,
        output_sig_path=None,
        datalist_path=None,
        input_dhcp_labels9_path=None,
        output_ep_metrics_path=None,
        **kwargs,
    ):
    
    # Required datalist arguments
    keys = [
        "input_mag_paths",
        "input_pha_paths",
        "input_ref_path",
        "input_mask_path",
        "output_sig_path",
    ]
    
    # Default
    dflt_kwargs = kwargs.copy()
    
    if datalist_path:
        # Load datalist
        datalist = json.load(open(datalist_path))
        
        # Iter over datalist
        for data in datalist:
            
            # Check arguments
            if not all([key in data for key in keys]):
                continue
            
            # Update arguments
            kwargs = dflt_kwargs.copy()
            kwargs.update(data)
   
            # Run dHCP TSE MSPOC
            dhcp_tse_mspoc(**kwargs)
        
    elif (input_mag_paths and 
          input_pha_paths and
          input_ref_path and
          input_mask_path and
          output_sig_path):
        
        # Run dHCP TSE MSPOC
        dhcp_tse_mspoc(
            input_mag_paths=input_mag_paths,
            input_pha_paths=input_pha_paths,
            input_ref_path=input_ref_path,
            input_mask_path=input_mask_path,
            output_sig_path=output_sig_path,
            input_dhcp_labels9_path=input_dhcp_labels9_path,
            output_ep_metrics_path=output_ep_metrics_path,
            **kwargs,
        )
        

def main():
    
    # Parse input arguments
    args = parse_args()

    # Main function call
    execute_dhcp_tse_mspoc_with_datalist(
        input_mag_paths=args.mags,
        input_pha_paths=args.phas,
        input_ref_path=args.ref,
        input_mask_path=args.mask,
        output_sig_path=args.sig,
        datalist_path=args.datalist,
        input_dhcp_labels9_path=args.dhcp_labels9,
        output_ep_metrics_path=args.ep_metric,
        verbose=args.verbose,
        debug=args.debug,
        cfg_path=args.cfg,
    )
    

if __name__ == "__main__":
    
    main()
