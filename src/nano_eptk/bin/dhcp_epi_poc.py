#!/usr/bin/env python

import argparse
import pathlib
import json

from nano_eptk.eppipe.dhcp_epi import dhcp_epi_poc


def parse_args():
    
    # Initialize parser
    parser = argparse.ArgumentParser(
        prog="dhcp_epi_poc",
        description=(
            "dhcp_epi_poc: part of nano eptk package.\n"
            "\n"
            "Apply the dHCP EPI-POC pipeline on input data.\n"
            "\n"    
            "Required arguments:\n"
            " - Input magnitude.\n"
            " - Input phase.\n"
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
    # Input magnitude
    parser.add_argument(
        "--mag", 
        type=pathlib.Path,
        help="Input magnitude.",
        default=None,
        metavar="\b",
    )
    
    # Input phase
    parser.add_argument(
        "--pha", 
        type=pathlib.Path,
        help="Input phase.",
        default=None,
        metavar="\b",
    )
    
    # Input mask
    parser.add_argument(
        "--mask", 
        type=pathlib.Path,
        help="Input mask.",
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
    
    # Input mask artefact
    parser.add_argument(
        "--mask_artefact",
        type=pathlib.Path,
        help="Input mask artefact.",
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


def execute_dhcp_epi_poc_with_datalist(
        input_mag_path=None,
        input_pha_path=None,
        input_mask_path=None,
        output_sig_path=None,
        datalist_path=None,
        input_dhcp_labels9_path=None,
        input_mask_artefact_path=None,
        output_ep_metrics_path=None,
        **kwargs,
    ):
    
    # Required datalist arguments
    keys = [
        "input_mag_path",
        "input_pha_path",
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
   
            # Run dHCP EPI POC
            dhcp_epi_poc(**kwargs)
        
    elif (input_mag_path and 
          input_pha_path and
          input_mask_path and
          output_sig_path):
        
        # Run dHCP EPI POC
        dhcp_epi_poc(
            input_mag_path=input_mag_path,
            input_pha_path=input_pha_path,
            input_mask_path=input_mask_path,
            output_sig_path=output_sig_path,
            input_dhcp_labels9_path=input_dhcp_labels9_path,
            input_mask_artefact_path=input_mask_artefact_path,
            output_ep_metrics_path=output_ep_metrics_path,
            **kwargs,
        )
        

def main():
    
    # Parse input arguments
    args = parse_args()

    # Main function call
    execute_dhcp_epi_poc_with_datalist(
        input_mag_path=args.mag,
        input_pha_path=args.pha,
        input_mask_path=args.mask,
        output_sig_path=args.sig,
        datalist_path=args.datalist,
        input_dhcp_labels9_path=args.dhcp_labels9,
        input_mask_artefact_path=args.mask_artefact,
        output_ep_metrics_path=args.ep_metric,
        verbose=args.verbose,
        debug=args.debug,
        cfg_path=args.cfg,
    )
    

if __name__ == "__main__":
    
    main()
