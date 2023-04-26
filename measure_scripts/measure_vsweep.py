import argparse
# import os
# import numpy as np
# import matplotlib.pyplot as plt
import time
# from copy import deepcopy
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqOcv, DtaqChrono

# Define args
parser = argparse.ArgumentParser(description='Run fast polarization map')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.ocp_args)
argc.add_args_from_dict(parser, argc.vsweep_args)
argc.add_args_from_dict(parser, argc.chrono_decimate_args)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure OCV
    # Write to file point-by-point
    ocv = DtaqOcv(write_mode='interval', write_interval=1,
                  exp_notes=args.exp_notes)

    # Configure chrono for voltage sweep
    chrono = DtaqChrono(mode='pot', write_mode='once', write_precision=6, exp_notes=args.exp_notes)

    for n in range(args.num_loops):
        print(f'Beginning cycle {n}\n-----------------------------')
        # If repeating measurement, add indicator for cycle number
        if args.num_loops > 1:
            suffix = args.file_suffix + f'_#{n}'
        else:
            suffix = args.file_suffix

        # Run OCV
        # -------------------
        rf.run_ocv(ocv, pstat, args, suffix)
        time.sleep(1)

        # Get measured OCV
        V_oc = ocv.get_ocv(10)  # average last 10 values
        print('OCV: {:.3f} V'.format(V_oc))

        # Run chronoamperometry
        # ------------------------
        iv_df = rf.run_v_sweep(chrono, pstat, args, suffix, V_oc=V_oc)
