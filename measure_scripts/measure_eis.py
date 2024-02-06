import argparse
# import os
# import numpy as np
# import matplotlib
# matplotlib.use('QtAgg')
# import matplotlib.pyplot as plt
import time

import arg_config as argc
from pygamry.dtaq import DtaqReadZ
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqOcv

# Define args
parser = argparse.ArgumentParser(description='Run EIS')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.ocp_args)
argc.add_args_from_dict(parser, argc.eis_args)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure EIS
    # Write at each frequency
    eis = DtaqReadZ(mode=args.eis_mode, readzspeed='ReadZSpeedNorm', write_mode='interval', write_interval=1,
                    exp_notes=args.exp_notes)

    for n in range(args.num_loops):
        print(f'Beginning cycle {n}\n-----------------------------')
        # If repeating measurement, add indicator for cycle number
        if args.num_loops > 1:
            suffix = args.file_suffix + f'_#{n}'
        else:
            suffix = args.file_suffix

        # Get OCV
        # -------------------
        V_oc = rf.test_ocv(pstat, num_points=9)        

        # Run EIS
        # -------------------
        rf.run_eis(eis, pstat, args, suffix, V_oc)
       
