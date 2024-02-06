import argparse
from copy import deepcopy
# import os
# import numpy as np
# import matplotlib.pyplot as plt
import time
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqPstatic, DtaqReadZ

# Define args
parser = argparse.ArgumentParser(description='Run potentiostatic scan followed by EIS at bias')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.eis_args)
argc.add_args_from_dict(parser, argc.pstatic_args)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure EIS
    # Write continuously
    eis = DtaqReadZ(mode='pot', readzspeed='ReadZSpeedNorm', write_mode='interval', write_interval=1,
                    exp_notes=args.exp_notes)

    # Configure PSTATIC
    # Write every minute
    pstatic = DtaqPstatic(write_mode='interval', write_interval=1,
                          exp_notes=args.exp_notes, leave_cell_on=True)

    for n in range(args.num_loops):
        print(f'Beginning cycle {n}\n-----------------------------')
        # If repeating measurement, add indicator for cycle number
        if args.num_loops > 1:
            suffix = args.file_suffix + f'_#{n}'
        else:
            suffix = args.file_suffix

        # Run PSTATIC
        # -------------------
        V_oc = rf.test_ocv(pstat)
        rf.run_pstatic(pstatic, pstat, args, suffix, V_oc, pwrpol_dtaq=None)

        # Run EIS at bias
        # -------------------
        # Set EIS bias to match PSTATIC
        bias_args = deepcopy(args)
        bias_args.eis_SDC = args.pstatic_VDC
        bias_args.eis_VDC_vs_VRef = args.pstatic_VDC_vs_VRef
        bias_suffix = 'VDC={}_{}'.format(args.pstatic_VDC, suffix)
        eis.start_with_cell_off = False
        rf.run_eis(eis, pstat, bias_args, bias_suffix, V_oc=V_oc, show_plot=False)

        # plt.close()
        time.sleep(1)
