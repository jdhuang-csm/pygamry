import argparse
# import os
# import numpy as np
# import matplotlib.pyplot as plt
import time
# from copy import deepcopy
import arg_config as argc
from pygamry.dtaq import DtaqReadZ
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqOcv, DtaqPwrPol, HybridSequencer

# Define args
parser = argparse.ArgumentParser(description='Run OCP, EIS, jv, and staircase')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.ocp_args)
argc.add_args_from_dict(parser, argc.eis_args)
argc.add_args_from_dict(parser, argc.pwrpol_args)
argc.add_args_from_dict(parser, argc.hybrid_args)
argc.add_args_from_dict(parser, argc.staircase_args)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure OCV
    # Write to file point-by-point
    ocv = DtaqOcv(write_mode='interval', write_interval=1,
                  exp_notes=args.exp_notes)

    # Configure EIS
    # Write continuously
    eis = DtaqReadZ(mode=args.eis_mode, readzspeed='ReadZSpeedNorm', write_mode='interval', write_interval=1,
                    exp_notes=args.exp_notes)

    # Configure PWRPOL
    # Write continuously
    pwrpol = DtaqPwrPol(write_mode='interval', write_interval=1, exp_notes=args.exp_notes)

    # Configure hybrid sequencer
    seq = HybridSequencer(mode='galv', update_step_size=not args.staircase_constant_step_size,
                          exp_notes=args.exp_notes)

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

        # Get measured OCV for EIS
        V_oc = ocv.get_ocv(10)  # average last 10 values
        print('OCV: {:.3f} V'.format(V_oc))

        # Run EIS
        # -------------------
        rf.run_eis(eis, pstat, args, suffix, V_oc)
        time.sleep(1)

        # Run pwrpol
        # ------------------
        rf.run_pwrpol(pwrpol, pstat, args, suffix)
        time.sleep(1)

        # Run hybrid staircase
        # -----------------------
        rf.run_hybrid_staircase(seq, pstat, args, suffix, jv_data=pwrpol.result_file)
        time.sleep(1)

