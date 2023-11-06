import argparse
# import os
import numpy as np
# import matplotlib.pyplot as plt
import time
# from copy import deepcopy
import arg_config as argc
import run_functions as rf
import pandas as pd

from pygamry.dtaq import get_pstat, DtaqOcv, DtaqChrono, HybridSequencer

# Define args
parser = argparse.ArgumentParser(description='Run fast polarization map')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.ocp_args)
argc.add_args_from_dict(parser, argc.hybrid_args)
argc.add_args_from_dict(parser, argc.staircase_args)

# Add subset of vsweep args (others are inferred from staircase args
argc.add_args_from_dict(
    parser,
    {
        '--vsweep_t_init': dict(type=float, default=1),
        '--vsweep_t_sample': dict(type=float, default=5e-3),
        '--vsweep_t_step': dict(type=float, default=5),
        '--vsweep_ocv_equil': dict(default=False, action='store_true'),
        '--vsweep_i_max': dict(type=float, default=1.0),
        '--vsweep_file': dict(type=str, default='none'),
        '--min_i_step': dict(type=float, default=-1)
    }
)

parser.add_argument('--disable_vsweep', action='store_true')

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Match vsweep args to staircase args
    args.vsweep_v_rms = args.hybrid_v_rms
    args.vsweep_num_steps = args.staircase_num_steps
    args.vsweep_direction = args.staircase_direction
    args.vsweep_rest_time = args.staircase_equil_time
    args.vsweep_v_min = args.staircase_v_min - 0.03
    args.vsweep_v_max = args.staircase_v_max + 0.03
    args.vsweep_rest_time = args.staircase_equil_time

    # Get pstat
    pstat = get_pstat()

    # Configure OCV
    # Write to file point-by-point
    ocv = DtaqOcv(write_mode='interval', write_interval=1,
                  exp_notes=args.exp_notes)

    # Configure chrono for voltage sweep
    chrono = DtaqChrono(mode='pot', write_mode='once', write_precision=6, exp_notes=args.exp_notes)

    # Configure hybrid sequencer
    seq = HybridSequencer(chrono_mode='galv', eis_mode='galv',
                          update_step_size=not args.staircase_constant_step_size,
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

        # Get measured OCV
        V_oc = ocv.get_ocv(10)  # average last 10 values
        print('OCV: {:.3f} V'.format(V_oc))

        # Run chronoamperometry
        # ------------------------
        if args.vsweep_file != 'none':
            iv_df = pd.read_csv(args.vsweep_file, sep='\t')
        elif not args.disable_vsweep:
            iv_df = rf.run_v_sweep(chrono, pstat, args, suffix, V_oc=V_oc)
        else:
            iv_df = None

        # TODO: figure out a better solution for this
        #  TEMP: in case that long-timescale transport losses cause vsweep current to decrease with increasing voltage
        #  Adjust the IV input to the hybrid sequencer to ensure that current increases monotonically
        #  with increasing voltage
        # Sort by voltage
        iv_df = iv_df.copy()
        sort_index = np.argsort(iv_df['Vf'].values)
        i_sort = iv_df['Im'].values[sort_index]
        v_sort = iv_df['Vf'].values[sort_index]
        # Enforce a minimum current step
        i_diff = np.diff(i_sort)
        if args.min_i_step == -1:
            # Use robust outlier thresh to set min step
            # min_i_step = 0.001
            iqr = np.percentile(i_diff, 75) - np.percentile(i_diff, 25)
            min_i_step = np.percentile(i_diff, 25) - 1.5 * iqr
        else:
            min_i_step = args.min_i_step
        print('min_i_step: {:.2e} A'.format(min_i_step))
        i_diff[i_diff < min_i_step] = min_i_step
        cumsum = np.cumsum(i_diff)
        cumsum = np.insert(cumsum, 0, 0)
        i_sort = i_sort[0] + cumsum
        iv_df['Im'] = i_sort
        iv_df['Vf'] = v_sort

        # Run hybrid staircase
        # -----------------------
        rf.run_hybrid_staircase(seq, pstat, args, suffix, jv_data=iv_df)
        time.sleep(1)
