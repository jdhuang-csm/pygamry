import argparse
# import os
# import numpy as np
# import matplotlib
# matplotlib.use('QtAgg')
# import matplotlib.pyplot as plt
import time

import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqReadZ, DtaqOcv

# Define args
parser = argparse.ArgumentParser(description='Run OCP and EIS')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.ocp_args)
argc.add_args_from_dict(parser, argc.eis_args)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure OCV
    # Write to file every minute
    ocv = DtaqOcv(write_mode='interval', write_interval=1,
                  exp_notes=args.exp_notes)

    # Configure EIS
    # Write continuously
    eis = DtaqReadZ(mode=args.eis_mode, readzspeed='ReadZSpeedNorm',
                    write_mode='interval', write_interval=1,
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
        # print('Running OCV')
        # ocv_file = os.path.join(args.data_path, f'OCP_{suffix}.DTA')
        # ocv_kst_file = os.path.join(args.data_path, 'Kst_OCP.DTA')
        # ocv.run(pstat, args.ocp_duration, args.ocp_sample_period, show_plot=True, result_file=ocv_file,
        #         kst_file=ocv_kst_file)
        # print('OCV done\n')
        #
        # plt.close()
        time.sleep(1)

        # Get measured OCV for EIS
        V_oc = ocv.get_ocv(window=10)  # average last 10 values  # average last 10 values
        print('OCV: {:.3f} V'.format(V_oc))

        # Run EIS
        # -------------------
        rf.run_eis(eis, pstat, args, suffix, V_oc)
        # # Get frequencies to measure
        # num_decades = np.log10(args.eis_max_freq) - np.log10(args.eis_min_freq)
        # num_freq = int(args.eis_ppd * num_decades) + 1
        # eis_freq = np.logspace(np.log10(args.eis_max_freq), np.log10(args.eis_min_freq), num_freq)
        #
        # # Determine DC voltage
        # if args.eis_VDC_vs_VRef:
        #     V_dc = args.eis_VDC
        # else:
        #     V_dc = args.eis_VDC + V_oc
        #
        # print('Running EIS')
        # eis_file = os.path.join(args.data_path, f'EIS_{suffix}.DTA')
        # eis_kst_file = os.path.join(args.data_path, 'Kst_EIS.DTA')
        # eis.run(pstat, eis_freq, V_dc, args.eis_VAC, args.eis_Z_guess, timeout=1000,
        #         show_plot=True, plot_interval=1, plot_type='all',
        #         result_file=eis_file, kst_file=eis_kst_file)
        # print('EIS done\n')
        #
        # plt.close()
        time.sleep(1)
