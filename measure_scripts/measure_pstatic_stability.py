import argparse
from copy import deepcopy
# import os
# import numpy as np
# import matplotlib.pyplot as plt
import time
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqReadZ, DtaqOcv, DtaqPwrPol, DtaqPstatic

# Define args
parser = argparse.ArgumentParser(description='Run potentiostatic stability test')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.ocp_args)
argc.add_args_from_dict(parser, argc.eis_args)
argc.add_args_from_dict(parser, argc.pwrpol_args)
argc.add_args_from_dict(parser, argc.pstatic_args)

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
    eis = DtaqReadZ(mode='pot', readzspeed='ReadZSpeedNorm', write_mode='interval', write_interval=1,
                    exp_notes=args.exp_notes)

    # Configure PWRPOL
    # Write continuously
    pwrpol = DtaqPwrPol(write_mode='interval', write_interval=1, exp_notes=args.exp_notes)

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

        # Run OCV
        # -------------------
        rf.run_ocv(ocv, pstat, args, suffix, show_plot=False)
        time.sleep(1)

        # Get measured OCV for EIS
        V_oc = ocv.get_ocv(window=10)  # average last 10 values
        print('OCV: {:.3f} V'.format(V_oc))

        # Run EIS
        # -------------------
        eis.start_with_cell_off = True
        rf.run_eis(eis, pstat, args, suffix, V_oc, show_plot=False)
        # # Get frequencies to measure
        # num_decades = np.log10(args.eis_max_freq) - np.log10(args.eis_min_freq)
        # num_freq = int(args.eis_ppd * num_decades) + 1
        # eis_freq = np.logspace(np.log10(args.eis_max_freq), np.log10(args.eis_min_freq), num_freq)
        #
        # # Determine DC voltage
        # if args.eis_VDC_vs_VRef:
        #     eis_VDC = args.eis_VDC
        # else:
        #     eis_VDC = args.eis_VDC + V_oc
        #
        # print('Running EIS')
        # eis_file = os.path.join(args.data_path, f'EIS_{suffix}.DTA')
        # eis_kst_file = os.path.join(args.data_path, 'Kst_EIS.DTA')
        # eis.run(pstat, eis_freq, eis_VDC, args.eis_VAC, args.eis_Z_guess, timeout=1000,
        #         show_plot=True, plot_interval=1, plot_type='all',
        #         result_file=eis_file, kst_file=eis_kst_file)
        # print('EIS done\n')
        #
        # plt.close()
        time.sleep(1)

        # Run PWRPOL
        # ------------------
        rf.run_pwrpol(pwrpol, pstat, args, suffix, show_plot=False)
        # print('Running PWRPOL')
        # pwrpol_file = os.path.join(args.data_path, f'PWRPOLARIZATION_{suffix}.DTA')
        # pwrpol_kst_file = os.path.join(args.data_path, 'Kst_PWRPOL.DTA')
        # pwrpol.run(pstat, args.pwrpol_i_final, args.pwrpol_scan_rate, args.pwrpol_sample_period,
        #            v_min=args.pwrpol_v_min, v_max=args.pwrpol_v_max,
        #            show_plot=True, result_file=pwrpol_file, kst_file=pwrpol_kst_file)
        # print('PWRPOL done\n')
        #
        # plt.close()
        time.sleep(1)

        # plt.close()
        time.sleep(1)
