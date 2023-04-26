import argparse
import os
import time
import numpy as np

import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqChrono, GamryCOM

# Define args
parser = argparse.ArgumentParser(description='Run repeating chronopotentiometry')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.chrono_decimate_args)
argc.add_args_from_dict(parser, argc.chrono_step_args)
parser.add_argument('--repeats', type=int, default=1)

if __name__ == '__main__':
    start_time = time.time()

    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Initialize dtaq. Need to leave cell on for repeats
    chrono = DtaqChrono('galv', write_mode='once', write_precision=6, exp_notes=args.exp_notes,
                        leave_cell_on=True, start_with_cell_off=False)

    # Set step sizes based on chrono_v_rms. If provided, ignore other args
    if args.chrono_v_rms is not None and not args.chrono_disable_find_i:
        # Test current
        v_oc = rf.test_ocv(pstat)
        s_rms = rf.find_current(pstat, v_oc + args.chrono_v_rms, 2.0)
        time.sleep(1)  # rest

        s_half_step = s_rms * np.sqrt(2)

        # Assume need to end at initial current
        args.chrono_geo_s_final = args.chrono_s_init
        args.chrono_geo_s_min = min(args.chrono_s_init, args.chrono_s_init + 2 * s_half_step)
        args.chrono_geo_s_max = max(args.chrono_s_init, args.chrono_s_init + 2 * s_half_step)

    # Configure decimation
    decimate = not args.disable_decimation
    if decimate:
        chrono.configure_decimation(args.decimate_during, args.decimation_prestep_points, args.decimation_interval,
                                    args.decimation_factor, args.decimation_max_t_sample)

    # Configure step signal
    chrono.configure_geostep_signal(args.chrono_s_init, args.chrono_geo_s_final,
                                    args.chrono_geo_s_min, args.chrono_geo_s_max,
                                    args.chrono_t_init, args.chrono_t_sample,
                                    args.chrono_geo_t_short, args.chrono_t_step,
                                    args.chrono_geo_num_scales, args.chrono_geo_steps_per_scale)

    # Get result file
    result_file = os.path.join(args.data_path, 'CHRONOP_{}.DTA'.format(args.file_suffix))

    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_IVT.DTA')
    else:
        kst_file = None

    print('Running CHRONO')
    chrono.run(pstat, result_file=result_file, kst_file=kst_file,
               decimate=decimate, show_plot=False, repeats=args.repeats)
    print('CHRONO done\n')

    # Turn cell off and close pstat
    pstat.SetCell(GamryCOM.CellOff)
    pstat.Close()

    print('Run time: {:.2f} s'.format(time.time() - start_time))
