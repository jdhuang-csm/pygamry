import argparse
import os
import time
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

    # Initialize dtaq
    chrono = DtaqChrono('galv', write_mode='once', write_precision=6, exp_notes=args.exp_notes,
                        leave_cell_on=True, start_with_cell_off=False)

    # # Configure decimation
    # chrono.configure_decimation(args.decimate_during, args.prestep_points, args.decimation_interval,
    #                             args.decimation_factor, args.max_t_sample)
    #
    # # Configure step signal
    # chrono.configure_dstep_signal(args.s_init, args.s_step1, args.s_step2, args.t_init, args.t_step1, args.t_step2,
    #                               args.t_sample)
    #
    # # Open pstat and turn cell off before starting run
    # pstat.Open()
    # pstat.SetCell(GamryCOM.CellOff)
    #
    # # Run
    # result_file = os.path.join(args.data_path, 'CHRONOP_{}.DTA'.format(args.file_suffix))
    # kst_file = os.path.join(args.data_path, 'Kst_IVT.DTA')
    # chrono.run(pstat, result_file=result_file, kst_file=kst_file,
    #            decimate=True, show_plot=False, repeats=args.repeats)

    rf.run_chrono(chrono, pstat, args, args.file_suffix, repeats=args.repeats)

    # Turn cell off and close pstat
    pstat.SetCell(GamryCOM.CellOff)
    pstat.Close()

    print('Run time: {:.2f} s'.format(time.time() - start_time))
