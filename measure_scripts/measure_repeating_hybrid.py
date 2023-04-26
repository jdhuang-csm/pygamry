import argparse
import os
# import time
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqChrono
from pygamry.sequencers import HybridSequencer

# Define args
parser = argparse.ArgumentParser(description='Run hybrid measurement')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.hybrid_args)


parser.add_argument('--condition_time', type=float, default=0)
parser.add_argument('--condition_t_sample', type=float, default=1e-3)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--stop_v_min', type=float, default=-1)
parser.add_argument('--stop_v_max', type=float, default=1)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure sequencer
    seq = HybridSequencer(mode='galv', update_step_size=True, exp_notes=args.exp_notes)

    # Condition
    if args.condition_time > 0:
        dt_chrono = DtaqChrono(mode='galv')

        dt_chrono.configure_mstep_signal(0, args.hybrid_i_init, 1,
                                         args.condition_time, args.condition_t_sample, n_steps=1
                                         )
        dt_chrono.leave_cell_on = True
        start_with_cell_off = False

        dt_chrono.configure_decimation('write', 20, 10, 2, 1)

        print('Conditioning at {:.3f} A for {:.0f} s...'.format(args.hybrid_i_init, args.condition_time))
        chrono_file = os.path.join(args.data_path, f'Conditioning_{args.file_suffix}.DTA')

        if args.kst_path is not None:
            kst_file = os.path.join(args.kst_path, 'Kst_IVT.DTA')
        else:
            kst_file = None

        dt_chrono.run(pstat, result_file=chrono_file, kst_file=kst_file, decimate=True)
    else:
        start_with_cell_off = True

    # Run hybrid measurements
    # TODO: incorporate end_at_init
    leave_cell_on = True
    for n in range(args.repeats):
        print(f'Beginning cycle {n}\n-----------------------------')
        # If repeating measurement, add indicator for cycle number
        if args.repeats > 1:
            suffix = args.file_suffix + f'_Cycle{n}'
        else:
            suffix = args.file_suffix

        # After first run, start with cell on
        if n > 0:
            start_with_cell_off = False

        # At last run, turn cell off
        if n == args.repeats - 1:
            leave_cell_on = False

        rf.run_hybrid(seq, pstat, args, suffix, show_plot=False,
                      start_with_cell_off=start_with_cell_off,
                      leave_cell_on=leave_cell_on)

        # Check voltage limits
        if seq.meas_v_min <= args.stop_v_min:
            print('STOPPING TEST: measured voltage {:.3f} V is below low threshold ({:.3f} V)'.format(
                seq.meas_v_min, args.stop_v_min)
            )

        if seq.meas_v_max >= args.stop_v_max:
            print('STOPPING TEST: measured voltage {:.3f} V is above high threshold ({:.3f} V)'.format(
                seq.meas_v_max, args.stop_v_max)
            )









