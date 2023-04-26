import argparse
import os
import time
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqChrono, GamryCOM

# Define args
parser = argparse.ArgumentParser(description='Run chronopotentiometry/chronoamperometry')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.chrono_decimate_args)
argc.add_args_from_dict(parser, argc.chrono_step_args)
parser.add_argument('--chrono_mode', type=str, default='galv')

if __name__ == '__main__':
    start_time = time.time()

    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Initialize dtaq
    chrono = DtaqChrono(args.chrono_mode, write_mode='once', write_precision=6, exp_notes=args.exp_notes)

    # Run chrono
    rf.run_chrono(chrono, pstat, args, args.file_suffix)

    print('Run time: {:.2f} s'.format(time.time() - start_time))
