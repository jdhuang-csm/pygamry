import argparse
# import time
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqOcv
from pygamry.sequencers import HybridSequencer

# Define args
parser = argparse.ArgumentParser(description='Run hybrid measurement')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.hybrid_args)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure sequencer
    seq = HybridSequencer(mode='galv', update_step_size=True, exp_notes=args.exp_notes)

    for n in range(args.num_loops):
        print(f'Beginning cycle {n}\n-----------------------------')
        # If repeating measurement, add indicator for cycle number
        if args.num_loops > 1:
            suffix = args.file_suffix + f'_#{n}'
        else:
            suffix = args.file_suffix

        # rf.test_resistance(pstat)
        rf.run_hybrid(seq, pstat, args, suffix, show_plot=False)







