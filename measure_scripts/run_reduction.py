import argparse
import os
import arg_config as argc

from pygamry.dtaq import get_pstat
from pygamry.reduction import DtaqReduction


parser = argparse.ArgumentParser(description='Run reduction procedure')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.ocp_args)

parser.add_argument('reduction_config_file', type=str)
# parser.add_argument('--ocp_duration', type=float, default=120e3)
# parser.add_argument('--ocp_sample_period', type=float, default=10)

if __name__ == '__main__':
    args = parser.parse_args()

    pstat = get_pstat()

    # Write to file point-by-point
    dtaq = DtaqReduction(args.reduction_config_file, write_mode='interval',
                         write_interval=1,
                         exp_notes=args.exp_notes)

    result_file = os.path.join(args.data_path, f'OCP_{args.file_suffix}.DTA')
    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_OCP.DTA')
    else:
        kst_file = None

    dtaq.run(pstat, args.ocp_duration, args.ocp_sample_period, args.data_path,
             result_file=result_file, kst_file=kst_file,
             show_plot=False)

