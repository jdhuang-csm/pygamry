import argparse
import time
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqPwrPol

# Define args
parser = argparse.ArgumentParser(description='Run jV curve')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.pwrpol_args)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    # Configure PWRPOL
    # Write continuously
    pwrpol = DtaqPwrPol(write_mode='interval', write_interval=1, exp_notes=args.exp_notes)

    for n in range(args.num_loops):
        print(f'Beginning cycle {n}\n-----------------------------')
        # If repeating measurement, add indicator for cycle number
        if args.num_loops > 1:
            suffix = args.file_suffix + f'_#{n}'
        else:
            suffix = args.file_suffix

        # Run pwrpol
        # ------------------

        pstat.Open()
        pstat.SetIERange(12)
        pstat.SetIchRange(2)
        print('IERange:', pstat.IERange())
        rf.run_pwrpol(pwrpol, pstat, args, suffix)
        time.sleep(1)
