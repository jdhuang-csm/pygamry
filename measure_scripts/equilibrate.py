import argparse
from copy import deepcopy
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import arg_config as argc
import run_functions as rf

from pygamry.dtaq import get_pstat

# Define args
parser = argparse.ArgumentParser(description='Run equilibration')
# Add predefined arguments
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.equil_args)
argc.add_args_from_dict(parser, argc.pstatic_equil_args)
argc.add_args_from_dict(parser, argc.gstatic_equil_args)


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Get pstat
    pstat = get_pstat()

    rf.equilibrate(pstat, args, args.file_suffix)
