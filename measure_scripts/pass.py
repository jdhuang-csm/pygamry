# Do nothing
import time
import argparse
import arg_config as argc

# Define args
parser = argparse.ArgumentParser(description='Do nothing for 1 second')
# Add standard arguments - these will be ignored
argc.add_args_from_dict(parser, argc.common_args)

time.sleep(1)
