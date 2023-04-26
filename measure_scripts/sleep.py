import time
import argparse

parser = argparse.ArgumentParser(description='Sleep for specified duration in seconds')
parser.add_argument('duration', type=float)

args = parser.parse_args()

print('Sleeping for {:.1f} s...'.format(args.duration))
time.sleep(args.duration)
