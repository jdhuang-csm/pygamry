import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, default=None)

args = parser.parse_args()
# print(args.text.find('\\t'))
# print(ast.literal_eval(args.text))
print(args.text)
# print(codecs.decode(args.text, 'unicode_escape'))
