import os
import shutil 
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tiles_dir')
parser.add_argument('tms_dir')
args = parser.parse_args()


if os.path.isdir(args.tms_dir):
    shutil.rmtree(args.tms_dir)
os.mkdir(args.tms_dir)

for fn in os.listdir(args.tiles_dir):
    z, y, x = map(int, re.match(r'(\d+)_(\d+)_(\d+)\.png', fn).groups())
    fn = os.path.join(args.tiles_dir, fn)
    y = 2 ** z - y - 1
    trg_dir = os.path.join(args.tms_dir, str(z), str(x))
    trg_name = os.path.join(trg_dir, '%s.png' % y )
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)
    os.rename(fn, trg_name)