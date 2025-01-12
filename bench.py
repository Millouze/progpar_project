import subprocess
import argparse
import os

os.chdir("build")

def run_murb_once(bodies, iters,im):
	sp = subprocess.Popen(["./bin/murb", "-i", str(iters), "-n", str(bodies), "--nv","--im",im],stdout=subprocess.PIPE)
	output = sp.stdout.read()
	stats = output.splitlines()[-1].split()
	ms = float(stats[3])
	fps = float(stats[5][1:])
	return ms, fps

def run_murb_many(bodies, iters, n, im):
	min_ms = 0
	max_fps = 0

	for _ in range(n):
		ms, fps = run_murb_once(bodies, iters,im)
		if fps > max_fps:
			max_fps = fps
			min_ms = ms
	
	return min_ms, max_fps

def gen_arg_parser():
	parser = argparse.ArgumentParser(
        prog="murb-bench")
	parser.add_argument("-i", help="number of iterations", type=int, default=1000)
	parser.add_argument("-n", help="number of bodies", type=int, default=300)
	parser.add_argument("-k", help="number of times to run the program", type=int, default=10)
	parser.add_argument("--im", help="simulation used",type=str,default="cpu+naive")
	return parser

if __name__ == "__main__":
	parser = gen_arg_parser().parse_args()

	ms, fps = run_murb_many(parser.n, parser.i, parser.k, parser.im)
	print("{} FPS, {} ms".format(fps, ms))
