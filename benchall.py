import subprocess
import argparse
import os

os.chdir("build")

implems = ["cpu+SIMD", "cpu+SIMD_2"]


def run_murb_once(bodies, iters, im):
    sp = subprocess.Popen(
        ["./bin/murb", "-i", str(iters), "-n", str(bodies), "--nv", "--gf", "--im", im],
        stdout=subprocess.PIPE,
    )
    output = sp.stdout.read()
    stats = output.splitlines()[-1].split()
    ms = float(stats[3])
    fps = float(stats[5][1:])
    gf = float(stats[7])
    return ms, fps, gf


def run_murb_many(bodies, iters, n, im):
    min_ms = 0
    max_fps = 0
    max_gf = 0
    fpss = []
    mss = []
    gfs = []

    for _ in range(n):
        ms, fps, gf = run_murb_once(bodies, iters, im)
        fpss.append(fps)
        mss.append(ms)
        gfs.append(gf)
        if fps > max_fps:
            max_fps = fps
            min_ms = ms
            max_gf = gf
    return min_ms, max_fps, max_gf, fpss, mss, gfs


def gen_arg_parser():
    parser = argparse.ArgumentParser(prog="murb-bench")
    parser.add_argument("-i", help="number of iterations", type=int, default=1000)
    parser.add_argument("-n", help="number of bodies", type=int, default=10000)
    parser.add_argument(
        "-k", help="number of times to run the program", type=int, default=10
    )
    parser.add_argument("--im", help="simulation used", type=str, default="cpu+naive")
    return parser


if __name__ == "__main__":
    parser = gen_arg_parser().parse_args()

    for im in implems:
        f = open("../bench/" + im + ".csv", "w")
        print("running bench for " + im + "\n")
        f.write("bodies,fps,ms,gf\n")
        for n in range(1000, parser.n, int(parser.n / 1000) * 100):
            f2 = open("../bench/" + im + "_" + str(n) + ".csv", "w")
            print(str(n) + " bodies")
            f2.write("fps,ms,gf\n")
            ms, fps, gf, fpss, mss, gfs = run_murb_many(n, 1000, parser.k, im)
            for a, b, c in zip(fpss, mss, gfs):
                f2.write(str(a) + "," + str(b) +","+str(c)+ "\n")
            f.write(str(n) + "," + str(fps) + "," + str(ms) +","+str(gf)+"\n")

""" print("{} FPS, {} ms".format(fps, ms)) """
