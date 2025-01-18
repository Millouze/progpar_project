import argparse
import statistics as stat
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


index_cuda = [1000, 6000, 11000]
index_ocl = [1000, 5000, 10000]
index_simd = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
columns = ["fps", "ms", "gf"]
implems_cpu = ["Naive", "Triangle", "SIMD_base", "SIMD"]
implems_omp = ["OpenMP", "SIMD+OpenMP"]
implems_gpu = ["OpenCL", "CUDA_AoS","CUDA","CUDA_opti"]
implems_cuda = ["CUDA_AoS","CUDA","CUDA_opti"]

array_30K_cpu = [0.014369, 0.0655738, 0.292821, 0.385509]
array_30K_omp = [0.439366, 1.71493]
array_30K_cuda = [9.85257, 10.4196, 16.7645]

title_dict = {'fontsize': 10,
 'fontweight' : 30}
error_style = {'capsize' : 2, 'elinewidth' : 2, 'capthick' : 5, 'alpha' : 0.5}

def plot_30K():
    fig, ax = plt.subplots()
    ax.bar(implems_cpu, array_30K_cpu, color='green', edgecolor='lightblue', linewidth=3)
    ax.bar(implems_omp, array_30K_omp, color='green', edgecolor='purple', linewidth=4)
    ax.bar('OpenCL', 9.30688, color='lightblue', edgecolor='orange', linewidth=4)
    ax.bar(implems_cuda, array_30K_cuda, color='lightblue', edgecolor='darkred', linewidth=4)
    plt.ylabel('FPS count', fontsize=30, color='red')
    plt.xlabel('version', fontsize=30, color='green')
    plt.title("Performance Speedup Graph For 30K Bodies", title_dict)
    plt.gca().legend(('cpu','cpu+omp', 'opencl', 'cuda'), loc="upper left")
    plt.show()
 
def run_plots(version, index_tab, run_nb, display):
    a = 0
    fps_dev = []
    ms_dev = []
    gf_dev = []
    fps_avg = []
    ms_avg = []
    gf_avg = []
    
    if index_tab=="ocl":
        index_array = index_ocl
    elif index_tab=="cuda":
        index_array = index_cuda
    else:
        index_array = index_simd

    for i in index_array:
        data = pd.read_csv(version+"_"+str(i)+".csv")
        fps_avg.append(data['fps'].sum()/run_nb)
        ms_avg.append(data['ms'].sum()/run_nb)
        gf_avg.append(data['gf'].sum()/run_nb)
        fps_dev.append(stat.stdev(data['fps']))
        ms_dev.append(stat.stdev(data['ms']))
        gf_dev.append(stat.stdev(data['gf']))
        a= a+1

    width=1000
    X_axis = np.array(index_array)
    plt.bar(X_axis - width/2, fps_avg, width=1000, yerr=fps_dev, label="FPS mean", error_kw=error_style)
    plt.bar(X_axis + width/2, gf_avg, width=1000, yerr=gf_dev, label="Gflops mean", error_kw=error_style)
    plt.xticks(X_axis)
    plt.ylabel('FPS mean / GFlops mean', fontsize=10, color='red')
    plt.xlabel('Number of bodies', fontsize=10, color='green')
    plt.title("Average FPS for Different Value of Bodies for "+display+" version", title_dict)
    plt.savefig(display, dpi=400)
    plt.legend()
    plt.show()


def gen_arg_parser():
    parser = argparse.ArgumentParser(
        prog="murb-bench")
    parser.add_argument("-v", help="version to plot", type=str, default="cpu+SIMD")
    parser.add_argument("-i", help="index array to choose between ocl|cuda|simd", type=str, default="simd")
    parser.add_argument("-ite", help="size of run samples", type=int, default=5)
    parser.add_argument("-display", help="for display purposes", type=str, default="precisez version")
    return parser

if __name__ == "__main__":
    parser = gen_arg_parser().parse_args()
    # plot_30K()
    run_plots(parser.v, parser.i, parser.ite, parser.display)

