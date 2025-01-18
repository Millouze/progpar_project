import argparse
import os
from numpy import genfromtxt, string_
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

title_dict = {'fontsize': 30,
 'fontweight' : 30}

fps = []
ms = []
gf = []

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
 
def run_plots(version, index_tab):
    a = 0
    
    if index_tab=="ocl":
        index_array = index_ocl
    elif index_tab=="cuda":
        index_array = index_cuda
    else:
        index_array = index_simd

    for i in index_array:
        data = pd.read_csv(version+"_"+str(i)+".csv")
        fps.append(data['fps'].sum()/5)
        ms.append(data['ms'].sum()/5)
        gf.append(data['gf'].sum()/5)
        a+=1

    print(fps)
    print(ms)
    print(gf)

def gen_arg_parser():
    parser = argparse.ArgumentParser(
        prog="murb-bench")
    parser.add_argument("-v", help="version to plot", type=str, default="cpu+LessComplex")
    parser.add_argument("-i", help="index array to choose between ocl|cuda|simd", type=str, default="simd")
    return parser

if __name__ == "__main__":
    parser = gen_arg_parser().parse_args()
    plot_30K()
    run_plots(parser.v, parser.i)

