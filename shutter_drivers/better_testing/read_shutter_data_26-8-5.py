import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
from datetime import date

DATE = date.today()

def logistic(x, A, k, h, c):
    return (A / (1 + np.exp(-k*(x-h))) + c)

def opshutter(file: str = "./shutter_pd_rise-open_26-8-4.CSV", save: bool = True) -> bool:
    opshutter_df = pd.read_csv(file, skiprows=15)
    print(opshutter_df)
    time_arr = np.array(opshutter_df["TIME"]).astype(float)*1000
    fngen_arr = np.array(opshutter_df["CH1"]).astype(float)
    photodiode_arr = np.array(opshutter_df["CH2"]).astype(float)

    fngen_peak = np.max(fngen_arr)
    fngen_peaktime = time_arr[np.argmax(fngen_arr)]
    photodiode_peak = np.max(photodiode_arr)
    photodiode_risestart = time_arr[np.where(photodiode_arr >= 0.1*photodiode_peak)[0][0]]
    photodiode_risestop = time_arr[np.where(photodiode_arr >= 0.9*photodiode_peak)[0][0]]
    print(fngen_peaktime, photodiode_risestart, photodiode_risestop)

    fig, ax1 = plt.subplots(figsize=(14,6))
    scatter_fngen = ax1.scatter(time_arr, fngen_arr, s=0.5, color="blue", label=f"Function Generator Rise, Peak={fngen_peak:.2f}V")
    fngen_startline = ax1.axvline(x=fngen_peaktime, linestyle="--", color="blue", 
                                  label=f"Function Generator Signal Start: $t_0={abs(fngen_peaktime):.2f}ms$")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    ax1.set_ylabel("Function Generator Voltage Signal (V)", color="blue")
    ax1.set_xlabel("Signal Time (ms)")
    ax2 = ax1.twinx()
    scatter_pd = ax2.scatter(time_arr, photodiode_arr, s=0.5, color="orange", label=f"Photodiode Rising Voltage Signal, Peak = {photodiode_peak:.2f}V")
    scatter_startline = ax2.axvline(x=photodiode_risestart, linestyle="--", color="orange", 
                                    label=f"Photodiode Rising Voltage 10% Max: $t_{{10}}={photodiode_risestart:.2f}ms$")

    p0 = [0.5, 1, 2.5, 0.]
    #pf[1] = 9.129e3
    pf, _ = curve_fit(logistic, time_arr, photodiode_arr, p0=p0)
    fit_pd = ax2.plot(time_arr, logistic(time_arr, pf[0], pf[1], pf[2], pf[3]), color="red", 
                      label=f"Fitted Logistic Function to Photodiode Signal:\n$V(t)=\\frac{{{pf[0]:.2f}}}{{1+e^{{-{pf[1]:.2f}(t-{pf[2]:.2f})}}}}+{pf[3]:.2f}$ V")[0]
    scatter_endline = ax2.axvline(x=photodiode_risestop, linestyle="--", color="yellow",
                                  label=f"Photodiode Rising Voltage 90% Max: $t_{{90}}={photodiode_risestop:.2f}ms$")
    ax2.tick_params(axis='y', labelcolor="orange")
    ax2.set_ylabel("Photodiode Voltage Signal (V)", color="orange")

    scatterplots = [scatter_fngen, fngen_startline, scatter_pd, scatter_startline, scatter_endline, fit_pd]
    print(scatterplots)
    legend_labels = [p.get_label() for p in scatterplots]
    ax1.legend(scatterplots, legend_labels)

    ax1.set_title("Photodiode Signal, Rising Edge", fontsize=16)

    if (save):
        plt.savefig(f"shutter_pd_open-plot_{DATE}.jpg")
    plt.show()
    plt.close()
    return True

def clshutter(file: str = "./shutter_pd_fall-close_26-8-4.CSV", save: bool = True) -> bool:
    clshutter_df = pd.read_csv(file, skiprows=15)
    print(clshutter_df)
    time_arr = np.array(clshutter_df["TIME"]).astype(float)*1000
    fngen_arr = np.array(clshutter_df["CH1"]).astype(float)
    photodiode_arr = np.array(clshutter_df["CH2"]).astype(float)

    fngen_peak = np.min(fngen_arr)
    fngen_peaktime = time_arr[np.argmin(fngen_arr)]
    photodiode_peak = np.max(photodiode_arr)
    photodiode_fallstart = time_arr[np.where(photodiode_arr <= 0.9*photodiode_peak)[0][0]]
    photodiode_fallstop = time_arr[np.where(photodiode_arr <= 0.1*photodiode_peak)[0][0]]
    print(fngen_peaktime, photodiode_fallstart, photodiode_fallstop)

    fig, ax1 = plt.subplots(figsize=(14,6))
    scatter_fngen = ax1.scatter(time_arr, fngen_arr, s=0.5, color="blue", label=f"Function Generator Fall, Min={fngen_peak:.2f}V")
    fngen_endline = ax1.axvline(x=fngen_peaktime, linestyle="--", color="blue", 
                                  label=f"Function Generator Signal End: $t_0={abs(fngen_peaktime):.2f}ms$")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    ax1.set_ylabel("Function Generator Voltage Signal (V)", color="blue")
    ax1.set_xlabel("Signal Time (ms)")
    ax2 = ax1.twinx()
    scatter_pd = ax2.scatter(time_arr, photodiode_arr, s=0.5, color="orange", label=f"Photodiode Falling Voltage Signal, Peak = {photodiode_peak:.2f}V")
    scatter_startline = ax2.axvline(x=photodiode_fallstart, linestyle="--", color="orange", 
                                    label=f"Photodiode Falling Voltage 90% Max: $t_{{90}}={photodiode_fallstart:.2f}ms$")

    p0 = [-0.5, 5, 521, 0.5]
    #pf[1] = 9.129e3
    pf, _ = curve_fit(logistic, time_arr, photodiode_arr, p0=p0)
    fit_pd = ax2.plot(time_arr, logistic(time_arr, pf[0], pf[1], pf[2], pf[3]), color="red", 
                      label=f"Fitted Logistic Function to Photodiode Signal:\n$V(t)=\\frac{{{pf[0]:.2f}}}{{1+e^{{-{pf[1]:.2f}(t-{pf[2]:.2f})}}}}+{pf[3]:.2f}$ V")[0]
    scatter_endline = ax2.axvline(x=photodiode_fallstop, linestyle="--", color="yellow",
                                  label=f"Photodiode Falling Voltage 10% Max: $t_{{10}}={photodiode_fallstop:.2f}ms$")
    ax2.tick_params(axis='y', labelcolor="orange")
    ax2.set_ylabel("Photodiode Voltage Signal (V)", color="orange")

    scatterplots = [scatter_fngen, fngen_endline, scatter_pd, scatter_startline, scatter_endline, fit_pd]
    print(scatterplots)
    legend_labels = [p.get_label() for p in scatterplots]
    ax1.legend(scatterplots, legend_labels, loc="center left")

    ax1.set_title("Photodiode Signal, Falling Edge", fontsize=16)

    if (save):
        plt.savefig(f"shutter_pd_close-plot_{DATE}.jpg")
    plt.show()
    plt.close()
    return True

if __name__ == "__main__":
    save: bool
    if (len(sys.argv) > 2):
        if sys.argv[2] == "save":
            save = True
    else:
        save = False
    if sys.argv[1] == "open":
        opshutter(save=save)
    elif sys.argv[1] == "close":
        clshutter(save=save)