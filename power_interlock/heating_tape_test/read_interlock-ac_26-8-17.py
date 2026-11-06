import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date

# Note: These datafiles were both with Heating Tape (101.2Ohm) connected

def sine(x, A, k, h, c):
    return ((A * np.sin(2*np.pi*k * (x - h))) + c)

def sine_lin_bg(x, A, k, h, m, b):
    return ((A * np.sin(2*np.pi*k * (x - h))) + m * x + b)

def find_sinusoidal_period_idx(arr):
    # Helper Function finds index estimating the half-period of a sinusoidal signal (over time)
    # finds where derivative is approximately 0
    sign = (arr[10] - arr[0]) // abs(arr[10] - arr[0])
    
    prev_idx10 = 0
    for curr_idx in range(len(arr)):
        if (curr_idx < 9): 
            curr_idx += 1
            continue
        if ((sign > 0) and (arr[curr_idx] - arr[prev_idx10] <= 1.)) or \
           ((sign < 0) and (arr[curr_idx] - arr[prev_idx10] >= -1.)):
            break
        prev_idx10 += 1
        curr_idx += 1
    return curr_idx

def find_high_cutoff_idx(arr, slide_window_size=2, threshold=2) -> int | None:
    # Helper Function finds cutoff array index between two different sinusoidal signal forms, given a window size
    # Takes Higher indices waveform
    print(arr[slide_window_size] - arr[0])
    for idx in range(0, len(arr) - slide_window_size):
        curr_ampl = arr[idx + slide_window_size] - arr[idx]
        print(curr_ampl)
        if (threshold > 0):
            if (abs(curr_ampl) > threshold):
                return idx
        else:
            if (abs(curr_ampl) < abs(threshold)):
                return idx            
    return -1

def find_low_cutoff_idx(arr, slide_window_size=10, threshold=0.2) -> int | None:
    # Helper Function finds cutoff array index between two different sinusoidal signal forms, given a window size
    # Takes Lower indices waveform
    baseline_ampl = arr[slide_window_size] - arr[0]
    for idx in range(0, len(arr) - slide_window_size, slide_window_size):
        curr_ampl = arr[idx + slide_window_size] - arr[idx]
        if (threshold > 0):
            if (abs(curr_ampl) > abs(baseline_ampl) + threshold):
                return idx
        else:
            if (abs(curr_ampl) < abs(baseline_ampl) + threshold):
                return idx            
    return -1

def read_relay_off(rise_fall: str, filename = "TEK00000.CSV", save = True):
    roff_df = pd.read_csv(filename, skiprows=15)
    t_data = roff_df["TIME"].astype(float)
    fngen_data = roff_df["CH2"].astype(float)
    ac_signal = roff_df["CH1"].astype(float)

    fig, ax = plt.subplots(figsize=(18, 9))
    relay_in_plot = ax.scatter(t_data, fngen_data, s=3, color="blue", label="Supply Voltage to Relay Input Circuit")
    ax.tick_params(axis="y", labelcolor="blue", labelsize=12)
    ax.set_xlim(t_data.min()-0.12, t_data.max()+0.01)
    ax2 = ax.twinx()
    ax2.set_ylim(-0.5, 0.7)
    ax2.tick_params(axis="y", labelcolor="green", labelsize=12)
    relay_out_plot = ax2.scatter(t_data, ac_signal, s=6, color="green")
    if (rise_fall == "Rise"):
        relay_out_plot.set_label(f"Relay AC Voltage Data (OFF$\\rightarrow$ON), with Parallel 101.2$\\Omega$ Supply Load")
    elif (rise_fall == "Fall"):
        relay_out_plot.set_label(f"Relay AC Voltage Data (ON$\\rightarrow$OFF), with Parallel 101.2$\\Omega$ Supply Load")

    # fitting procedure:
    fit_start_idx, fit_end_idx = -1, -1
    fit_idx = None
    init_cutoff_estim = 0
    if (rise_fall == "Rise"):
        fit_end_idx = find_low_cutoff_idx(ac_signal, slide_window_size=10, threshold=0.2)
        fit_idx = fit_end_idx
    elif (rise_fall == "Fall"):
        cutoff_init_estim = 10111
        fit_start_idx = find_high_cutoff_idx(ac_signal[cutoff_init_estim:].copy().reset_index(drop=True), slide_window_size=1300, threshold=-0.3)
        fit_idx = cutoff_init_estim + fit_start_idx
    print(fit_idx)
    print(t_data[fit_idx])
    if (fit_start_idx >= 0):
        fit_mask = ac_signal.index >= fit_idx
    elif (fit_end_idx >= 0):
        fit_mask = ac_signal.index <= fit_idx
    p0_off = [1, 60, -0.04, 2.1]
    popt_off, _ = curve_fit(sine, t_data[fit_mask], ac_signal[fit_mask], p0_off)
    print(popt_off)
    cutoff_line = ax2.axvline(x=t_data[fit_idx], color="red", linestyle="--", 
                              label=f"Cut-off Voltage for Relay Input")
    ac_fit = ax2.plot(t_data[fit_mask], sine(t_data[fit_mask], popt_off[0], popt_off[1], popt_off[2], popt_off[3]),
             color="lightgreen", linewidth=1.5, label="Fit of AC Leakage Signal, Relay OFF")[0]

    combined_plots = [relay_in_plot, relay_out_plot, cutoff_line, ac_fit]
    plot_labels = [p.get_label() for p in combined_plots]
    ax.legend(combined_plots, plot_labels, loc="upper left", fontsize=12, markerscale=3)

    fit_text = f"Cut-off Voltage for Relay Input: ~{fngen_data[fit_idx]}V ({t_data[fit_idx]:.3f}s)\n"
    fit_text += f"Relay AC Fit Form: $V(t) = A\\sin{{2\\pi k(t - h)}} + c$\n"
    fit_text += f" Relay OFF-State Fit:\n   $V_{{off}}(t) \\approx {popt_off[0]:.3f}\\sin{{(2\\pi {popt_off[1]:.3f}(t - {popt_off[2]:.3f}))}} + ({popt_off[3]:.3f})$V\n"
    fig.text(0.14, 0.72, fit_text, fontsize=13, verticalalignment="top")

    if (rise_fall == "Rise"):
        ax.set_title("Power Interlock Relay AC Output for Varying Driving Input, Input Rising Edge (OFF Scale)")
    elif (rise_fall == "Fall"):
        ax.set_title("Power Interlock Relay AC Output for Varying Driving Input, Input Falling Edge (OFF Scale)")
    ax.set_ylabel("Function Generator Ramp Voltage (V)", color="blue")
    ax.set_xlabel("Time (s)")
    ax2.set_ylabel("Relay AC Output Voltage (V)", color="green")
    if (save):
        if (rise_fall == "Rise"):
            plt.savefig(f"relay-off-rise_plot_{date.today()}.png")
        elif (rise_fall == "Fall"):
            plt.savefig(f"relay-off-fall_plot_{date.today()}.png")
    plt.show()

def read_fn_gen(rise_fall: str, filename = "TEK00002.CSV", save = True):
    roff_df = pd.read_csv(filename, skiprows=15)
    t_data = roff_df["TIME"].astype(float)
    fngen_data = roff_df["CH2"].astype(float)
    ac_signal = roff_df["CH1"].astype(float)

    fig, fngen_ax = plt.subplots(figsize=(18, 9))
    relay_in_plot = fngen_ax.scatter(t_data, fngen_data, s=3, color="blue", label="Supply Voltage Ramp to Relay Input Circuit")
    fngen_ax.tick_params(axis="y", labelcolor="blue", labelsize=12)
    fngen_ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
    fngen_ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    fngen_ax.set_xlim(t_data.min()-0.15, t_data.max()+0.01)
    fngen_ax.set_ylim(1.6, 3)
    ac_ax = fngen_ax.twinx()
    ac_ax.set_ylim(-180, 200)
    relay_out_plot = ac_ax.scatter(t_data, ac_signal, s=6, color="orange")
    if (rise_fall == "Rise"):
        relay_out_plot.set_label(f"Relay AC Voltage Data (OFF$\\rightarrow$ON), with Parallel 101.2$\\Omega$ Supply Load")
    elif (rise_fall == "Fall"):
        relay_out_plot.set_label(f"Relay AC Voltage Data (ON$\\rightarrow$OFF), with Parallel 101.2$\\Omega$ Supply Load")
    ac_ax.tick_params(axis="y", labelcolor="orange", labelsize=12)
    ac_ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15))

    # fitting procedure: fit a sinusoid to each side of a cutoff time:
    
    # p0 takes the format: [Ampl., Freq., Horiz. Offset, Vertical Offset]
    fit_start_idx, fit_end_idx = -1, -1
    fit_idx = None
    cutoff_init_estim = 0
    if (rise_fall == "Rise"):
        fit_start_idx = find_high_cutoff_idx(ac_signal, slide_window_size=2, threshold=2)
        fit_idx = fit_start_idx
    elif (rise_fall == "Fall"):
        cutoff_init_estim = 10271
        fit_end_idx = find_low_cutoff_idx(ac_signal[cutoff_init_estim:].copy().reset_index(drop=True), slide_window_size=550, threshold=-150)
        print(fit_end_idx)
        fit_idx = cutoff_init_estim + fit_end_idx
    p0_on = [-172, 60, 0, 0]
    if (fit_start_idx >= 0):
        fit_mask = ac_signal.index >= fit_idx
    elif (fit_end_idx >= 0):
        fit_mask = ac_signal.index <= fit_idx
    popt_on, _ = curve_fit(sine, t_data[fit_mask], ac_signal[fit_mask], p0_on)
    print(f"Fitted parameters: {popt_on}\n")
    cutoff_line = ac_ax.axvline(x=t_data[fit_idx], color="red", linestyle="--", label="Cutoff Voltage between ON/OFF States")
    fit_on_plot = ac_ax.plot(t_data[fit_mask], sine(t_data[fit_mask], popt_on[0], popt_on[1], popt_on[2], popt_on[3]),
                             color="coral", linewidth=1.5, label="Fit of Relay AC Signal in ON State")[0]
    
    combined_plots = [relay_in_plot, relay_out_plot, cutoff_line, fit_on_plot]
    plot_labels = [p.get_label() for p in combined_plots]
    fngen_ax.legend(combined_plots, plot_labels, loc="upper left", fontsize=12, markerscale=3)

    fit_text = f"Cut-off Voltage for Relay Input: ~{fngen_data[fit_idx]}V ({t_data[fit_idx]:.3f}s)\n"
    fit_text += f"Relay AC Fit Form: $V(t) = A\\sin{{2\\pi k(t - h)}} + c$\n"
    fit_text += f" Relay ON-State Fit:\n   $V_{{on}}(t) \\approx {popt_on[0]:.3f}\\sin{{(2\\pi {popt_on[1]:.3f}(t - {popt_on[2]:.3f}))}} + ({popt_on[3]:.3f})$V\n"
    fig.text(0.14, 0.72, fit_text, fontsize=13, verticalalignment="top")

    if (rise_fall == "Rise"):
        fngen_ax.set_title("Power Interlock Relay AC Output for Varying Driving Input, Input Rising Edge (ON Scale)", fontsize=16)
    elif (rise_fall == "Fall"):
        fngen_ax.set_title("Power Interlock Relay AC Output for Varying Driving Input, Input Falling Edge (ON Scale)", fontsize=16)
    fngen_ax.set_xlabel("Time (s)")
    fngen_ax.set_ylabel("Function Generator Voltage (V)", color="blue")
    ac_ax.set_ylabel("Relay AC Output Voltage (V)", color="orange")
    if save:
        if (rise_fall == "Rise"):
            plt.savefig(f"relay-on-rise_plot_{date.today()}.png")
        elif (rise_fall == "Fall"):
            plt.savefig(f"relay-on-fall_plot_{date.today()}.png")
    plt.show()

if __name__ == "__main__":
    file_type = input("Signal Datafile (Relay Off / Fn Gen): ")
    rise_fall = input("Rising or Falling Edge? (Rise / Fall): ")
    if (file_type == "Relay Off"):
        if (rise_fall == "Rise"):
            read_relay_off("Rise", filename = "ac_rise_small.CSV")
        elif (rise_fall == "Fall"):
            read_relay_off("Fall", filename = "ac_fall_small.CSV")
    elif (file_type == "Fn Gen"):
        if (rise_fall == "Rise"):
            read_fn_gen("Rise", filename = "ac_rise_large.CSV")
        elif (rise_fall == "Fall"):
            read_fn_gen("Fall", filename = "ac_fall_large_zoom.CSV")