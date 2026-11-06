import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date

# Note: These datafiles were both with Tenma power supply connected
def read_relay_off(filename = "TEK00000.CSV", save = True):
    roff_df = pd.read_csv(filename, skiprows=15)
    print(roff_df.columns)
    t_data = roff_df["TIME"].astype(float)
    rigol_data = roff_df["CH2"].astype(float)
    ac_signal = roff_df["CH1"].astype(float)

    fig, ax = plt.subplots(figsize=(10,8))
    relay_in_plot = ax.scatter(t_data, rigol_data, s=0.5, color="blue", label="Supply Voltage to Relay Input Circuit")
    ax2 = ax.twinx()
    relay_out_plot = ax2.scatter(t_data, ac_signal, s=0.5, color="orange", label="AC Voltage from Relay Output (OFF), with Parallel Tenma Supply Load")
    plt.show()

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

def find_cutoff_idx(arr, slide_window_size) -> int | None:
    # Helper Function finds cutoff array index between two different sinusoidal signal forms, given a window size
    off_ampl_estimate = abs(arr[slide_window_size] - arr[0]) / 2
    cutoff_idx = None
    for idx in range(slide_window_size, len(arr) - slide_window_size):
        if (abs(arr[idx + slide_window_size] - arr[idx]) / 2 > (5.3 * off_ampl_estimate)):
            cutoff_idx = idx + (slide_window_size // 2)
            break
    return cutoff_idx

def read_fn_gen(filename = "TEK00002.CSV", save = True):
    roff_df = pd.read_csv(filename, skiprows=15)
    t_data = roff_df["TIME"].astype(float)
    fngen_data = roff_df["CH2"].astype(float)
    ac_signal = roff_df["CH1"].astype(float)

    fig, fngen_ax = plt.subplots(figsize=(16, 8))
    relay_in_plot = fngen_ax.scatter(t_data, fngen_data, s=0.5, color="purple", label="Supply Voltage Ramp to Relay Input Circuit")
    fngen_ax.tick_params(axis="y", labelcolor="purple", labelsize=12)
    fngen_ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15))
    fngen_ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    ac_ax = fngen_ax.twinx()
    relay_out_plot = ac_ax.scatter(t_data, ac_signal, s=0.5, color="orange", label="Relay AC Voltage Output (OFF), with Parallel Tenma Supply Load")
    ac_ax.tick_params(axis="y", labelcolor="orange", labelsize=12)
    ac_ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15))

    # fitting procedure: fit a sinusoid to each side of a cutoff time:
    half_period_idx = find_sinusoidal_period_idx(ac_signal)
    cutoff_idx = find_cutoff_idx(ac_signal, half_period_idx)
    cutoff_line = ac_ax.axvline(x=t_data[cutoff_idx], color="red", linestyle="--",
                                label=f"Estimated Cutoff Time Between Relay On/Off, $t_{{cutoff}}\\approx{t_data[cutoff_idx]:.3f}$s")

    # p0 takes the format: [Ampl., Freq., Horiz. Offset, Vertical Offset]
    p0_off = [26, 60, 0, 0.1, -90]
    p0_on = [172, 60, 0.02, 0]
    popt_off, _ = curve_fit(sine_lin_bg, t_data.iloc[0:cutoff_idx], ac_signal.iloc[0:cutoff_idx], p0_off)
    popt_on, _ = curve_fit(sine, t_data.iloc[cutoff_idx:], ac_signal.iloc[cutoff_idx:], p0_on)
    fit_off_plot = ac_ax.plot(t_data[0:cutoff_idx], sine_lin_bg(t_data[0:cutoff_idx], popt_off[0], popt_off[1], popt_off[2], popt_off[3], popt_off[4]),
                              color="green", label="Fit of Relay AC Signal in OFF State")[0]
    fit_on_plot = ac_ax.plot(t_data[cutoff_idx:], sine(t_data[cutoff_idx:], popt_on[0], popt_on[1], popt_on[2], popt_on[3]),
                             color="blue", label="Fit of Relay AC Signal in ON State")[0]
    
    combined_plots = [relay_in_plot, relay_out_plot, cutoff_line, fit_off_plot, fit_on_plot]
    plot_labels = [p.get_label() for p in combined_plots]
    fngen_ax.legend(combined_plots, plot_labels)

    fit_text = f"Cut-off Voltage for Relay Input: ~{fngen_data[cutoff_idx]}V ({t_data[cutoff_idx]:.3f}s)\n"
    fit_text += f"Relay AC Fit Form: $V(t) = A\\sin{{2\\pi k(t - h)}} + c$\n"
    fit_text += f" Relay OFF-State Fit:\n   $V_{{off}}(t) \\approx {popt_off[0]:.3f}\\sin{{(2\\pi {popt_off[1]:.3f}(t - {popt_off[2]:.3f}))}} + ({popt_off[3]:.3f})t + ({popt_off[4]:.3f})$V\n"
    fit_text += f" Relay ON-State Fit:\n   $V_{{on}}(t) \\approx {popt_on[0]:.3f}\\sin{{(2\\pi {popt_on[1]:.3f}(t - {popt_on[2]:.3f}))}} + ({popt_on[3]:.3f})$V\n"
    fig.text(0.14, 0.7, fit_text, fontsize=12, verticalalignment="top")

    fngen_ax.set_title("Power Interlock Relay AC Output for Varying Driving Input, ON/OFF States", fontsize=16)
    fngen_ax.set_xlabel("Time (s)")
    fngen_ax.set_ylabel("Function Generator Voltage (V)", color="purple")
    ac_ax.set_ylabel("Relay AC Output Voltage (V)", color="orange")
    if save:
        plt.savefig(f"relay-control-plot_{date.today()}.png")
    plt.show()

if __name__ == "__main__":
    file_type = input("Signal Datafile (Relay Off / Fn Gen): ")
    if (file_type == "Relay Off"):
        read_relay_off(filename = "TEK00000.CSV")
    elif (file_type == "Fn Gen"):
        read_fn_gen(filename = "TEK00001.CSV")