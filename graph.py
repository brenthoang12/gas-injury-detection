# throw error here if no file see
# deep dive on the code itself. 

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def mode_label(mode: str) -> str:
    labels = {"V": "Voltage (V)", "R": "Resistance (Ω)", "P": "Concentration (ppm)"}
    return labels.get(mode, mode)

def save_sensor(time_s, values, title, y_label, color, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    valid = values.notna()
    ax.plot(time_s[valid], values[valid], linewidth=0.8, color=color)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

def main(csv_path: str):
    df = pd.read_csv(csv_path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])

    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()

    mode = df["mode"].mode().iloc[0]
    gas_y_label = mode_label(mode)

    dt_prefix = t0.strftime("%Y%m%d-%H%M")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "graph")
    os.makedirs(out_dir, exist_ok=True)

    sensors = [
        ("temp_C", "Temperature",         "°C",        "tab:red"),
        ("rh_pct", "Humidity",            "%RH",       "tab:blue"),
        ("voc",    "VOC",                 gas_y_label, "tab:green"),
        ("nh3",    "NH3",                 gas_y_label, "tab:orange"),
        ("hcho",   "HCHO (Formaldehyde)", gas_y_label, "tab:purple"),
    ]

    for col, title, y_label, color in sensors:
        filename = f"{dt_prefix}-{col.upper()}.png"
        out_path = os.path.join(out_dir, filename)
        save_sensor(df["time_s"], df[col], title, y_label, color, out_path)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.csv" 
    main(path)
