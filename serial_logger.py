#!/usr/bin/env python3
"""
Serial data logger for gas-injury-detection.

Reads $DATA frames from the ESP32, validates XOR checksums to discard
garbage/partial frames, prints a live table to the console, and writes
every valid reading to a timestamped CSV file.

Optionally opens a real-time matplotlib window with 7 subplots
(temperature, humidity, VOC, NH3, HCHO, H2S ppm, EtOH ppm).

Setup:
    pip install pyserial matplotlib

Usage:
    python serial_logger.py [--port PORT] [--baud BAUD] [--no-plot]

Frame format emitted by the ESP32:
    $DATA,<millis_ms>,<temp_C>,<rh_pct>,<mode>,<voc>,<nh3>,<hcho>,
          <h2s_vref>,<h2s_vgas>,<h2s_ppm>,<etoh_vref>,<etoh_vgas>,<etoh_ppm>*<CRC>\r\n
    - <mode> is 'V' (raw voltage in V) or 'P' (estimated PPM).
    - Numeric fields use NAN when a sensor is faulted or disabled.
    - CRC is the 2-digit uppercase hex XOR of every byte between '$' and '*'.
    - h2s_*/etoh_* fields are always emitted: vref and vgas in V (4 dp), ppm (2 dp).

Toggle output mode on the ESP32 by sending 'v' (voltage) or 'p' (PPM) over serial.

"""

import argparse
import collections
import csv
import math
import sys
import threading
from datetime import datetime

import serial

# ── Default configuration ─────────────────────────────────────────────────────
DEFAULT_PORT = "/dev/cu.usbserial-0001"  # macOS — change to e.g. COM3 on Windows
DEFAULT_BAUD = 115200
CSV_DIR      = "."                       # directory for output CSV files

FIELDS = [
    "wall_time", "millis_ms", "temp_C", "rh_pct", "mode",
    "voc", "nh3", "hcho",
    "h2s_vref", "h2s_vgas", "h2s_ppm",
    "etoh_vref", "etoh_vgas", "etoh_ppm",
]

# ── Plot configuration ────────────────────────────────────────────────────────
PLOT_FIELDS = [
    # (data_key,   subplot_title,  y_label,  line_color)
    ("temp_C",   "Temperature",   "°C",     "tab:red"),
    ("rh_pct",   "Humidity",      "% RH",   "tab:blue"),
    ("voc",      "VOC",           "V",      "tab:orange"),
    ("nh3",      "NH3",           "V",      "tab:green"),
    ("hcho",     "HCHO",          "V",      "tab:purple"),
    ("h2s_ppm",  "H2S",           "ppm",    "tab:brown"),
    ("etoh_ppm", "EtOH",          "ppm",    "tab:pink"),
]
MAX_PLOT_POINTS = 300   # rolling window (~5 min at 1 Hz)

# ── Frame parsing ─────────────────────────────────────────────────────────────

def verify_checksum(line):
    """
    Verify the NMEA-style XOR checksum of a raw line.

    Expected format:  $<payload>*<XX>
    The checksum XX is the hex XOR of every byte in <payload>
    (i.e. between '$' and '*', exclusive).

    Returns the payload string on success, or None if the line is malformed
    or the checksum does not match — covers framing errors and garbage bits.
    """
    if not line.startswith("$"):
        return None

    star = line.rfind("*")
    if star == -1 or star + 2 > len(line) - 1:
        return None

    payload      = line[1:star]
    checksum_hex = line[star + 1 : star + 3]

    try:
        expected = int(checksum_hex, 16)
    except ValueError:
        return None

    computed = 0
    for ch in payload:
        computed ^= ord(ch)

    return payload if computed == expected else None


def parse_frame(payload):
    """
    Parse the inner payload of a validated $DATA frame into a dict.

    Returns a dict with typed values, or None if the payload is malformed.
    Fields containing 'NAN' (case-insensitive) become float('nan').
    'mode' is the single character 'V' (voltage) or 'P' (PPM).
    """
    fields = payload.split(",")
    if len(fields) != 14 or fields[0] != "DATA":
        return None

    try:
        def to_float(s):
            return float("nan") if s.upper() == "NAN" else float(s)

        return {
            "millis_ms": int(fields[1]),
            "temp_C":    to_float(fields[2]),
            "rh_pct":    to_float(fields[3]),
            "mode":      fields[4].upper(),
            "voc":       to_float(fields[5]),
            "nh3":       to_float(fields[6]),
            "hcho":      to_float(fields[7]),
            "h2s_vref":  to_float(fields[8]),
            "h2s_vgas":  to_float(fields[9]),
            "h2s_ppm":   to_float(fields[10]),
            "etoh_vref": to_float(fields[11]),
            "etoh_vgas": to_float(fields[12]),
            "etoh_ppm":  to_float(fields[13]),
        }
    except (ValueError, IndexError):
        return None


# ── Display helpers ───────────────────────────────────────────────────────────

def fmt(value, decimals=2, width=9):
    """Format a float for the live table; show FAULT for NaN."""
    if math.isnan(value):
        return "FAULT".rjust(width)
    return f"{value:.{decimals}f}".rjust(width)


def print_header():
    print(
        f"\n{'Wall time':26}  "
        f"{'Temp(C)':>8}  "
        f"{'RH(%)':>7}  "
        f"{'Mode':>4}  "
        f"{'VOC':>9}  "
        f"{'NH3':>9}  "
        f"{'HCHO':>9}  "
        f"{'H2S_Vref':>9}  "
        f"{'H2S_Vgas':>9}  "
        f"{'H2S_ppm':>8}  "
        f"{'EtOH_Vref':>10}  "
        f"{'EtOH_Vgas':>10}  "
        f"{'EtOH_ppm':>9}"
    )
    print("-" * 140)


def print_row(wall_time, rec):
    mode = rec["mode"]
    unit_decimals = 3 if mode == "V" else 2
    print(
        f"{wall_time:26}  "
        f"{fmt(rec['temp_C'], 1, 8)}  "
        f"{fmt(rec['rh_pct'], 1, 7)}  "
        f"{'[' + mode + ']':>4}  "
        f"{fmt(rec['voc'],       unit_decimals, 9)}  "
        f"{fmt(rec['nh3'],       unit_decimals, 9)}  "
        f"{fmt(rec['hcho'],      unit_decimals, 9)}  "
        f"{fmt(rec['h2s_vref'],  4, 9)}  "
        f"{fmt(rec['h2s_vgas'],  4, 9)}  "
        f"{fmt(rec['h2s_ppm'],   2, 8)}  "
        f"{fmt(rec['etoh_vref'], 4, 10)}  "
        f"{fmt(rec['etoh_vgas'], 4, 10)}  "
        f"{fmt(rec['etoh_ppm'],  2, 9)}"
    )


# ── Live plotter ──────────────────────────────────────────────────────────────

class LivePlotter:
    """
    Maintains a rolling deque of sensor readings and drives a matplotlib
    FuncAnimation window.  Thread-safe: call push() from any thread.
    """

    def __init__(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.gridspec as gridspec

        self._lock  = threading.Lock()
        self._t     = collections.deque(maxlen=MAX_PLOT_POINTS)   # elapsed seconds
        self._data  = {key: collections.deque(maxlen=MAX_PLOT_POINTS)
                       for key, *_ in PLOT_FIELDS}
        self._t0_ms = None   # millis_ms of the first frame

        # ── Build figure ──────────────────────────────────────────────────────
        self._fig = plt.figure(figsize=(13, 11))
        self._fig.suptitle("Gas Sensor — Live Readings", fontsize=13, fontweight="bold")

        # 4×2 grid; bottom row spans both columns for EtOH
        gs = gridspec.GridSpec(4, 2, figure=self._fig, hspace=0.55, wspace=0.35)

        subplot_specs = [
            gs[0, 0], gs[0, 1],   # temp, rh
            gs[1, 0], gs[1, 1],   # voc, nh3
            gs[2, 0], gs[2, 1],   # hcho, h2s_ppm
            gs[3, :],             # etoh_ppm — full width
        ]

        self._lines = {}
        for spec, (key, title, ylabel, color) in zip(subplot_specs, PLOT_FIELDS):
            ax = self._fig.add_subplot(spec)
            ax.set_title(title, fontsize=9, pad=3)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel("elapsed (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, linestyle="--")
            line, = ax.plot([], [], color=color, linewidth=1.3)
            self._lines[key] = (ax, line)

        self._ani = animation.FuncAnimation(
            self._fig,
            self._animate,
            interval=1000,       # refresh every second
            blit=False,
            cache_frame_data=False,
        )

    def push(self, record):
        """Called from the serial reader thread with a parsed record dict."""
        with self._lock:
            if self._t0_ms is None:
                self._t0_ms = record["millis_ms"]
            elapsed = (record["millis_ms"] - self._t0_ms) / 1000.0
            self._t.append(elapsed)
            for key, *_ in PLOT_FIELDS:
                self._data[key].append(record.get(key, float("nan")))

    def _animate(self, _frame):
        with self._lock:
            t    = list(self._t)
            snap = {k: list(v) for k, v in self._data.items()}

        if len(t) < 2:
            return

        for key, (ax, line) in self._lines.items():
            line.set_data(t, snap[key])
            ax.relim()
            ax.autoscale_view()

    def save(self, path):
        """Save the current figure to *path* (format inferred from extension)."""
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot : saved → {path}")

    def show(self):
        import matplotlib.pyplot as plt
        try:
            plt.show()
        except KeyboardInterrupt:
            pass   # Ctrl-C while the window is open — handled cleanly below


# ── Serial reader (runs in background thread when plotting) ───────────────────

def _read_serial(ser, writer, csv_file, plotter, stop_event, counters):
    while not stop_event.is_set():
        try:
            raw = ser.readline()
        except serial.SerialException as exc:
            # Suppress "Bad file descriptor" that fires when the port is closed
            # on shutdown — it is expected and not a real error.
            if not stop_event.is_set():
                print(f"\nSerial error: {exc}", file=sys.stderr)
            break

        try:
            line = raw.decode("ascii", errors="replace").strip()
        except Exception:
            continue

        if not line:
            continue

        if not line.startswith("$"):
            print(f"  {line}")
            continue

        payload = verify_checksum(line)
        if payload is None:
            counters["bad"] += 1
            print(f"  [SKIP] bad checksum (total: {counters['bad']}): {line!r}")
            continue

        counters["bad"] = 0

        record = parse_frame(payload)
        if record is None:
            print(f"  [SKIP] unrecognised frame: {payload!r}")
            continue

        wall_time = datetime.now().isoformat(timespec="milliseconds")
        writer.writerow({"wall_time": wall_time, **record})
        csv_file.flush()

        counters["good"] += 1
        print_row(wall_time, record)

        if plotter is not None:
            plotter.push(record)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gas sensor serial logger")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable the real-time graph window")
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path  = f"{CSV_DIR}/readings_{timestamp}.csv"
    plot_path = f"{CSV_DIR}/readings_{timestamp}.png"

    print(f"Port : {args.port}  Baud: {args.baud}")
    print(f"Log  : {csv_path}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
        ser.reset_input_buffer()
    except serial.SerialException as exc:
        print(f"Cannot open port: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Set up plotter (before starting threads so matplotlib inits on main thread)
    plotter = None
    if not args.no_plot:
        try:
            plotter = LivePlotter()
            print("Plot : real-time graph enabled (close window or Ctrl-C to stop)")
        except ImportError:
            print("Plot : matplotlib not found — run  pip install matplotlib  to enable",
                  file=sys.stderr)

    counters   = {"good": 0, "bad": 0}
    stop_event = threading.Event()

    with ser, open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        csv_file.flush()
        print_header()

        if plotter is not None:
            # Serial runs in background; matplotlib blocks main thread via show()
            t = threading.Thread(
                target=_read_serial,
                args=(ser, writer, csv_file, plotter, stop_event, counters),
                daemon=True,
            )
            t.start()
            try:
                plotter.show()   # blocks until window is closed
            finally:
                stop_event.set()
                plotter.save(plot_path)
        else:
            # No plot — run serial loop directly on main thread
            try:
                _read_serial(ser, writer, csv_file, None, stop_event, counters)
            except KeyboardInterrupt:
                pass

    print(f"\nCaptured {counters['good']} readings → {csv_path}")


if __name__ == "__main__":
    main()
