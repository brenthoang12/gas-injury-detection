#!/usr/bin/env python3
"""
Serial data logger for gas-injury-detection.

Reads $DATA frames from the ESP32, validates XOR checksums to discard
garbage/partial frames, prints a live table to the console, and writes
every valid reading to a timestamped CSV file.

Setup:
    pip install pyserial

Usage:
    python serial_logger.py [--port PORT] [--baud BAUD]

Frame format emitted by the ESP32:
    $DATA,<millis_ms>,<temp_C>,<rh_pct>,<mode>,<voc>,<nh3>,<hcho>*<CRC>\r\n
    - <mode> is 'V' (raw voltage in V) or 'P' (estimated PPM).
    - Numeric fields use NAN when a sensor is faulted or disabled.
    - CRC is the 2-digit uppercase hex XOR of every byte between '$' and '*'.

Toggle output mode on the ESP32 by sending 'v' (voltage) or 'p' (PPM) over serial.
"""

import argparse
import csv
import math
import sys
from datetime import datetime

import serial

# ── Default configuration ─────────────────────────────────────────────────────
DEFAULT_PORT = "/dev/cu.usbserial-0001"  # macOS — change to e.g. COM3 on Windows
DEFAULT_BAUD = 115200
CSV_DIR      = "."                       # directory for output CSV files

FIELDS = ["wall_time", "millis_ms", "temp_C", "rh_pct", "mode", "voc", "nh3", "hcho"]


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
    if len(fields) != 8 or fields[0] != "DATA":
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
        f"{'HCHO':>10}"
    )
    print("-" * 84)


def print_row(wall_time, rec):
    mode = rec["mode"]
    unit_decimals = 3 if mode == "V" else 2
    print(
        f"{wall_time:26}  "
        f"{fmt(rec['temp_C'], 1, 8)}  "
        f"{fmt(rec['rh_pct'], 1, 7)}  "
        f"{'[' + mode + ']':>4}  "
        f"{fmt(rec['voc'],  unit_decimals, 9)}  "
        f"{fmt(rec['nh3'],  unit_decimals, 9)}  "
        f"{fmt(rec['hcho'], unit_decimals, 10)}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gas sensor serial logger")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"Baud rate (default: {DEFAULT_BAUD})")
    args = parser.parse_args()

    csv_path = f"{CSV_DIR}/readings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Port : {args.port}  Baud: {args.baud}")
    print(f"Log  : {csv_path}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
        ser.reset_input_buffer()   # discard any partial line already in the buffer
    except serial.SerialException as exc:
        print(f"Cannot open port: {exc}", file=sys.stderr)
        sys.exit(1)

    bad_frames  = 0   # consecutive checksum failures — logged for diagnostics
    good_frames = 0

    with ser, open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        csv_file.flush()

        print_header()

        try:
            while True:
                # ── Read one line ───────────────────────────────────────────
                try:
                    raw = ser.readline()
                except serial.SerialException as exc:
                    print(f"\nSerial error: {exc}", file=sys.stderr)
                    break

                try:
                    line = raw.decode("ascii", errors="replace").strip()
                except Exception:
                    continue

                if not line:
                    continue

                # ── Non-data lines (FAULT / WARN / INFO from firmware) ──────
                if not line.startswith("$"):
                    print(f"  {line}")
                    continue

                # ── Validate checksum ───────────────────────────────────────
                payload = verify_checksum(line)
                if payload is None:
                    bad_frames += 1
                    print(f"  [SKIP] bad checksum (total: {bad_frames}): {line!r}")
                    continue

                bad_frames = 0  # reset streak on a clean frame

                # ── Parse fields ────────────────────────────────────────────
                record = parse_frame(payload)
                if record is None:
                    print(f"  [SKIP] unrecognised frame: {payload!r}")
                    continue

                # ── Log to CSV ──────────────────────────────────────────────
                wall_time = datetime.now().isoformat(timespec="milliseconds")
                writer.writerow({"wall_time": wall_time, **record})
                csv_file.flush()

                good_frames += 1
                print_row(wall_time, record)

        except KeyboardInterrupt:
            pass

    print(f"\nCaptured {good_frames} readings → {csv_path}")


if __name__ == "__main__":
    main()
