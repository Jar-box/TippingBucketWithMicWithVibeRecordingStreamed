import socket
import struct

import csv
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_trend_line(data: list, window: int = 20) -> list:
    """Compute a simple moving average trend line."""
    if len(data) < window:
        return data
    trend = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        trend.append(sum(data[start:end]) / (end - start))
    return trend


# ======== CONFIG ========
WINDOW_SECONDS = 30
RESET_INTERVAL = 60
MIC_DB_REF = 1.0
MIC_SUPPRESSION_SECONDS = 0.35
SMOOTHING_FACTOR = 0.01

# === RAIN RATE CONFIGURATION ===
TIP_MM_PER_TIP = 0.0335  # mm per tip (from TIPPING_BUCKET_SPECS.md)
VALID_TIP_INTERVAL_MIN_S = (
    2.52  # minimum physically valid tip interval (hardware limit)
)
SATURATION_THRESHOLD_MM_HR = 47.5  # near hardware max 47.9 mm/hr

# === CALIBRATION COEFFICIENT FILE ===
CALIB_COEFF_FILE = Path("data") / "calibration_coefficients.json"

# === PERFORMANCE OPTIMIZATIONS ===
UPDATE_EVERY_N_SAMPLES = 50  # Update plot less frequently for smoother rendering
FIGURE_SIZE = (12, 7)

# === ZOOM LIMITS ===
MIN_TIME_ZOOM = 30.0
MIN_AMPLITUDE_ZOOM = 1024.0

# === VISUAL STYLE ===
TIP_LINE_COLOR = "blue"
TIP_LINE_WIDTH = 2.5
TIP_LINE_ALPHA = 0.9
TIP_LINE_STYLE = "--"
MIC_RAW_COLOR = "mediumpurple"
MIC_RAW_ALPHA = 0.75
MIC_SMOOTH_COLOR = "darkorange"
MIC_SMOOTH_ALPHA = 0.95

# === SOCKET SETTINGS ===
MCAST_GRP = "230.138.19.201"
MCAST_PORT = 5007
IS_ALL_GROUPS = True
SOCKET_TIMEOUT = 0.02  # 20ms - very short to keep main loop responsive

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.settimeout(SOCKET_TIMEOUT)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(
    socket.SOL_SOCKET, socket.SO_RCVBUF, 262144
)  # 256KB buffer - large to absorb bursts
if IS_ALL_GROUPS:
    sock.bind(("", MCAST_PORT))
else:
    sock.bind((MCAST_GRP, MCAST_PORT))
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)


# ======== OUTPUT FILE ========
now = datetime.now()
stamp = now.strftime("%Y%m%d_%H%M%S")
date_folder = now.strftime("%Y-%m-%d")
hour_folder = now.strftime("%H")

output_dir = Path("data") / date_folder / hour_folder
output_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = output_dir / f"rain_intensity_{stamp}.csv"
OUTPUT_PNG = output_dir / f"rain_intensity_{stamp}.png"
OUTPUT_HTML = output_dir / f"rain_intensity_{stamp}.html"

# ======== STATE ========
series_t = deque()
series_bucket = deque()
series_mic_raw = deque()
series_mic_filtered = deque()
series_mic_amp_raw = deque()
series_mic_amp_filtered = deque()
tip_times = deque()

archive_t = deque()
archive_bucket = deque()
archive_mic_raw = deque()
archive_mic_filtered = deque()
archive_mic_amp_raw = deque()
archive_mic_amp_filtered = deque()
archive_tip_times = deque()

# Archive deques for intensity data
archive_bucket_intensity = deque()
archive_mic_intensity = deque()
archive_fused_intensity = deque()
archive_intensity_mode = deque()
archive_saturation_flag = deque()
archive_rms = deque()
archive_tip_valid = deque()

last_reset_time = 0
tip_count_at_reset = 0
mic_amp_sum_at_reset = 0
sample_count_since_reset = 0
previous_tip_count = 0
mic_suppressed_until = 0.0
sample_counter = 0
smoothed_mic_amp = 0.0
smoothed_mic_db = 0.0
mic_amp_suppressed_raw = 0.0
final_tip_count = 0
final_first_tip_count = 0

# === INTERVAL RMS STATE ===
interval_sample_buffer = []  # Buffer to collect raw sound_amp samples between tips
interval_rms = 0.0  # RMS computed at each tip boundary

# === CALIBRATION STATE ===
calibration_pairs = []  # List of (rms, intensity_mm_hr, timestamp, quality_flag) tuples
MIN_PAIRS_FOR_FIT = 5  # Minimum valid pairs before trusting calibration

# Load previous calibration coefficients if available
calib_A = 0.01  # Default slope (rough linear approximation)
calib_B = 0.0  # Default intercept
calib_loaded = False

if CALIB_COEFF_FILE.exists():
    try:
        with open(CALIB_COEFF_FILE, "r") as cf:
            calib_data = json.load(cf)
            calib_A = calib_data.get("A", 0.01)
            calib_B = calib_data.get("B", 0.0)
            calib_loaded = True
            print(
                f"[CALIBRATION] Loaded A={calib_A:.6f}, B={calib_B:.3f} from {CALIB_COEFF_FILE}"
            )
    except Exception as e:
        print(f"[CALIBRATION] Warning: Failed to load coefficients: {e}")
else:
    print(
        f"[CALIBRATION] No existing calibration file. Using defaults: A={calib_A}, B={calib_B}"
    )


def exponential_smooth(
    current_value: float, smoothed_value: float, alpha: float = SMOOTHING_FACTOR
) -> float:
    return alpha * current_value + (1.0 - alpha) * smoothed_value


def prune_old(now_ts: float) -> None:
    while series_t and (now_ts - series_t[0]) > WINDOW_SECONDS:
        series_t.popleft()
        series_bucket.popleft()
        series_mic_raw.popleft()
        series_mic_filtered.popleft()
        series_mic_amp_raw.popleft()
        series_mic_amp_filtered.popleft()
    while tip_times and (now_ts - tip_times[0]) > WINDOW_SECONDS:
        tip_times.popleft()


print("OPTIMIZED SINGLE-THREADED version")
print(f"Socket timeout: {SOCKET_TIMEOUT*1000:.1f}ms (very short, tight loop)")
print(f"Update frequency: Every {UPDATE_EVERY_N_SAMPLES} samples")
print()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "elapsed_s",
            "unix_ts",
            "arduino_ms",
            "sound_amp",
            "tip_count",
            "last_tip_dt_ms",
            "reed_state",
            "bucket_rate_mm_hr",
            "mic_db_unfiltered",
            "mic_db_filtered",
            "tip_interval_s",
            "tip_valid",
            "bucket_intensity_mm_hr",
            "saturation_flag",
            "interval_rms",
            "calib_pair_used",
            "pair_quality_reason",
            "intensity_mic_mm_hr",
            "intensity_fused_mm_hr",
            "intensity_mode",
        ]
    )

    plt.ion()
    fig, ax_amp_left = plt.subplots(1, 1, figsize=(12, 6))

    (line_mic_amp_raw,) = ax_amp_left.plot(
        [],
        [],
        label="Mic amplitude RAW",
        color=MIC_RAW_COLOR,
        alpha=MIC_RAW_ALPHA,
        linewidth=1.5,
    )
    (line_mic_amp_smooth,) = ax_amp_left.plot(
        [],
        [],
        label="Mic amplitude SMOOTHED",
        color=MIC_SMOOTH_COLOR,
        alpha=MIC_SMOOTH_ALPHA,
        linewidth=2,
    )

    # Add a dummy line for tipping events in the legend
    ax_amp_left.plot(
        [],
        [],
        label="Tipping Events",
        color=TIP_LINE_COLOR,
        alpha=TIP_LINE_ALPHA,
        linewidth=TIP_LINE_WIDTH,
        linestyle=TIP_LINE_STYLE,
    )

    ax_amp_left.set_title(
        "Mic Amplitude: RAW vs SMOOTHED", fontsize=14, fontweight="bold"
    )
    ax_amp_left.set_xlabel("Duration (s)", fontsize=12)
    ax_amp_left.set_ylabel("Mic amplitude", fontsize=12)
    ax_amp_left.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax_amp_left.grid(True, alpha=0.3)

    fig.tight_layout()

    print("Reading from Arduino/RPi... Press Ctrl+C to stop")

    start_ts = None
    last_reset_time = None
    last_diagnostics = None
    base_unix_ts = None
    first_arduino_ms = None
    first_tip_count = None

    tip_lines_amp_left = []
    is_closed = False
    last_tip_count_drawn = 0
    last_gui_update = time.time()

    def on_close(event):
        global is_closed
        is_closed = True

    cid = fig.canvas.mpl_connect("close_event", on_close)

    try:
        while not is_closed:
            # Read multiple packets per iteration to keep up with fast data
            packets_read_this_iter = 0
            max_packets_per_iter = 5  # Read up to 5 packets before rendering

            while packets_read_this_iter < max_packets_per_iter:
                try:
                    line = sock.recv(10240).decode("utf-8")
                    if not line:
                        break
                except socket.timeout:
                    # No data available, that's fine
                    break

                if line.startswith("ts_ms"):
                    continue

                parts = line.split(",")
                if len(parts) < 5:
                    continue

                try:
                    arduino_ms = int(parts[0])
                    sound_amp = int(parts[1])
                    tip_count = int(parts[2])
                    last_tip_dt_ms = int(parts[3])
                    reed_state = int(parts[4])
                except ValueError:
                    continue

                if first_arduino_ms is None:
                    first_arduino_ms = arduino_ms
                    base_unix_ts = time.time() - (arduino_ms / 1000.0)
                    start_ts = base_unix_ts + (first_arduino_ms / 1000.0)
                    last_reset_time = start_ts
                    last_diagnostics = start_ts
                    first_tip_count = tip_count
                    final_first_tip_count = tip_count

                # Update final counts for HTML generation
                final_tip_count = tip_count

                # Skip if not initialized (type guard)
                if (
                    start_ts is None
                    or base_unix_ts is None
                    or first_arduino_ms is None
                    or last_reset_time is None
                    or last_diagnostics is None
                    or first_tip_count is None
                ):
                    continue

                now_ts = base_unix_ts + (arduino_ms / 1000.0)
                elapsed_s = (arduino_ms - first_arduino_ms) / 1000.0

                if now_ts - last_reset_time >= RESET_INTERVAL:
                    last_reset_time = now_ts
                    tip_count_at_reset = tip_count
                    mic_amp_sum_at_reset = 0
                    sample_count_since_reset = 0
                    print(
                        f"[RESET] Tip counter reset at {elapsed_s:.1f}s, tips={tip_count}"
                    )

                tips_since_reset = tip_count - tip_count_at_reset
                sample_count_since_reset += 1
                time_since_reset = now_ts - last_reset_time
                tips_since_start = tip_count - first_tip_count

                if elapsed_s % 5 < 0.5:
                    print(
                        f"[{elapsed_s:.1f}s] amp={sound_amp}, tips={tip_count}, tips_since_start={tips_since_start}, delta_tips={tips_since_reset}, dt={last_tip_dt_ms}ms"
                    )

                # Calculate bucket intensity from tip interval
                tip_interval_s = 0.0
                tip_valid = False
                bucket_intensity_mm_hr = 0.0
                saturation_flag = False
                calib_pair_used = False
                pair_quality_reason = ""

                if tip_count > previous_tip_count:
                    # New tip detected

                    # Compute interval RMS from samples collected since last tip
                    if len(interval_sample_buffer) > 0:
                        # RMS = sqrt(mean(x^2))
                        try:
                            sum_squares = sum(s * s for s in interval_sample_buffer)
                            mean_square = sum_squares / len(interval_sample_buffer)
                            interval_rms = (
                                math.sqrt(mean_square) if mean_square >= 0 else 0.0
                            )
                            # Clamp to reasonable range (0-1023 ADC range)
                            interval_rms = max(0.0, min(interval_rms, 1023.0))
                            # Check for NaN/inf
                            if not math.isfinite(interval_rms):
                                interval_rms = 0.0
                        except (ValueError, ZeroDivisionError):
                            interval_rms = 0.0
                    else:
                        interval_rms = 0.0

                    if last_tip_dt_ms > 0:
                        tip_interval_s = last_tip_dt_ms / 1000.0

                        # Validate tip interval against physical minimum
                        if tip_interval_s >= VALID_TIP_INTERVAL_MIN_S:
                            tip_valid = True
                            # Calculate intensity: (mm/tip) / (s/tip) * (3600 s/hr)
                            try:
                                bucket_intensity_mm_hr = (
                                    TIP_MM_PER_TIP / tip_interval_s
                                ) * 3600.0
                                # Clamp to reasonable range and check for NaN/inf
                                bucket_intensity_mm_hr = max(
                                    0.0, bucket_intensity_mm_hr
                                )
                                if not math.isfinite(bucket_intensity_mm_hr):
                                    bucket_intensity_mm_hr = 0.0
                                    tip_valid = False
                            except (ValueError, ZeroDivisionError):
                                bucket_intensity_mm_hr = 0.0
                                tip_valid = False

                            # Check for saturation
                            if bucket_intensity_mm_hr >= SATURATION_THRESHOLD_MM_HR:
                                saturation_flag = True

                            # Store calibration pair if valid and we have samples
                            if interval_rms > 0 and len(interval_sample_buffer) >= 5:
                                calibration_pairs.append(
                                    (
                                        interval_rms,
                                        bucket_intensity_mm_hr,
                                        now_ts,
                                        "valid",
                                    )
                                )
                                calib_pair_used = True
                                pair_quality_reason = "valid"
                            elif len(interval_sample_buffer) < 5:
                                pair_quality_reason = "too_few_samples"
                            else:
                                pair_quality_reason = "zero_rms"
                        else:
                            # Interval too short - likely bounce or physically impossible
                            tip_valid = False
                            bucket_intensity_mm_hr = 0.0
                            pair_quality_reason = "interval_too_short"
                    else:
                        pair_quality_reason = "no_last_tip_dt"

                    # Clear interval buffer for next tip period
                    interval_sample_buffer = []

                    previous_tip_count = tip_count
                    mic_suppressed_until = now_ts + MIC_SUPPRESSION_SECONDS
                    tip_times.append(now_ts)
                    archive_tip_times.append(now_ts)

                # Collect current sample in interval buffer for next RMS calculation
                interval_sample_buffer.append(sound_amp)

                # Legacy bucket_rate field (keep for backward compatibility)
                bucket_rate = bucket_intensity_mm_hr if tip_valid else 0.0

                # === PHASE 3: INTENSITY OUTPUT CHANNELS ===
                # Compute mic-based intensity estimate (using interval RMS from most recent tip)
                intensity_mic_mm_hr = 0.0
                if interval_rms > 0:
                    # Apply calibration model: I = A * RMS + B
                    try:
                        intensity_mic_mm_hr = calib_A * interval_rms + calib_B
                        # Clamp to non-negative and check for NaN/inf
                        intensity_mic_mm_hr = max(0.0, intensity_mic_mm_hr)
                        if not math.isfinite(intensity_mic_mm_hr):
                            intensity_mic_mm_hr = 0.0
                    except (ValueError, TypeError):
                        intensity_mic_mm_hr = 0.0

                # Determine output mode (staged rollout: mostly BUCKET for now)
                intensity_mode = "UNCALIBRATED"
                intensity_fused_mm_hr = 0.0

                if not calib_loaded and len(calibration_pairs) < MIN_PAIRS_FOR_FIT:
                    # Not yet calibrated
                    intensity_mode = "UNCALIBRATED"
                    intensity_fused_mm_hr = bucket_intensity_mm_hr
                elif saturation_flag and intensity_mic_mm_hr > 0:
                    # Bucket saturated, use mic if available
                    intensity_mode = "MIC"
                    intensity_fused_mm_hr = intensity_mic_mm_hr
                elif tip_valid:
                    # Normal range, trust bucket
                    intensity_mode = "BUCKET"
                    intensity_fused_mm_hr = bucket_intensity_mm_hr
                elif intensity_mic_mm_hr > 0:
                    # No valid tip but mic has data (light rain case)
                    intensity_mode = "MIC"
                    intensity_fused_mm_hr = intensity_mic_mm_hr
                else:
                    # No data
                    intensity_mode = "UNCALIBRATED"
                    intensity_fused_mm_hr = 0.0

                mic_rate_unfiltered = 20.0 * math.log10(max(sound_amp, 1) / MIC_DB_REF)
                mic_amp_unfiltered = sound_amp

                if sample_counter == 0:
                    smoothed_mic_amp = mic_amp_unfiltered
                    smoothed_mic_db = mic_rate_unfiltered
                else:
                    smoothed_mic_amp = exponential_smooth(
                        mic_amp_unfiltered, smoothed_mic_amp, SMOOTHING_FACTOR
                    )
                    smoothed_mic_db = exponential_smooth(
                        mic_rate_unfiltered, smoothed_mic_db, SMOOTHING_FACTOR
                    )

                if now_ts < mic_suppressed_until:
                    mic_amp_suppressed_raw = smoothed_mic_amp
                else:
                    mic_amp_suppressed_raw = mic_amp_unfiltered

                # Keep filtered channels as pure EMA outputs, independent of tip events.
                mic_amp_filtered = smoothed_mic_amp
                mic_rate_filtered = smoothed_mic_db

                series_t.append(now_ts)
                series_bucket.append(bucket_rate)
                series_mic_raw.append(mic_rate_unfiltered)
                series_mic_filtered.append(mic_rate_filtered)
                series_mic_amp_raw.append(mic_amp_suppressed_raw)
                series_mic_amp_filtered.append(mic_amp_filtered)
                prune_old(now_ts)

                archive_t.append(now_ts)
                archive_bucket.append(bucket_rate)
                archive_mic_raw.append(mic_rate_unfiltered)
                archive_mic_filtered.append(mic_rate_filtered)
                archive_mic_amp_raw.append(mic_amp_suppressed_raw)
                archive_mic_amp_filtered.append(mic_amp_filtered)

                # Archive intensity data
                archive_bucket_intensity.append(bucket_intensity_mm_hr)
                archive_mic_intensity.append(intensity_mic_mm_hr)
                archive_fused_intensity.append(intensity_fused_mm_hr)
                archive_intensity_mode.append(intensity_mode)
                archive_saturation_flag.append(saturation_flag)
                archive_rms.append(interval_rms)
                archive_tip_valid.append(tip_valid)

                writer.writerow(
                    [
                        elapsed_s,
                        now_ts,
                        arduino_ms,
                        sound_amp,
                        tip_count,
                        last_tip_dt_ms,
                        reed_state,
                        bucket_rate,
                        mic_rate_unfiltered,
                        mic_rate_filtered,
                        tip_interval_s,
                        tip_valid,
                        bucket_intensity_mm_hr,
                        saturation_flag,
                        interval_rms,
                        calib_pair_used,
                        pair_quality_reason,
                        intensity_mic_mm_hr,
                        intensity_fused_mm_hr,
                        intensity_mode,
                    ]
                )
                if sample_counter % 100 == 0:
                    f.flush()

                sample_counter += 1
                packets_read_this_iter += 1

                # Diagnostics every 5 seconds
                if now_ts - last_diagnostics >= 5.0:
                    elapsed_total = now_ts - start_ts
                    samples_per_sec = (
                        sample_counter / elapsed_total if elapsed_total > 0 else 0
                    )
                    print(f"[{elapsed_total:.1f}s] {samples_per_sec:.1f} samples/sec")
                    last_diagnostics = now_ts

            # Update plot every N samples (not every packet)
            if sample_counter % UPDATE_EVERY_N_SAMPLES != 0:
                # Even when not updating plot, process GUI events periodically to keep responsive
                current_time = time.time()
                if current_time - last_gui_update > 0.1:  # Every 100ms
                    fig.canvas.flush_events()
                    last_gui_update = current_time
                continue

            if not series_t:
                continue

            t_rel = [t - start_ts for t in series_t]

            try:
                line_mic_amp_raw.set_data(t_rel, list(series_mic_amp_raw))
                line_mic_amp_smooth.set_data(t_rel, list(series_mic_amp_filtered))

                while last_tip_count_drawn < len(tip_times):
                    tip_ts = tip_times[last_tip_count_drawn]
                    x = tip_ts - start_ts
                    if x >= 0:
                        tip_lines_amp_left.append(
                            ax_amp_left.axvline(
                                x=x,
                                color=TIP_LINE_COLOR,
                                alpha=TIP_LINE_ALPHA,
                                linewidth=TIP_LINE_WIDTH,
                                linestyle=TIP_LINE_STYLE,
                            )
                        )
                    last_tip_count_drawn += 1

                if t_rel:
                    x_max = t_rel[-1]
                    x_min = max(0, x_max - WINDOW_SECONDS)
                    if x_max - x_min < MIN_TIME_ZOOM:
                        x_max = x_min + MIN_TIME_ZOOM

                    old_tip_lines = [
                        ln for ln in tip_lines_amp_left if ln.get_xdata()[0] < x_min
                    ]
                    for ln in old_tip_lines:
                        ln.remove()
                    tip_lines_amp_left = [
                        ln for ln in tip_lines_amp_left if ln.get_xdata()[0] >= x_min
                    ]

                    recent_raw = (
                        [v for v in series_mic_amp_raw] if series_mic_amp_raw else [0]
                    )
                    recent_filtered = (
                        [v for v in series_mic_amp_filtered]
                        if series_mic_amp_filtered
                        else [0]
                    )
                    if recent_raw or recent_filtered:
                        y_min = min(min(recent_raw), min(recent_filtered))
                        y_max = max(max(recent_raw), max(recent_filtered))
                    else:
                        y_min, y_max = 0, 1024

                    if y_max - y_min < MIN_AMPLITUDE_ZOOM:
                        y_center = (y_max + y_min) / 2
                        y_min = y_center - MIN_AMPLITUDE_ZOOM / 2
                        y_max = y_center + MIN_AMPLITUDE_ZOOM / 2
                        if y_min < 0:
                            y_min = 0
                            y_max = MIN_AMPLITUDE_ZOOM

                    ax_amp_left.set_ylim(y_min, y_max)
                    ax_amp_left.set_xlim(x_min, x_max)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception as e:
                # Silently continue on rendering errors
                pass

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        plt.ioff()
        is_closed = True

        # Disconnect the close event handler before closing
        try:
            fig.canvas.mpl_disconnect(cid)
        except:
            pass

        # Close the figure gracefully
        try:
            plt.close(fig)
        except:
            pass

        if archive_t:
            print("Generating final plot...")
            fig_archive, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            t0_arch = archive_t[0]
            t_rel_arch = [t - t0_arch for t in archive_t]

            t_real_arch = [datetime.fromtimestamp(t) for t in archive_t]
            t_numeric_arch = mdates.date2num(t_real_arch)

            # ===== SUBPLOT 1: Rainfall Intensity (mm/hr) =====
            ax_intensity = axes[0]

            # Plot intensity traces
            ax_intensity.plot(
                t_numeric_arch,
                list(archive_fused_intensity),
                label="Fused Intensity",
                color="#2E86AB",
                linewidth=2.5,
            )
            ax_intensity.plot(
                t_numeric_arch,
                list(archive_bucket_intensity),
                label="Bucket Intensity",
                color="#06A77D",
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
            )
            ax_intensity.plot(
                t_numeric_arch,
                list(archive_mic_intensity),
                label="Mic Intensity",
                color="#D62246",
                linewidth=1.5,
                linestyle=":",
                alpha=0.8,
            )

            # Add saturation threshold line
            ax_intensity.axhline(
                y=SATURATION_THRESHOLD_MM_HR,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"Saturation Threshold ({SATURATION_THRESHOLD_MM_HR} mm/hr)",
            )

            # Color background by mode
            for i in range(len(t_numeric_arch) - 1):
                mode = archive_intensity_mode[i]
                if mode == "BUCKET":
                    color = "#90EE90"  # light green
                elif mode == "MIC":
                    color = "#ADD8E6"  # light blue
                else:  # UNCALIBRATED
                    color = "#D3D3D3"  # light gray
                ax_intensity.axvspan(
                    t_numeric_arch[i],
                    t_numeric_arch[i + 1],
                    facecolor=color,
                    alpha=0.15,
                    linewidth=0,
                )

            ax_intensity.set_title(
                "Rainfall Intensity - Full Session",
                fontsize=14,
                fontweight="bold",
            )
            ax_intensity.set_ylabel("Intensity (mm/hr)", fontsize=12)
            ax_intensity.legend(loc="upper right", framealpha=0.95, fontsize=10)
            ax_intensity.grid(True, alpha=0.3)

            # ===== SUBPLOT 2: RMS & Mic Amplitude =====
            ax_rms = axes[1]
            ax_amp = ax_rms.twinx()

            # Plot RMS on left y-axis
            line1 = ax_rms.plot(
                t_numeric_arch,
                list(archive_rms),
                label="Interval RMS",
                color="#FF6B35",
                linewidth=2,
            )

            # Plot amplitude on right y-axis
            line2 = ax_amp.plot(
                t_numeric_arch,
                list(archive_mic_amp_raw),
                label="Mic amplitude RAW",
                color=MIC_RAW_COLOR,
                alpha=MIC_RAW_ALPHA,
                linewidth=1.5,
            )
            line3 = ax_amp.plot(
                t_numeric_arch,
                list(archive_mic_amp_filtered),
                label="Mic amplitude SMOOTHED",
                color=MIC_SMOOTH_COLOR,
                alpha=MIC_SMOOTH_ALPHA,
                linewidth=2,
            )

            ax_rms.set_ylabel("Interval RMS", fontsize=12, color="#FF6B35")
            ax_amp.set_ylabel("Mic amplitude", fontsize=12)
            ax_rms.tick_params(axis="y", labelcolor="#FF6B35")

            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax_rms.legend(
                lines, labels, loc="upper right", framealpha=0.95, fontsize=10
            )
            ax_rms.grid(True, alpha=0.3)

            # ===== SUBPLOT 3: Operational Mode & Status =====
            ax_mode = axes[2]

            # Convert mode strings to numeric values for plotting
            mode_numeric = []
            for mode in archive_intensity_mode:
                if mode == "UNCALIBRATED":
                    mode_numeric.append(0)
                elif mode == "BUCKET":
                    mode_numeric.append(1)
                elif mode == "MIC":
                    mode_numeric.append(2)
                else:
                    mode_numeric.append(0)

            ax_mode.plot(
                t_numeric_arch,
                mode_numeric,
                label="Intensity Mode",
                color="#4A4A4A",
                linewidth=2,
                drawstyle="steps-post",
            )
            ax_mode.set_yticks([0, 1, 2])
            ax_mode.set_yticklabels(["UNCALIBRATED", "BUCKET", "MIC"])

            # Overlay saturation flags
            sat_times = [
                t_numeric_arch[i]
                for i, flag in enumerate(archive_saturation_flag)
                if flag
            ]
            sat_values = [2.2 for _ in sat_times]
            if sat_times:
                ax_mode.scatter(
                    sat_times,
                    sat_values,
                    color="red",
                    marker="^",
                    s=50,
                    alpha=0.7,
                    label="Saturation",
                    zorder=5,
                )

            # Overlay tip_valid flags
            valid_times = [
                t_numeric_arch[i] for i, flag in enumerate(archive_tip_valid) if flag
            ]
            valid_values = [2.4 for _ in valid_times]
            if valid_times:
                ax_mode.scatter(
                    valid_times,
                    valid_values,
                    color="blue",
                    marker="o",
                    s=30,
                    alpha=0.5,
                    label="Valid Tip",
                    zorder=5,
                )

            ax_mode.set_ylabel("Mode / Status", fontsize=12)
            ax_mode.set_xlabel("Time", fontsize=12)
            ax_mode.legend(loc="upper right", framealpha=0.95, fontsize=10)
            ax_mode.grid(True, alpha=0.3)
            ax_mode.set_ylim(-0.5, 2.8)

            # ===== Add tipping event lines to all subplots =====
            for tip_ts in archive_tip_times:
                tip_numeric = float(mdates.date2num(datetime.fromtimestamp(tip_ts)))
                for ax in axes:
                    ax.axvline(
                        x=tip_numeric,
                        color=TIP_LINE_COLOR,
                        alpha=TIP_LINE_ALPHA,
                        linewidth=TIP_LINE_WIDTH,
                        linestyle=TIP_LINE_STYLE,
                    )

            # ===== Format x-axis for all subplots =====
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())
            fig_archive.autofmt_xdate()

            fig_archive.tight_layout()
            fig_archive.savefig(OUTPUT_PNG, dpi=100, bbox_inches="tight")
            plt.close(fig_archive)

            print("Generating interactive HTML plot...")
            print(f"DEBUG: archive_t length = {len(archive_t)}")
            print(f"DEBUG: archive_mic_amp_raw length = {len(archive_mic_amp_raw)}")
            print(
                f"DEBUG: archive_mic_amp_filtered length = {len(archive_mic_amp_filtered)}"
            )
            if archive_mic_amp_raw:
                print(f"DEBUG: First 5 raw values: {list(archive_mic_amp_raw)[:5]}")
                print(f"DEBUG: Last 5 raw values: {list(archive_mic_amp_raw)[-5:]}")
                print(
                    f"DEBUG: Min raw: {min(archive_mic_amp_raw)}, Max raw: {max(archive_mic_amp_raw)}"
                )

            total_tips = len(archive_tip_times)
            fig_interactive = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    "Rainfall Intensity (mm/hr)",
                    "RMS & Mic Amplitude",
                    "Operational Mode & Status",
                ),
                shared_xaxes=True,
                vertical_spacing=0.08,
                specs=[
                    [{"secondary_y": False}],
                    [{"secondary_y": True}],
                    [{"secondary_y": False}],
                ],
            )

            # ===== ROW 1: Rainfall Intensity =====
            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_fused_intensity),
                    name="Fused Intensity",
                    mode="lines",
                    line=dict(color="#2E86AB", width=2.5),
                    hovertemplate="<b>Fused</b><br>Time: %{x|%H:%M:%S}<br>Intensity: %{y:.2f} mm/hr<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_bucket_intensity),
                    name="Bucket Intensity",
                    mode="lines",
                    line=dict(color="#06A77D", width=1.5, dash="dash"),
                    opacity=0.8,
                    hovertemplate="<b>Bucket</b><br>Time: %{x|%H:%M:%S}<br>Intensity: %{y:.2f} mm/hr<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_mic_intensity),
                    name="Mic Intensity",
                    mode="lines",
                    line=dict(color="#D62246", width=1.5, dash="dot"),
                    opacity=0.8,
                    hovertemplate="<b>Mic</b><br>Time: %{x|%H:%M:%S}<br>Intensity: %{y:.2f} mm/hr<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Add saturation threshold line
            fig_interactive.add_hline(
                y=SATURATION_THRESHOLD_MM_HR,
                line_dash="dash",
                line_color="orange",
                opacity=0.7,
                annotation_text=f"Saturation ({SATURATION_THRESHOLD_MM_HR} mm/hr)",
                annotation_position="right",
                row=1,
                col=1,
            )

            # Add mode background coloring
            i = 0
            while i < len(t_real_arch):
                start_idx = i
                current_mode = archive_intensity_mode[i]

                # Find end of this mode region
                while (
                    i < len(t_real_arch) and archive_intensity_mode[i] == current_mode
                ):
                    i += 1
                end_idx = i - 1

                # Set color based on mode
                if current_mode == "BUCKET":
                    color = "rgba(144, 238, 144, 0.15)"  # light green
                elif current_mode == "MIC":
                    color = "rgba(173, 216, 230, 0.15)"  # light blue
                else:  # UNCALIBRATED
                    color = "rgba(211, 211, 211, 0.15)"  # light gray

                fig_interactive.add_vrect(
                    x0=t_real_arch[start_idx],
                    x1=t_real_arch[end_idx],
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    row=1,
                    col=1,
                )

            # ===== ROW 2: RMS & Mic Amplitude =====
            # RMS on primary y-axis
            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_rms),
                    name="Interval RMS",
                    mode="lines",
                    line=dict(color="#FF6B35", width=2),
                    hovertemplate="<b>RMS</b><br>Time: %{x|%H:%M:%S}<br>RMS: %{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )

            # Mic amplitude on secondary y-axis
            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_mic_amp_raw),
                    name="Mic amplitude RAW",
                    mode="lines",
                    line=dict(color=MIC_RAW_COLOR, width=1.5),
                    opacity=MIC_RAW_ALPHA,
                    hovertemplate="<b>RAW</b><br>Time: %{x|%H:%M:%S}<br>Amplitude: %{y:.0f}<extra></extra>",
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_mic_amp_filtered),
                    name="Mic amplitude SMOOTHED",
                    mode="lines",
                    line=dict(color=MIC_SMOOTH_COLOR, width=2),
                    opacity=MIC_SMOOTH_ALPHA,
                    hovertemplate="<b>SMOOTHED</b><br>Time: %{x|%H:%M:%S}<br>Amplitude: %{y:.0f}<extra></extra>",
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

            # ===== ROW 3: Operational Mode & Status =====
            # Convert mode strings to numeric for plotting
            mode_numeric = []
            for mode in archive_intensity_mode:
                if mode == "UNCALIBRATED":
                    mode_numeric.append(0)
                elif mode == "BUCKET":
                    mode_numeric.append(1)
                elif mode == "MIC":
                    mode_numeric.append(2)
                else:
                    mode_numeric.append(0)

            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=mode_numeric,
                    name="Intensity Mode",
                    mode="lines",
                    line=dict(color="#4A4A4A", width=2, shape="hv"),
                    hovertemplate="<b>Mode</b><br>Time: %{x|%H:%M:%S}<br>Mode: %{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )

            # Add saturation flags
            sat_times = [
                t_real_arch[i] for i, flag in enumerate(archive_saturation_flag) if flag
            ]
            sat_values = [2.2 for _ in sat_times]
            if sat_times:
                fig_interactive.add_trace(
                    go.Scatter(
                        x=sat_times,
                        y=sat_values,
                        name="Saturation",
                        mode="markers",
                        marker=dict(color="red", symbol="triangle-up", size=8),
                        opacity=0.7,
                        hovertemplate="<b>Saturation Event</b><br>Time: %{x|%H:%M:%S}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

            # Add valid tip flags
            valid_times = [
                t_real_arch[i] for i, flag in enumerate(archive_tip_valid) if flag
            ]
            valid_values = [2.4 for _ in valid_times]
            if valid_times:
                fig_interactive.add_trace(
                    go.Scatter(
                        x=valid_times,
                        y=valid_values,
                        name="Valid Tip",
                        mode="markers",
                        marker=dict(color="blue", symbol="circle", size=6),
                        opacity=0.5,
                        hovertemplate="<b>Valid Tip</b><br>Time: %{x|%H:%M:%S}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

            # ===== Add tipping event lines to all rows =====
            for tip_ts in archive_tip_times:
                tip_datetime = datetime.fromtimestamp(tip_ts)
                for row_num in [1, 2, 3]:
                    fig_interactive.add_vline(
                        x=tip_datetime,
                        line_dash="dash",
                        line_color=TIP_LINE_COLOR,
                        opacity=TIP_LINE_ALPHA,
                        line_width=TIP_LINE_WIDTH,
                        row=row_num,
                        col=1,
                    )

            # ===== Update layout =====
            fig_interactive.update_layout(
                title=f"Rainfall Intensity & Diagnostics - Full Session - Interactive (Total Tips: {total_tips})",
                hovermode="x unified",
                template="plotly_white",
                height=1400,
                showlegend=True,
            )

            # Update x-axes
            fig_interactive.update_xaxes(
                title_text="Time", tickformat="%H:%M:%S", tickmode="auto", row=1, col=1
            )
            fig_interactive.update_xaxes(
                title_text="Time", tickformat="%H:%M:%S", tickmode="auto", row=2, col=1
            )
            fig_interactive.update_xaxes(
                title_text="Time", tickformat="%H:%M:%S", tickmode="auto", row=3, col=1
            )

            # Update y-axes
            fig_interactive.update_yaxes(title_text="Intensity (mm/hr)", row=1, col=1)
            fig_interactive.update_yaxes(
                title_text="Interval RMS", row=2, col=1, secondary_y=False
            )
            fig_interactive.update_yaxes(
                title_text="Mic Amplitude", row=2, col=1, secondary_y=True
            )
            fig_interactive.update_yaxes(
                title_text="Mode / Status",
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["UNCALIBRATED", "BUCKET", "MIC"],
                range=[-0.5, 2.8],
                row=3,
                col=1,
            )

            fig_interactive.write_html(OUTPUT_HTML)
            print(f"Interactive plot saved to {OUTPUT_HTML}")

        # === PHASE 3: FIT CALIBRATION COEFFICIENTS ===
        if len(calibration_pairs) >= MIN_PAIRS_FOR_FIT:
            print(f"\n{'='*60}")
            print(f"CALIBRATION FIT SUMMARY")
            print(f"{'='*60}")
            print(f"Total valid calibration pairs collected: {len(calibration_pairs)}")

            # Extract RMS and intensity arrays
            rms_values = np.array([pair[0] for pair in calibration_pairs])
            intensity_values = np.array([pair[1] for pair in calibration_pairs])

            # Perform least-squares linear regression: intensity = A * rms + B
            # Using numpy polyfit (degree 1 for linear)
            coeffs = np.polyfit(rms_values, intensity_values, 1)
            new_A = coeffs[0]  # slope
            new_B = coeffs[1]  # intercept

            # Calculate fit quality metrics
            predicted = new_A * rms_values + new_B
            residuals = intensity_values - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((intensity_values - np.mean(intensity_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))

            print(f"\nFitted coefficients:")
            print(f"  A (slope):     {new_A:.6f}")
            print(f"  B (intercept): {new_B:.3f}")
            print(f"\nFit quality:")
            print(f"  R² score:      {r_squared:.4f}")
            print(f"  RMSE:          {rmse:.3f} mm/hr")
            print(f"  RMS range:     [{rms_values.min():.1f}, {rms_values.max():.1f}]")
            print(
                f"  Intensity range: [{intensity_values.min():.2f}, {intensity_values.max():.2f}] mm/hr"
            )

            # Save coefficients to file
            try:
                CALIB_COEFF_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CALIB_COEFF_FILE, "w") as cf:
                    json.dump(
                        {
                            "A": float(new_A),
                            "B": float(new_B),
                            "r_squared": float(r_squared),
                            "rmse": float(rmse),
                            "sample_count": len(calibration_pairs),
                            "rms_min": float(rms_values.min()),
                            "rms_max": float(rms_values.max()),
                            "intensity_min": float(intensity_values.min()),
                            "intensity_max": float(intensity_values.max()),
                            "timestamp": datetime.now().isoformat(),
                        },
                        cf,
                        indent=2,
                    )
                print(f"\nCalibration coefficients saved to {CALIB_COEFF_FILE}")
            except Exception as e:
                print(f"\nWarning: Failed to save calibration coefficients: {e}")

            print(f"{'='*60}\n")
        else:
            print(
                f"\n[CALIBRATION] Only {len(calibration_pairs)} valid pairs collected."
            )
            print(
                f"[CALIBRATION] Need at least {MIN_PAIRS_FOR_FIT} pairs for reliable fitting."
            )
            print(f"[CALIBRATION] Coefficients NOT updated.\n")

        print(f"\nDone! CSV saved to {OUTPUT_CSV}")
        print(f"Static plot saved to {OUTPUT_PNG}")
        print(f"Interactive plot saved to {OUTPUT_HTML}")
