import socket
import struct

import csv
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go


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
WINDOW_SECONDS = 30  # Plot window length (seconds)
TIP_MM_PER_TIP = 100  # Typical tipping bucket size (mm per tip)
RESET_INTERVAL = 60  # Reset tip count every N seconds for rate calculation
# Mic amplitude -> decibels (relative). Use 1.0 as reference to avoid -inf.
MIC_DB_REF = 1.0
MIC_SUPPRESSION_SECONDS = 0.8  # Suppress mic for this many seconds after a bucket tip

# === SMOOTHING PARAMETERS ===
SMOOTHING_FACTOR = 0.01  # Exponential smoothing alpha (0=max smoothing, 1=no smoothing)
# Lower values = more smoothing, prevents single spikes
# 0.15 allows gradual rain accumulation while filtering transient noise

# === ZOOM LIMITS ===
MIN_TIME_ZOOM = 1.0  # Minimum x-axis range in seconds
MIN_AMPLITUDE_ZOOM = 100.0  # Minimum y-axis range for amplitude

# === RAIN DETECTION PARAMETERS ===
BASELINE_WINDOW = 60  # seconds - calculate baseline during quiet periods
BASELINE_MIN_SAMPLES = 20  # minimum samples needed to establish baseline
START_THRESHOLD_OFFSET = 100  # amplitude above baseline to trigger rain detection
START_THRESHOLD_MULTIPLIER = 1.5  # or baseline * this value (whichever is larger)
START_CONFIRMATION_TIME = 5.0  # seconds - wait for tip confirmation (expected 1.7s avg)
START_MIN_ELEVATED_TIME = 3.0  # seconds - mic must stay elevated before declaring rain detected
START_NO_TIP_TIMEOUT = 10.0  # seconds - if no tip but mic elevated, still confirm rain (light rain)
END_THRESHOLD_OFFSET = 20  # amplitude near baseline to trigger rain end
END_CONFIRMATION_TIME = 45.0  # seconds - wait for siphon drainage + buffer (2.74s min interval)
END_MIC_BASELINE_TIME = 10.0  # seconds - mic must return to baseline before end countdown

# Rain states
RAIN_STATE_NO_RAIN = 0
RAIN_STATE_DETECTED = 1
RAIN_STATE_CONFIRMED = 2
RAIN_STATE_ENDING = 3

MCAST_GRP = "230.138.19.201"
MCAST_PORT = 5007
IS_ALL_GROUPS = True

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.settimeout(2.0)  # seconds
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
if IS_ALL_GROUPS:
    # on this port, receives ALL multicast groups
    sock.bind(("", MCAST_PORT))
else:
    # on this port, listen ONLY to MCAST_GRP
    sock.bind((MCAST_GRP, MCAST_PORT))
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)

sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)


# ======== OUTPUT FILE ========
now = datetime.now()
stamp = now.strftime("%Y%m%d_%H%M%S")
date_folder = now.strftime("%Y-%m-%d")
hour_folder = now.strftime("%H")

# Create organized folder structure: data/YYYY-MM-DD/HH/
output_dir = Path("data") / date_folder / hour_folder
output_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = output_dir / f"rain_intensity_{stamp}.csv"
OUTPUT_PNG = output_dir / f"rain_intensity_{stamp}.png"
OUTPUT_HTML = output_dir / f"rain_intensity_{stamp}.html"

# ======== STATE ========
# Live display series (pruned to 30s window)
series_t = deque()
series_bucket = deque()
series_mic_raw = deque()
series_mic_filtered = deque()
series_mic_amp_raw = deque()
series_mic_amp_filtered = deque()
tip_times = deque()

# Archive series (ALL data for final PNG)
archive_t = deque()
archive_bucket = deque()
archive_mic_raw = deque()
archive_mic_filtered = deque()
archive_mic_amp_raw = deque()
archive_mic_amp_filtered = deque()
archive_tip_times = deque()

last_reset_time = 0
tip_count_at_reset = 0
mic_amp_sum_at_reset = 0
sample_count_since_reset = 0
previous_tip_count = 0
mic_suppressed_until = 0.0  # Timestamp when mic suppression ends
mic_history = deque(maxlen=10)  # dB history
mic_amp_history = deque(maxlen=10)  # amplitude history

# Exponentially smoothed values (initialized to zero, will converge over first few samples)
smoothed_mic_amp = 0.0
smoothed_mic_db = 0.0

# Raw amplitude with suppression applied (but less smoothing than orange line)
mic_amp_suppressed_raw = 0.0

# ======== RAIN DETECTION STATE ========
rain_state = RAIN_STATE_NO_RAIN
baseline_samples = deque(maxlen=BASELINE_WINDOW * 2)  # Store samples for baseline calculation
baseline_mic_amp = 0.0  # Current calculated baseline
rain_detection_time = 0.0  # When rain was first detected (mic threshold crossed)
rain_start_time = 0.0  # When rain was confirmed (tip occurred or timeout)
rain_end_countdown_start = 0.0  # When rain ending countdown started
last_tip_time = 0.0  # Timestamp of last tip
rain_events = []  # List of rain events: [(start_time, end_time, total_tips, max_intensity), ...]
current_rain_start_tip_count = 0  # Tip count when rain started


def exponential_smooth(
    current_value: float, smoothed_value: float, alpha: float = SMOOTHING_FACTOR
) -> float:
    """Apply exponential smoothing: new_smoothed = alpha * current + (1 - alpha) * previous_smoothed."""
    return alpha * current_value + (1.0 - alpha) * smoothed_value


def prune_old(now_ts: float) -> None:
    """Keep only samples within the rolling window."""
    while series_t and (now_ts - series_t[0]) > WINDOW_SECONDS:
        series_t.popleft()
        series_bucket.popleft()
        series_mic_raw.popleft()
        series_mic_filtered.popleft()
        series_mic_amp_raw.popleft()
        series_mic_amp_filtered.popleft()
    while tip_times and (now_ts - tip_times[0]) > WINDOW_SECONDS:
        tip_times.popleft()


def calculate_baseline(samples: deque) -> float:
    """Calculate baseline mic amplitude from recent quiet period samples."""
    if len(samples) < BASELINE_MIN_SAMPLES:
        return 0.0
    # Use median to avoid outliers
    sorted_samples = sorted(samples)
    mid = len(sorted_samples) // 2
    if len(sorted_samples) % 2 == 0:
        return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2.0
    else:
        return sorted_samples[mid]


def get_rain_state_name(state: int) -> str:
    """Get human-readable name for rain state."""
    states = {
        RAIN_STATE_NO_RAIN: "NO_RAIN",
        RAIN_STATE_DETECTED: "RAIN_DETECTED",
        RAIN_STATE_CONFIRMED: "RAIN_CONFIRMED",
        RAIN_STATE_ENDING: "RAIN_ENDING",
    }
    return states.get(state, "UNKNOWN")


print("Running.")

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
            "rain_state",
            "baseline_amp",
        ]
    )

    # ----- Plot setup -----
    plt.ion()
    fig, ax_amp_left = plt.subplots(1, 1, figsize=(12, 6))

    # ===== ARCHIVED: Other plot axes (commented out for future use) =====
    # fig, ((ax_db_left, ax_db_right), (ax_amp_left, ax_amp_right)) = plt.subplots(
    #     2, 2, figsize=(14, 9), sharex="col"
    # )

    (line_mic_amp_raw,) = ax_amp_left.plot(
        [], [], label="Mic amplitude RAW", color="green", linewidth=1.5
    )
    (line_mic_amp_smooth,) = ax_amp_left.plot(
        [], [], label="Mic amplitude SMOOTHED", color="orange", linewidth=2
    )

    # ===== ARCHIVED: Other axis setup (commented out for future use) =====
    # ax_db_left.set_title("Unfiltered Mic (dB, tips as lines)")
    # ax_db_right.set_title("Filtered Mic (dB, tips as lines)")
    # ax_db_left.set_yscale("symlog", linthresh=1.0)
    # ax_db_right.set_yscale("symlog", linthresh=1.0)
    # ax_db_left.set_ylim(0, 100)
    # ax_db_right.set_ylim(0, 100)

    ax_amp_left.set_title(
        "Mic Amplitude: RAW vs SMOOTHED", fontsize=14, fontweight="bold"
    )
    ax_amp_left.set_xlabel("Duration (s)", fontsize=12, labelpad=35)
    ax_amp_left.set_ylabel("Mic amplitude", fontsize=12)
    ax_amp_left.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax_amp_left.grid(True, alpha=0.3)

    # Create secondary x-axis for real time labels
    ax_time_left = ax_amp_left.twiny()
    ax_time_left.set_xlabel("Real Time", fontsize=10)
    ax_time_left.xaxis.set_ticks_position("bottom")
    ax_time_left.xaxis.set_label_position("bottom")
    ax_time_left.spines["bottom"].set_position(("outward", 40))

    fig.tight_layout()

    print("Reading serial... Press Ctrl+C to stop")

    start_ts = time.time()
    last_reset_time = start_ts

    tip_lines_db_left = []
    tip_lines_db_right = []
    tip_lines_amp_left = []
    tip_lines_amp_right = []
    rain_shade_patches = []  # Track rain period shading
    is_closed = False

    def on_close(event):
        global is_closed
        is_closed = True

    cid = fig.canvas.mpl_connect("close_event", on_close)

    try:
        while not is_closed:
            try:
                line = sock.recv(10240).decode("utf-8")
                if not line:
                    continue
            except socket.timeout:
                raise KeyboardInterrupt()

            # Skip any non-CSV header or malformed lines
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
                print(f"[PARSE ERROR] {line}")
                continue

            now_ts = time.time()
            elapsed_s = now_ts - start_ts

            # Reset mechanism: calculate rates over intervals
            if now_ts - last_reset_time >= RESET_INTERVAL:
                last_reset_time = now_ts
                tip_count_at_reset = tip_count
                mic_amp_sum_at_reset = 0
                sample_count_since_reset = 0
                print(
                    f"[RESET] Tip counter reset at {elapsed_s:.1f}s, tips={tip_count}"
                )

            # Calculate deltas since last reset
            tips_since_reset = tip_count - tip_count_at_reset
            sample_count_since_reset += 1
            time_since_reset = now_ts - last_reset_time

            # Debug: print every 5 seconds
            if elapsed_s % 5 < 0.5:
                print(
                    f"[{elapsed_s:.1f}s] amp={sound_amp}, tips={tip_count}, delta_tips={tips_since_reset}, dt={last_tip_dt_ms}ms"
                )

            # Compute bucket rate: spike only when a new tip occurs, then drop to 0
            if tip_count > previous_tip_count:
                # New tip detected! Use a fixed spike value for visibility
                bucket_rate = (
                    40000.0  # Fixed spike in mm/hr (adjust to match mic scale)
                )
                previous_tip_count = tip_count
                # Suppress mic for a short period to avoid mechanical noise
                mic_suppressed_until = now_ts + MIC_SUPPRESSION_SECONDS
                tip_times.append(now_ts)
                archive_tip_times.append(now_ts)
                last_tip_time = now_ts  # Track for rain detection
            else:
                # No new tip, rate is 0
                bucket_rate = 0.0

            # Compute mic-based level (dB, relative)
            mic_rate_unfiltered = 20.0 * math.log10(max(sound_amp, 1) / MIC_DB_REF)
            mic_amp_unfiltered = sound_amp

            # === EXPONENTIAL SMOOTHING ===
            # Apply exponential smoothing to both amplitude and dB
            # This prevents single spikes (single raindrops) from registering as high readings
            # while allowing gradual rainfall accumulation to be captured
            smoothed_mic_amp = exponential_smooth(
                mic_amp_unfiltered, smoothed_mic_amp, SMOOTHING_FACTOR
            )
            smoothed_mic_db = exponential_smooth(
                mic_rate_unfiltered, smoothed_mic_db, SMOOTHING_FACTOR
            )

            # === SUPPRESSION WITHOUT SMOOTHING (for green line) ===
            # Apply suppression to raw amplitude, but keep it unsmoothed
            if now_ts < mic_suppressed_until:
                # During suppression, flatten to the smoothed baseline to avoid mechanical noise
                mic_amp_suppressed_raw = smoothed_mic_amp
            else:
                # Normal operation: use raw unsmoothed value
                mic_amp_suppressed_raw = mic_amp_unfiltered

            # Filtered amplitude: apply suppression if recently tipped, otherwise use smoothed value
            if now_ts < mic_suppressed_until:
                # During suppression, use average of past samples to avoid mechanical noise
                if len(mic_amp_history) > 0:
                    mic_amp_filtered = sum(mic_amp_history) / len(mic_amp_history)
                else:
                    mic_amp_filtered = smoothed_mic_amp  # Fallback to smoothed value

                if len(mic_history) > 0:
                    mic_rate_filtered = sum(mic_history) / len(mic_history)
                else:
                    mic_rate_filtered = smoothed_mic_db  # Fallback to smoothed value
            else:
                # Normal operation: use smoothed value instead of raw value
                mic_amp_filtered = smoothed_mic_amp
                mic_rate_filtered = smoothed_mic_db

            # Track history for suppression period
            mic_history.append(mic_rate_filtered)
            mic_amp_history.append(mic_amp_filtered)

            # ======== RAIN DETECTION STATE MACHINE ========
            prev_rain_state = rain_state
            
            # Calculate dynamic baseline (only during NO_RAIN state)
            if rain_state == RAIN_STATE_NO_RAIN:
                baseline_samples.append(smoothed_mic_amp)
                baseline_mic_amp = calculate_baseline(baseline_samples)
            
            # Determine threshold values
            if baseline_mic_amp > 0:
                start_threshold = max(
                    baseline_mic_amp + START_THRESHOLD_OFFSET,
                    baseline_mic_amp * START_THRESHOLD_MULTIPLIER
                )
                end_threshold = baseline_mic_amp + END_THRESHOLD_OFFSET
            else:
                # No baseline established yet, use fixed values
                start_threshold = 150
                end_threshold = 50
            
            # State machine logic
            if rain_state == RAIN_STATE_NO_RAIN:
                # Check if mic amplitude exceeds start threshold
                if smoothed_mic_amp > start_threshold:
                    # Check if sustained for minimum time
                    if rain_detection_time == 0.0:
                        rain_detection_time = now_ts
                    elif (now_ts - rain_detection_time) >= START_MIN_ELEVATED_TIME:
                        # Transition to RAIN_DETECTED
                        rain_state = RAIN_STATE_DETECTED
                        print(f"\n[RAIN DETECTED] at {elapsed_s:.1f}s - Mic: {smoothed_mic_amp:.0f} > Threshold: {start_threshold:.0f} (baseline: {baseline_mic_amp:.0f})")
                else:
                    # Reset detection timer if mic drops below threshold
                    rain_detection_time = 0.0
            
            elif rain_state == RAIN_STATE_DETECTED:
                # Waiting for tip confirmation or timeout
                if tip_count > previous_tip_count or (last_tip_time > 0 and (now_ts - rain_detection_time) < START_CONFIRMATION_TIME):
                    # Tip occurred! Confirm rain
                    rain_state = RAIN_STATE_CONFIRMED
                    rain_start_time = now_ts
                    current_rain_start_tip_count = tip_count
                    print(f"[RAIN CONFIRMED] at {elapsed_s:.1f}s - Tip detected, total tips: {tip_count}")
                elif (now_ts - rain_detection_time) >= START_NO_TIP_TIMEOUT:
                    # No tip but mic stayed elevated - light rain confirmed
                    rain_state = RAIN_STATE_CONFIRMED
                    rain_start_time = now_ts
                    current_rain_start_tip_count = tip_count
                    print(f"[RAIN CONFIRMED - Light Rain] at {elapsed_s:.1f}s - No tip but sustained mic elevation")
                elif smoothed_mic_amp < start_threshold:
                    # False alarm, mic dropped back down
                    rain_state = RAIN_STATE_NO_RAIN
                    rain_detection_time = 0.0
                    print(f"[RAIN CANCELED] at {elapsed_s:.1f}s - False alarm, mic dropped")
            
            elif rain_state == RAIN_STATE_CONFIRMED:
                # Active rain - check for end conditions
                if smoothed_mic_amp < end_threshold:
                    # Mic returned to near baseline
                    if rain_end_countdown_start == 0.0:
                        rain_end_countdown_start = now_ts
                    elif (now_ts - rain_end_countdown_start) >= END_MIC_BASELINE_TIME:
                        # Mic has been at baseline long enough, start end countdown
                        rain_state = RAIN_STATE_ENDING
                        print(f"[RAIN ENDING] at {elapsed_s:.1f}s - Mic at baseline, waiting for siphon drainage...")
                else:
                    # Mic still elevated, reset countdown
                    rain_end_countdown_start = 0.0
            
            elif rain_state == RAIN_STATE_ENDING:
                # Waiting for siphon drainage and no new tips
                time_in_ending = now_ts - rain_end_countdown_start
                
                # Check for abort conditions
                if smoothed_mic_amp > end_threshold or tip_count > previous_tip_count:
                    # Rain resumed!
                    rain_state = RAIN_STATE_CONFIRMED
                    rain_end_countdown_start = 0.0
                    print(f"[RAIN RESUMED] at {elapsed_s:.1f}s - Activity detected during ending phase")
                elif time_in_ending >= END_CONFIRMATION_TIME:
                    # End confirmed!
                    rain_duration = now_ts - rain_start_time
                    total_rain_tips = tip_count - current_rain_start_tip_count
                    rain_events.append((rain_start_time, now_ts, total_rain_tips, 0.0))
                    print(f"\n[RAIN ENDED] at {elapsed_s:.1f}s")
                    print(f"  Duration: {rain_duration:.1f}s ({rain_duration/60:.1f} min)")
                    print(f"  Total tips: {total_rain_tips}")
                    print(f"  Rainfall: {total_rain_tips * 0.025:.2f} mm\n")
                    
                    # Reset to NO_RAIN
                    rain_state = RAIN_STATE_NO_RAIN
                    rain_detection_time = 0.0
                    rain_start_time = 0.0
                    rain_end_countdown_start = 0.0
                    baseline_samples.clear()  # Recalculate baseline

            # Store
            series_t.append(now_ts)
            series_bucket.append(bucket_rate)
            series_mic_raw.append(mic_rate_unfiltered)
            series_mic_filtered.append(mic_rate_filtered)
            series_mic_amp_raw.append(mic_amp_suppressed_raw)
            series_mic_amp_filtered.append(mic_amp_filtered)
            prune_old(now_ts)  # Prune live display to 30s window

            # Archive ALL data for final PNG
            archive_t.append(now_ts)
            archive_bucket.append(bucket_rate)
            archive_mic_raw.append(mic_rate_unfiltered)
            archive_mic_filtered.append(mic_rate_filtered)
            archive_mic_amp_raw.append(mic_amp_suppressed_raw)
            archive_mic_amp_filtered.append(mic_amp_filtered)

            # Archive ALL data for final PNG
            archive_t.append(now_ts)
            archive_bucket.append(bucket_rate)
            archive_mic_raw.append(mic_rate_unfiltered)
            archive_mic_filtered.append(mic_rate_filtered)
            archive_mic_amp_raw.append(mic_amp_unfiltered)
            archive_mic_amp_filtered.append(mic_amp_filtered)

            # Write log
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
                    get_rain_state_name(rain_state),
                    baseline_mic_amp,
                ]
            )
            f.flush()

            # Update plot
            t_rel = [t - start_ts for t in series_t]

            line_mic_amp_raw.set_data(t_rel, list(series_mic_amp_raw))
            line_mic_amp_smooth.set_data(t_rel, list(series_mic_amp_filtered))

            # Remove old tip lines and redraw for current window (amp_left only)
            for ln in tip_lines_amp_left:
                ln.remove()
            tip_lines_amp_left.clear()

            for tip_ts in tip_times:
                x = tip_ts - start_ts
                tip_lines_amp_left.append(
                    ax_amp_left.axvline(
                        x=x, color="red", alpha=0.4, linewidth=1.5, linestyle="--"
                    )
                )

            # Add rain period shading
            for patch in rain_shade_patches:
                patch.remove()
            rain_shade_patches.clear()

            # Shade active rain periods (CONFIRMED or ENDING states)
            if rain_state in [RAIN_STATE_CONFIRMED, RAIN_STATE_ENDING]:
                x_start = rain_start_time - start_ts
                x_end = now_ts - start_ts
                rain_shade_patches.append(
                    ax_amp_left.axvspan(
                        x_start, x_end, color="lightblue", alpha=0.2, label="Rain Period"
                    )
                )
            
            # Also shade completed rain events within the window
            for rain_event in rain_events:
                rain_evt_start, rain_evt_end, _, _ = rain_event
                x_start_evt = rain_evt_start - start_ts
                x_end_evt = rain_evt_end - start_ts
                # Only shade if within current window
                if x_end_evt >= (now_ts - start_ts - WINDOW_SECONDS):
                    rain_shade_patches.append(
                        ax_amp_left.axvspan(
                            x_start_evt, x_end_evt, color="lightblue", alpha=0.2
                        )
                    )

            # ===== ARCHIVED: Other tip lines (commented out for future use) =====
            # for ln in tip_lines_db_left: ln.remove()
            # for ln in tip_lines_db_right: ln.remove()
            # for ln in tip_lines_amp_right: ln.remove()
            # tip_lines_db_left.clear(); tip_lines_db_right.clear(); tip_lines_amp_right.clear()

            ax_amp_left.relim()
            ax_amp_left.autoscale_view()

            # Apply zoom limits
            x_max = t_rel[-1]
            x_min = max(0, x_max - WINDOW_SECONDS)
            # Ensure minimum x-axis range
            if x_max - x_min < MIN_TIME_ZOOM:
                x_max = x_min + MIN_TIME_ZOOM

            # Ensure minimum y-axis range
            y_min, y_max = ax_amp_left.get_ylim()
            if y_max - y_min < MIN_AMPLITUDE_ZOOM:
                y_center = (y_max + y_min) / 2
                y_min = y_center - MIN_AMPLITUDE_ZOOM / 2
                y_max = y_center + MIN_AMPLITUDE_ZOOM / 2
                # Ensure y_min is never negative (amplitude can't be negative)
                if y_min < 0:
                    y_min = 0
                    y_max = MIN_AMPLITUDE_ZOOM
                ax_amp_left.set_ylim(y_min, y_max)

            ax_amp_left.set_xlim(x_min, x_max)

            # Update secondary time axis
            ax_time_left.set_xlim(x_min, x_max)

            # Format time ticks to show real clock time
            def format_time_tick(x, pos):
                real_time = start_ts + x
                return datetime.fromtimestamp(real_time).strftime("%H:%M:%S")

            ax_time_left.xaxis.set_major_formatter(FuncFormatter(format_time_tick))

            plt.pause(0.001)  # Minimal pause for plot update
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        plt.ioff()
        plt.close(fig)
        is_closed = True

        # Save PNG with FULL archive data
        if archive_t:
            fig_archive, ax_amp_left_arch = plt.subplots(1, 1, figsize=(14, 7))
            t0_arch = archive_t[0]
            t_rel_arch = [t - t0_arch for t in archive_t]

            ax_amp_left_arch.plot(
                t_rel_arch,
                list(archive_mic_amp_raw),
                label="Mic amplitude RAW",
                color="green",
                alpha=0.7,
                linewidth=1.5,
            )
            ax_amp_left_arch.plot(
                t_rel_arch,
                list(archive_mic_amp_filtered),
                label="Mic amplitude SMOOTHED",
                color="orange",
                alpha=0.85,
                linewidth=2,
            )
            
            # Add rain period shading to archive plot
            for rain_event in rain_events:
                rain_evt_start, rain_evt_end, total_tips, _ = rain_event
                x_start_evt = rain_evt_start - t0_arch
                x_end_evt = rain_evt_end - t0_arch
                ax_amp_left_arch.axvspan(
                    x_start_evt, x_end_evt, color="lightblue", alpha=0.2, label="Rain Period" if rain_events[0] == rain_event else ""
                )
                # Add text annotation for rainfall amount
                x_mid = (x_start_evt + x_end_evt) / 2
                rainfall_mm = total_tips * 0.025
                duration_min = (rain_evt_end - rain_evt_start) / 60
                ax_amp_left_arch.text(
                    x_mid, ax_amp_left_arch.get_ylim()[1] * 0.9,
                    f"{rainfall_mm:.2f}mm\n{duration_min:.1f}min",
                    ha="center", va="top", fontsize=9, 
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                )
            ax_amp_left_arch.set_title(
                "Mic Amplitude (RAW vs SMOOTHED) - Full Session",
                fontsize=14,
                fontweight="bold",
            )
            ax_amp_left_arch.set_xlabel("Duration (s)", fontsize=12, labelpad=35)
            ax_amp_left_arch.set_ylabel("Mic amplitude", fontsize=12)
            ax_amp_left_arch.legend(loc="upper right", framealpha=0.95, fontsize=11)
            ax_amp_left_arch.grid(True, alpha=0.3)

            # Create secondary x-axis for real time labels on archive plot
            ax_time_arch = ax_amp_left_arch.twiny()
            ax_time_arch.set_xlabel("Real Time", fontsize=10)
            ax_time_arch.xaxis.set_ticks_position("bottom")
            ax_time_arch.xaxis.set_label_position("bottom")
            ax_time_arch.spines["bottom"].set_position(("outward", 40))
            ax_time_arch.set_xlim(ax_amp_left_arch.get_xlim())

            # Format time ticks to show real clock time
            def format_archive_time_tick(x, pos):
                real_time = t0_arch + x
                return datetime.fromtimestamp(real_time).strftime("%H:%M:%S")

            ax_time_arch.xaxis.set_major_formatter(
                FuncFormatter(format_archive_time_tick)
            )

            # Draw tip lines on archive plot
            for tip_ts in archive_tip_times:
                x = tip_ts - t0_arch
                ax_amp_left_arch.axvline(
                    x=x, color="red", alpha=0.3, linewidth=1, linestyle="--"
                )

            # Apply minimum y-axis zoom to saved PNG (height only, not time)
            y_min_arch, y_max_arch = ax_amp_left_arch.get_ylim()
            if y_max_arch - y_min_arch < MIN_AMPLITUDE_ZOOM:
                y_center_arch = (y_max_arch + y_min_arch) / 2
                y_min_arch = y_center_arch - MIN_AMPLITUDE_ZOOM / 2
                y_max_arch = y_center_arch + MIN_AMPLITUDE_ZOOM / 2
                # Ensure y_min is never negative (amplitude can't be negative)
                if y_min_arch < 0:
                    y_min_arch = 0
                    y_max_arch = MIN_AMPLITUDE_ZOOM
                ax_amp_left_arch.set_ylim(y_min_arch, y_max_arch)

            fig_archive.tight_layout()
            fig_archive.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
            plt.close(fig_archive)

            # Generate interactive Plotly HTML
            print("Generating interactive HTML plot...")
            fig_interactive = go.Figure()

            # Add RAW amplitude trace
            fig_interactive.add_trace(
                go.Scatter(
                    x=t_rel_arch,
                    y=list(archive_mic_amp_raw),
                    name="Mic amplitude RAW",
                    mode="lines",
                    line=dict(color="green", width=1.5),
                    hovertemplate="<b>RAW</b><br>Time: %{x:.2f}s<br>Amplitude: %{y:.0f}<extra></extra>",
                )
            )

            # Add SMOOTHED amplitude trace
            fig_interactive.add_trace(
                go.Scatter(
                    x=t_rel_arch,
                    y=list(archive_mic_amp_filtered),
                    name="Mic amplitude SMOOTHED",
                    mode="lines",
                    line=dict(color="orange", width=2),
                    hovertemplate="<b>SMOOTHED</b><br>Time: %{x:.2f}s<br>Amplitude: %{y:.0f}<extra></extra>",
                )
            )

            # Add tip lines as vertical lines
            for tip_ts in archive_tip_times:
                x_tip = tip_ts - t0_arch
                fig_interactive.add_vline(
                    x=x_tip,
                    line_color="red",
                    line_width=1,
                    line_dash="dash",
                    opacity=0.3,
                    annotation_text="Tip",
                    annotation_position="top",
                )

            # Add rain period shading as rectangles
            for i, rain_event in enumerate(rain_events):
                rain_evt_start, rain_evt_end, total_tips, _ = rain_event
                x_start_evt = rain_evt_start - t0_arch
                x_end_evt = rain_evt_end - t0_arch
                rainfall_mm = total_tips * 0.025
                duration_min = (rain_evt_end - rain_evt_start) / 60
                
                fig_interactive.add_vrect(
                    x0=x_start_evt, x1=x_end_evt,
                    fillcolor="lightblue", opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text=f"Rain: {rainfall_mm:.2f}mm, {duration_min:.1f}min",
                    annotation_position="top left" if i % 2 == 0 else "top right",
                )

            # Update layout for interactivity
            fig_interactive.update_layout(
                title="Mic Amplitude (RAW vs SMOOTHED) - Full Session - Interactive",
                xaxis_title="Time (s)",
                yaxis_title="Mic Amplitude",
                hovermode="x unified",
                template="plotly_white",
                height=700,
                showlegend=True,
                legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
            )

            # Save interactive HTML
            fig_interactive.write_html(OUTPUT_HTML)
            print(f"Interactive plot saved to {OUTPUT_HTML}")

        # Print rain event summary
        if rain_events:
            print("\n" + "=" * 60)
            print("RAIN EVENT SUMMARY")
            print("=" * 60)
            for i, rain_event in enumerate(rain_events, 1):
                rain_evt_start, rain_evt_end, total_tips, _ = rain_event
                rainfall_mm = total_tips * 0.025
                duration_sec = rain_evt_end - rain_evt_start
                duration_min = duration_sec / 60
                start_time_str = datetime.fromtimestamp(rain_evt_start).strftime("%H:%M:%S")
                end_time_str = datetime.fromtimestamp(rain_evt_end).strftime("%H:%M:%S")
                print(f"\nEvent {i}:")
                print(f"  Start: {start_time_str}")
                print(f"  End:   {end_time_str}")
                print(f"  Duration: {duration_min:.1f} minutes ({duration_sec:.0f}s)")
                print(f"  Tips: {total_tips}")
                print(f"  Rainfall: {rainfall_mm:.2f} mm")
                if duration_sec > 0:
                    intensity_mm_hr = (rainfall_mm / duration_sec) * 3600
                    print(f"  Avg Intensity: {intensity_mm_hr:.1f} mm/hr")
            print("=" * 60 + "\n")
        else:
            print("\nNo rain events detected during this session.\n")

        print(f"\nDone! CSV saved to {OUTPUT_CSV}")
        print(f"Static plot saved to {OUTPUT_PNG}")
        print(f"Interactive plot saved to {OUTPUT_HTML}")
