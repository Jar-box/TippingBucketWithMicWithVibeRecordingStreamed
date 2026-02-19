import socket
import struct

import csv
import math
import os
import time
from collections import deque, defaultdict
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
MIC_DB_REF = 1.0
MIC_SUPPRESSION_SECONDS = 0.35

# === SMOOTHING PARAMETERS ===
SMOOTHING_FACTOR = 0.01

# === PERFORMANCE OPTIMIZATIONS ===
UPDATE_EVERY_N_SAMPLES = 10  # Only update plot every N samples
FIGURE_SIZE = (12, 7)

# === ZOOM LIMITS ===
MIN_TIME_ZOOM = 30.0
MIN_AMPLITUDE_ZOOM = 1024.0

MCAST_GRP = "230.138.19.201"
MCAST_PORT = 5007
IS_ALL_GROUPS = True

# ========== DIAGNOSTIC INSTRUMENTATION ==========
timing_stats = defaultdict(list)


def log_timing(operation: str, duration_ms: float):
    """Track timing for each operation type."""
    timing_stats[operation].append(duration_ms)


def print_timing_summary():
    """Print performance stats every N seconds."""
    print("\n" + "=" * 60)
    print("PERFORMANCE DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for op, times in sorted(timing_stats.items()):
        if times:
            avg = sum(times) / len(times)
            max_t = max(times)
            min_t = min(times)
            print(
                f"{op:30s}: avg={avg:6.2f}ms | max={max_t:6.2f}ms | min={min_t:6.2f}ms | count={len(times)}"
            )
    print("=" * 60 + "\n")


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.settimeout(2.0)  # seconds
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

last_reset_time = 0
tip_count_at_reset = 0
mic_amp_sum_at_reset = 0
sample_count_since_reset = 0
previous_tip_count = 0
mic_suppressed_until = 0.0
mic_history = deque(maxlen=10)
mic_amp_history = deque(maxlen=10)
sample_counter = 0
smoothed_mic_amp = 0.0
smoothed_mic_db = 0.0
mic_amp_suppressed_raw = 0.0


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


print("Running DIAGNOSTIC version with detailed timing instrumentation")
print(f"Socket timeout: 2.0s (waiting for Arduino data)")
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
        ]
    )

    plt.ion()
    fig, ax_amp_left = plt.subplots(1, 1, figsize=(12, 6))

    (line_mic_amp_raw,) = ax_amp_left.plot(
        [], [], label="Mic amplitude RAW", color="green", linewidth=1.5
    )
    (line_mic_amp_smooth,) = ax_amp_left.plot(
        [], [], label="Mic amplitude SMOOTHED", color="orange", linewidth=2
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
    print("(Diagnostic data will be printed every 5 seconds)")
    print()

    start_ts = time.time()
    last_reset_time = start_ts
    last_diagnostic_print = start_ts

    tip_lines_amp_left = []
    is_closed = False
    last_tip_count_drawn = 0

    def on_close(event):
        global is_closed
        is_closed = True

    cid = fig.canvas.mpl_connect("close_event", on_close)

    try:
        while not is_closed:
            # ===== RECEIVE FROM SOCKET =====
            recv_start = time.time()
            try:
                line = sock.recv(10240).decode("utf-8")
                if not line:
                    pass
            except socket.timeout:
                elapsed_total = time.time() - start_ts
                print(
                    f"[{elapsed_total:.1f}s] Socket timeout - no data received. Check Arduino/RPi connection."
                )
                raise KeyboardInterrupt()

            recv_time = (time.time() - recv_start) * 1000
            log_timing("Socket_recv", recv_time)

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

            # ===== DATA PROCESSING =====
            process_start = time.time()

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

            if elapsed_s % 5 < 0.5:
                print(
                    f"[{elapsed_s:.1f}s] amp={sound_amp}, tips={tip_count}, delta_tips={tips_since_reset}, dt={last_tip_dt_ms}ms"
                )

            if tip_count > previous_tip_count:
                bucket_rate = 40000.0
                previous_tip_count = tip_count
                mic_suppressed_until = now_ts + MIC_SUPPRESSION_SECONDS
                tip_times.append(now_ts)
                archive_tip_times.append(now_ts)
            else:
                bucket_rate = 0.0

            mic_rate_unfiltered = 20.0 * math.log10(max(sound_amp, 1) / MIC_DB_REF)
            mic_amp_unfiltered = sound_amp

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

            if now_ts < mic_suppressed_until:
                if len(mic_amp_history) > 0:
                    mic_amp_filtered = sum(mic_amp_history) / len(mic_amp_history)
                else:
                    mic_amp_filtered = smoothed_mic_amp

                if len(mic_history) > 0:
                    mic_rate_filtered = sum(mic_history) / len(mic_history)
                else:
                    mic_rate_filtered = smoothed_mic_db
            else:
                mic_amp_filtered = smoothed_mic_amp
                mic_rate_filtered = smoothed_mic_db

            mic_history.append(mic_rate_filtered)
            mic_amp_history.append(mic_amp_filtered)

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
                ]
            )
            if sample_counter % 100 == 0:
                f.flush()

            process_time = (time.time() - process_start) * 1000
            log_timing("Data_processing", process_time)

            # === THROTTLED PLOT UPDATE ===
            sample_counter += 1
            if sample_counter % UPDATE_EVERY_N_SAMPLES != 0:
                continue

            plot_start = time.time()

            if not series_t:
                continue

            t_rel = [t - start_ts for t in series_t]

            line_mic_amp_raw.set_data(t_rel, list(series_mic_amp_raw))
            line_mic_amp_smooth.set_data(t_rel, list(series_mic_amp_filtered))

            while last_tip_count_drawn < len(tip_times):
                tip_ts = tip_times[last_tip_count_drawn]
                x = tip_ts - start_ts
                if x >= 0:
                    tip_lines_amp_left.append(
                        ax_amp_left.axvline(
                            x=x, color="red", alpha=0.4, linewidth=1.5, linestyle="--"
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

            pause_start = time.time()
            plt.pause(0.001)
            pause_time = (time.time() - pause_start) * 1000

            plot_time = (time.time() - plot_start) * 1000
            log_timing("Plot_update", plot_time)
            log_timing("Matplotlib_pause", pause_time)

            # Print diagnostics every 5 seconds
            if now_ts - last_diagnostic_print >= 5.0:
                print_timing_summary()
                last_diagnostic_print = now_ts
                timing_stats.clear()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        plt.ioff()
        plt.close(fig)
        is_closed = True

        if archive_t:
            print("Generating final plot...")
            fig_archive, ax_amp_left_arch = plt.subplots(1, 1, figsize=(14, 7))
            t0_arch = archive_t[0]
            t_rel_arch = [t - t0_arch for t in archive_t]

            t_real_arch = [datetime.fromtimestamp(t) for t in archive_t]
            t_numeric_arch = mdates.date2num(t_real_arch)

            ax_amp_left_arch.plot(
                t_numeric_arch,
                list(archive_mic_amp_raw),
                label="Mic amplitude RAW",
                color="green",
                alpha=0.7,
                linewidth=1.5,
            )
            ax_amp_left_arch.plot(
                t_numeric_arch,
                list(archive_mic_amp_filtered),
                label="Mic amplitude SMOOTHED",
                color="orange",
                alpha=0.85,
                linewidth=2,
            )
            ax_amp_left_arch.set_title(
                "Mic Amplitude (RAW vs SMOOTHED) - Full Session",
                fontsize=14,
                fontweight="bold",
            )
            ax_amp_left_arch.set_xlabel("Time", fontsize=12)
            ax_amp_left_arch.set_ylabel("Mic amplitude", fontsize=12)
            ax_amp_left_arch.legend(loc="upper right", framealpha=0.95, fontsize=11)
            ax_amp_left_arch.grid(True, alpha=0.3)

            ax_amp_left_arch.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax_amp_left_arch.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig_archive.autofmt_xdate()

            for tip_ts in archive_tip_times:
                tip_numeric = float(mdates.date2num(datetime.fromtimestamp(tip_ts)))
                ax_amp_left_arch.axvline(
                    x=tip_numeric, color="red", alpha=0.3, linewidth=1.5, linestyle="--"
                )

            y_min_arch, y_max_arch = ax_amp_left_arch.get_ylim()
            if y_max_arch - y_min_arch < MIN_AMPLITUDE_ZOOM:
                y_center_arch = (y_max_arch + y_min_arch) / 2
                y_min_arch = y_center_arch - MIN_AMPLITUDE_ZOOM / 2
                y_max_arch = y_center_arch + MIN_AMPLITUDE_ZOOM / 2
                if y_min_arch < 0:
                    y_min_arch = 0
                    y_max_arch = MIN_AMPLITUDE_ZOOM
                ax_amp_left_arch.set_ylim(y_min_arch, y_max_arch)

            fig_archive.tight_layout()
            fig_archive.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
            plt.close(fig_archive)

            print("Generating interactive HTML plot...")
            fig_interactive = go.Figure()

            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_mic_amp_raw),
                    name="Mic amplitude RAW",
                    mode="lines",
                    line=dict(color="green", width=1.5),
                    hovertemplate="<b>RAW</b><br>Time: %{x|%H:%M:%S}<br>Amplitude: %{y:.0f}<extra></extra>",
                )
            )

            fig_interactive.add_trace(
                go.Scatter(
                    x=t_real_arch,
                    y=list(archive_mic_amp_filtered),
                    name="Mic amplitude SMOOTHED",
                    mode="lines",
                    line=dict(color="orange", width=2),
                    hovertemplate="<b>SMOOTHED</b><br>Time: %{x|%H:%M:%S}<br>Amplitude: %{y:.0f}<extra></extra>",
                )
            )

            for tip_ts in archive_tip_times:
                fig_interactive.add_vline(
                    x=tip_ts,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.3,
                )

            fig_interactive.update_layout(
                title="Mic Amplitude (RAW vs SMOOTHED) - Full Session - Interactive",
                xaxis_title="Time",
                yaxis_title="Mic Amplitude",
                hovermode="x unified",
                template="plotly_white",
                height=700,
                showlegend=True,
                legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
                xaxis=dict(
                    tickformat="%H:%M:%S",
                    tickmode="auto",
                ),
            )

            fig_interactive.write_html(OUTPUT_HTML)
            print(f"Interactive plot saved to {OUTPUT_HTML}")

        print(f"\nDone! CSV saved to {OUTPUT_CSV}")
        print(f"Static plot saved to {OUTPUT_PNG}")
        print(f"Interactive plot saved to {OUTPUT_HTML}")
