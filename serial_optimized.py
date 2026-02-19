import serial
import serial.tools.list_ports

import socket
import struct
import csv
import math
import os
import time
import threading
import queue
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    from pydub import AudioSegment
except Exception:
    sd = None
    sf = None
    AudioSegment = None


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


# ======== SERIAL CONFIG ========
SERIAL_PORT = "COM8"  # Change to your Arduino's COM port (or /dev/ttyUSB0 on Linux)
BAUD_RATE = 115200  # Must match Arduino's Serial.begin() rate
SERIAL_TIMEOUT = 2.0  # seconds

# ======== CONFIG ========
WINDOW_SECONDS = 30  # Plot window length (seconds)
TIP_MM_PER_TIP = 100  # Typical tipping bucket size (mm per tip)
RESET_INTERVAL = 60  # Reset tip count every N seconds for rate calculation
# Mic amplitude -> decibels (relative). Use 1.0 as reference to avoid -inf.
MIC_DB_REF = 1.0
MIC_SUPPRESSION_SECONDS = 0.35  # Suppress mic for this many seconds after a bucket tip

# === SMOOTHING PARAMETERS ===
SMOOTHING_FACTOR = 0.01  # Exponential smoothing alpha (0=max smoothing, 1=no smoothing)
# Lower values = more smoothing, prevents single spikes
# 0.15 allows gradual rain accumulation while filtering transient noise

# === PERFORMANCE OPTIMIZATIONS ===
UPDATE_EVERY_N_SAMPLES = 10  # Only update plot every N samples (reduces CPU/GPU load)
FIGURE_SIZE = (12, 7)  # Smaller figure = faster rendering

# === ZOOM LIMITS ===
MIN_TIME_ZOOM = 30.0  # Minimum x-axis range in seconds
MIN_AMPLITUDE_ZOOM = 1024.0  # Minimum y-axis range for amplitude

# ======== AUDIO RECORDING (UDP MULTICAST) ========
AUDIO_ENABLE = True
AUDIO_SAMPLE_RATE = 8000  # Match Arduino sampling rate
AUDIO_CHANNELS = 1
AUDIO_BITRATE = "192k"  # MP3 bitrate (high quality mono)
MCAST_GRP = "230.138.19.201"
MCAST_PORT = 5007
IS_ALL_GROUPS = True
DC_OFFSET = 512  # 10-bit ADC midpoint


def list_serial_ports():
    """List all available serial ports."""
    ports = serial.tools.list_ports.comports()
    print("\n=== Available Serial Ports ===")
    for port, desc, hwid in sorted(ports):
        print(f"  {port}: {desc}")
    print()


def open_serial_connection(port: str, baudrate: int, timeout: float):
    """Open serial connection to Arduino."""
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"Connected to {port} at {baudrate} baud")
        time.sleep(2)  # Wait for Arduino to reset after connection
        return ser
    except serial.SerialException as e:
        print(f"\nERROR: Could not open serial port {port}")
        print(f"  {e}")
        print("\nTroubleshooting:")
        print("  1. Check that Arduino is plugged in via USB")
        print("  2. Verify the correct COM port in the script (SERIAL_PORT variable)")
        print("  3. Close Arduino IDE Serial Monitor if it's open")
        print("  4. Check Device Manager (Windows) or 'ls /dev/tty*' (Linux/Mac)")
        list_serial_ports()
        raise


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
OUTPUT_WAV = output_dir / f"rain_intensity_{stamp}.wav"
OUTPUT_MP3 = output_dir / f"rain_intensity_{stamp}.mp3"

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
sample_counter = 0  # Performance: count samples between plot updates

# Exponentially smoothed values (initialized to zero, will converge over first few samples)
smoothed_mic_amp = 0.0
smoothed_mic_db = 0.0

# Raw amplitude with suppression applied (but less smoothing than orange line)
mic_amp_suppressed_raw = 0.0


class UDPAudioRecorder:
    """Records raw audio samples from UDP multicast stream to WAV file."""

    def __init__(
        self,
        wav_path: Path,
        sample_rate: int,
        channels: int,
        mcast_grp: str,
        mcast_port: int,
        is_all_groups: bool,
        dc_offset: int,
    ):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.mcast_grp = mcast_grp
        self.mcast_port = mcast_port
        self.is_all_groups = is_all_groups
        self.dc_offset = dc_offset
        self.stop_event = threading.Event()
        self.thread = None
        self.sock = None
        self.file = None

    def _recorder(self):
        """Background thread that receives UDP packets and writes audio samples to WAV."""
        try:
            # Setup UDP multicast socket
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            self.sock.settimeout(1.0)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self.is_all_groups:
                self.sock.bind(("", self.mcast_port))
            else:
                self.sock.bind((self.mcast_grp, self.mcast_port))
            mreq = struct.pack(
                "4sl", socket.inet_aton(self.mcast_grp), socket.INADDR_ANY
            )
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            # Open WAV file for writing
            if not sf:
                print("[AUDIO] soundfile not available, cannot record")
                return

            self.file = sf.SoundFile(
                self.wav_path,
                mode="w",
                samplerate=self.sample_rate,
                channels=self.channels,
                subtype="PCM_16",
            )

            audio_buffer = b""

            while not self.stop_event.is_set():
                try:
                    data = self.sock.recv(10240)
                    if not data:
                        continue

                    audio_buffer += data

                    # Process complete 2-byte samples
                    samples = []
                    while len(audio_buffer) >= 2:
                        high = audio_buffer[0]
                        low = audio_buffer[1]
                        audio_buffer = audio_buffer[2:]

                        # Skip tip markers (0xFF 0xFF followed by tip count)
                        if high == 0xFF and low == 0xFF:
                            if len(audio_buffer) >= 2:
                                tip_low = audio_buffer[0]
                                tip_high = audio_buffer[1]
                                tip_count = tip_low | (tip_high << 8)
                                if tip_count < 10000:
                                    audio_buffer = audio_buffer[2:]
                                    continue

                        # Convert 10-bit ADC to 16-bit PCM (centered around DC offset)
                        adc_value = (high << 8) | low
                        pcm_value = int(
                            (adc_value - self.dc_offset) * 32.0
                        )  # Scale to 16-bit range

                        # Clamp to 16-bit range
                        if pcm_value > 32767:
                            pcm_value = 32767
                        elif pcm_value < -32768:
                            pcm_value = -32768

                        samples.append(pcm_value)

                    # Write batch to file
                    if samples:
                        import numpy as np

                        self.file.write(np.array(samples, dtype=np.int16))

                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"[AUDIO] Recording error: {e}")
                    break

        finally:
            if self.file:
                self.file.close()
            if self.sock:
                self.sock.close()

    def start(self):
        """Start background recording thread."""
        self.thread = threading.Thread(target=self._recorder, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop recording and close files."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3.0)


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str):
    if not AudioSegment:
        print("[AUDIO] pydub/ffmpeg not available; keeping WAV only")
        return False
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate=bitrate)
    return True


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


print("=" * 60)
print("SERIAL VERSION - Direct USB connection to Arduino")
print("=" * 60)
list_serial_ports()

# Open serial connection
try:
    ser = open_serial_connection(SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT)
except:
    exit(1)

print("Running OPTIMIZED version (plot updates throttled for better performance)")

audio_recorder = None
if AUDIO_ENABLE:
    if not sf:
        print("[AUDIO] soundfile not available; audio recording disabled")
    else:
        try:
            audio_recorder = UDPAudioRecorder(
                OUTPUT_WAV,
                sample_rate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                mcast_grp=MCAST_GRP,
                mcast_port=MCAST_PORT,
                is_all_groups=IS_ALL_GROUPS,
                dc_offset=DC_OFFSET,
            )
            audio_recorder.start()
            print(f"[AUDIO] Recording UDP stream to {OUTPUT_WAV}")
            print(f"[AUDIO] Listening on multicast {MCAST_GRP}:{MCAST_PORT}")
        except Exception as e:
            audio_recorder = None
            print(f"[AUDIO] Failed to start recording: {e}")

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

    # ----- Plot setup -----
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

    print("Reading from Arduino... Press Ctrl+C to stop")

    start_ts = time.time()
    last_reset_time = start_ts

    tip_lines_amp_left = []
    is_closed = False

    def on_close(event):
        global is_closed
        is_closed = True

    cid = fig.canvas.mpl_connect("close_event", on_close)

    try:
        while not is_closed:
            try:
                # Read line from serial port
                line = ser.readline().decode("utf-8").strip()
                if not line:
                    continue
            except serial.SerialException as e:
                print(f"\n[SERIAL ERROR] {e}")
                break
            except UnicodeDecodeError:
                continue  # Skip malformed bytes

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
                ]
            )
            f.flush()

            # === PERFORMANCE OPTIMIZATION: Throttle plot updates ===
            sample_counter += 1
            if sample_counter % UPDATE_EVERY_N_SAMPLES != 0:
                continue  # Skip plot update to reduce CPU/GPU load

            # Update plot (only every N samples)
            if not series_t:
                continue

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

            # Batch all axis updates together for efficiency
            if t_rel:
                x_max = t_rel[-1]
                x_min = max(0, x_max - WINDOW_SECONDS)
                # Ensure minimum x-axis range
                if x_max - x_min < MIN_TIME_ZOOM:
                    x_max = x_min + MIN_TIME_ZOOM

                ax_amp_left.relim()
                ax_amp_left.autoscale_view()

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

            plt.pause(0.01)  # Slightly longer pause for smoother updates
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        print("Serial port closed")

        if audio_recorder:
            audio_recorder.stop()
            if convert_wav_to_mp3(OUTPUT_WAV, OUTPUT_MP3, AUDIO_BITRATE):
                print(f"[AUDIO] MP3 saved to {OUTPUT_MP3}")

        plt.ioff()
        plt.close(fig)
        is_closed = True

        # Save PNG with FULL archive data
        if archive_t:
            print("Generating final plot...")
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

        print(f"\nDone! CSV saved to {OUTPUT_CSV}")
        print(f"Static plot saved to {OUTPUT_PNG}")
        print(f"Interactive plot saved to {OUTPUT_HTML}")
        if audio_recorder and OUTPUT_WAV.exists():
            print(f"Audio WAV saved to {OUTPUT_WAV}")
            if OUTPUT_MP3.exists():
                print(f"Audio MP3 saved to {OUTPUT_MP3}")
