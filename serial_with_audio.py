"""
SERIAL WITH AUDIO VERSION
Receives sensor data (CSV) AND raw audio samples via USB/Serial connection.
Requires Arduino running TippingBucketSerialAudio.ino at 230400 baud.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
import csv
import math
import struct
import time
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go

try:
    import soundfile as sf
    import numpy as np
    from pydub import AudioSegment
except Exception:
    sf = None
    np = None
    AudioSegment = None


# ======== SERIAL CONFIG ========
SERIAL_PORT = "COM8"  # Change to your Arduino's COM port
BAUD_RATE = 230400  # MUST match Arduino's Serial.begin() rate (higher for audio)
SERIAL_TIMEOUT = 2.0  # seconds

# ======== CONFIG ========
WINDOW_SECONDS = 30
TIP_MM_PER_TIP = 100
RESET_INTERVAL = 60
MIC_DB_REF = 1.0
MIC_SUPPRESSION_SECONDS = 0.35

# === SMOOTHING PARAMETERS ===
SMOOTHING_FACTOR = 0.01

# === PERFORMANCE OPTIMIZATIONS ===
UPDATE_EVERY_N_SAMPLES = 10
MIN_TIME_ZOOM = 30.0
MIN_AMPLITUDE_ZOOM = 1024.0

# === AUDIO RECORDING (VIA SERIAL) ===
AUDIO_ENABLE = True
AUDIO_SAMPLE_RATE = 8000  # 8 kHz (matches Arduino actual rate with overhead)
AUDIO_CHANNELS = 1
AUDIO_BITRATE = "128k"
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
        time.sleep(2)  # Wait for Arduino to reset
        return ser
    except serial.SerialException as e:
        print(f"\nERROR: Could not open serial port {port}")
        print(f"  {e}")
        print("\nTroubleshooting:")
        print("  1. Check that Arduino is plugged in via USB")
        print("  2. Verify the correct COM port (SERIAL_PORT variable)")
        print("  3. Close Arduino IDE Serial Monitor if it's open")
        print("  4. Make sure Arduino is running TippingBucketSerialAudio.ino")
        list_serial_ports()
        raise


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
OUTPUT_WAV = output_dir / f"rain_intensity_{stamp}.wav"
OUTPUT_MP3 = output_dir / f"rain_intensity_{stamp}.mp3"

# ======== STATE ========
series_t = deque()
series_mic_amp_raw = deque()
series_mic_amp_filtered = deque()
tip_times = deque()

archive_t = deque()
archive_mic_amp_raw = deque()
archive_mic_amp_filtered = deque()
archive_tip_times = deque()

last_reset_time = 0
tip_count_at_reset = 0
previous_tip_count = 0
mic_suppressed_until = 0.0
mic_amp_history = deque(maxlen=10)
sample_counter = 0

smoothed_mic_amp = 0.0

# Audio recording
audio_file = None
audio_samples_written = 0


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str):
    if not AudioSegment:
        print("[AUDIO] pydub/ffmpeg not available; keeping WAV only")
        return False
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        return True
    except FileNotFoundError:
        print("[AUDIO] ffmpeg not found in PATH; keeping WAV only")
        print("[AUDIO] Install: winget install ffmpeg")
        return False
    except Exception as e:
        print(f"[AUDIO] MP3 conversion failed: {e}")
        return False


def exponential_smooth(
    current_value: float, smoothed_value: float, alpha: float = SMOOTHING_FACTOR
) -> float:
    return alpha * current_value + (1.0 - alpha) * smoothed_value


def prune_old(now_ts: float) -> None:
    """Keep only samples within the rolling window."""
    while series_t and (now_ts - series_t[0]) > WINDOW_SECONDS:
        series_t.popleft()
        series_mic_amp_raw.popleft()
        series_mic_amp_filtered.popleft()
    while tip_times and (now_ts - tip_times[0]) > WINDOW_SECONDS:
        tip_times.popleft()


print("=" * 70)
print("SERIAL WITH AUDIO - USB connection with audio streaming")
print("=" * 70)
list_serial_ports()

try:
    ser = open_serial_connection(SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT)
except:
    exit(1)

# Open audio file
if AUDIO_ENABLE and sf:
    try:
        audio_file = sf.SoundFile(
            OUTPUT_WAV,
            mode="w",
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            subtype="PCM_16",
        )
        print(
            f"[AUDIO] Recording to {OUTPUT_WAV} at {AUDIO_SAMPLE_RATE} Hz (~10kHz sampling with overhead)"
        )
    except Exception as e:
        print(f"[AUDIO] Failed to open WAV file: {e}")
        audio_file = None
else:
    print("[AUDIO] Recording disabled")

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
    fig, ax_amp = plt.subplots(1, 1, figsize=(12, 6))

    (line_mic_amp_raw,) = ax_amp.plot(
        [], [], label="Mic amplitude RAW", color="green", linewidth=1.5
    )
    (line_mic_amp_smooth,) = ax_amp.plot(
        [], [], label="Mic amplitude SMOOTHED", color="orange", linewidth=2
    )

    ax_amp.set_title(
        "Mic Amplitude: RAW vs SMOOTHED (Audio via Serial)",
        fontsize=14,
        fontweight="bold",
    )
    ax_amp.set_xlabel("Duration (s)", fontsize=12, labelpad=35)
    ax_amp.set_ylabel("Mic amplitude", fontsize=12)
    ax_amp.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax_amp.grid(True, alpha=0.3)

    ax_time = ax_amp.twiny()
    ax_time.set_xlabel("Real Time", fontsize=10)
    ax_time.xaxis.set_ticks_position("bottom")
    ax_time.xaxis.set_label_position("bottom")
    ax_time.spines["bottom"].set_position(("outward", 40))

    fig.tight_layout()

    print("Reading from Arduino... Press Ctrl+C to stop")
    print("Waiting for data...")

    start_ts = time.time()
    last_reset_time = start_ts
    tip_lines_amp = []
    is_closed = False

    def on_close(event):
        global is_closed
        is_closed = True

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        while not is_closed:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
            except serial.SerialException as e:
                print(f"\n[SERIAL ERROR] {e}")
                break
            except UnicodeDecodeError:
                continue

            # ===== PARSE AUDIO DATA =====
            if line.startswith("AUDIO:"):
                if not audio_file or not np:
                    continue

                try:
                    # Read count (2 bytes after "AUDIO:")
                    remaining = line[6:]  # Skip "AUDIO:"
                    if len(remaining) < 2:
                        continue

                    # Convert to int safely
                    count_high = (
                        ord(remaining[0])
                        if isinstance(remaining[0], str)
                        else int(remaining[0])
                    )
                    count_low = (
                        ord(remaining[1])
                        if isinstance(remaining[1], str)
                        else int(remaining[1])
                    )
                    sample_count = int((count_high << 8) | count_low)

                    # Read samples (2 bytes each)
                    samples = []
                    pos = 2
                    for _ in range(sample_count):
                        if pos + 2 > len(remaining):
                            break
                        high = (
                            ord(remaining[pos])
                            if isinstance(remaining[pos], str)
                            else int(remaining[pos])
                        )
                        low = (
                            ord(remaining[pos + 1])
                            if isinstance(remaining[pos + 1], str)
                            else int(remaining[pos + 1])
                        )
                        adc_value = int((high << 8) | low)

                        # Convert 10-bit ADC to 16-bit PCM
                        pcm_value = int((adc_value - DC_OFFSET) * 32.0)
                        pcm_value = max(-32768, min(32767, pcm_value))  # Clamp
                        samples.append(pcm_value)
                        pos += 2

                    if samples and np:
                        audio_file.write(np.array(samples, dtype=np.int16))
                        audio_samples_written += len(samples)

                        # Report every 10000 samples (~2.5 seconds at 4kHz)
                        if audio_samples_written % 10000 < len(samples):
                            duration = audio_samples_written / AUDIO_SAMPLE_RATE
                            print(
                                f"[AUDIO] Recorded {duration:.1f}s ({audio_samples_written} samples)"
                            )

                except Exception as e:
                    print(f"[AUDIO PARSE ERROR] {e}")

                continue

            # ===== PARSE CSV DATA =====
            # Skip headers
            if line.startswith("ts_ms") or line.startswith("SERIAL_AUDIO_MODE"):
                continue

            # Skip debug messages
            if "Pin state" in line or "HIGH" in line or "LOW" in line:
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

            now_ts = time.time()
            elapsed_s = now_ts - start_ts

            # Reset mechanism
            if now_ts - last_reset_time >= RESET_INTERVAL:
                last_reset_time = now_ts
                tip_count_at_reset = tip_count

            # Bucket tip detection
            if tip_count > previous_tip_count:
                bucket_rate = 40000.0
                previous_tip_count = tip_count
                mic_suppressed_until = now_ts + MIC_SUPPRESSION_SECONDS
                tip_times.append(now_ts)
                archive_tip_times.append(now_ts)
            else:
                bucket_rate = 0.0

            # Microphone processing
            mic_rate_unfiltered = 20.0 * math.log10(max(sound_amp, 1) / MIC_DB_REF)
            mic_amp_unfiltered = sound_amp

            smoothed_mic_amp = exponential_smooth(
                mic_amp_unfiltered, smoothed_mic_amp, SMOOTHING_FACTOR
            )

            if now_ts < mic_suppressed_until:
                mic_amp_suppressed_raw = smoothed_mic_amp
                mic_amp_filtered = (
                    smoothed_mic_amp
                    if not mic_amp_history
                    else sum(mic_amp_history) / len(mic_amp_history)
                )
            else:
                mic_amp_suppressed_raw = mic_amp_unfiltered
                mic_amp_filtered = smoothed_mic_amp

            mic_amp_history.append(mic_amp_filtered)

            # Store data
            series_t.append(now_ts)
            series_mic_amp_raw.append(mic_amp_suppressed_raw)
            series_mic_amp_filtered.append(mic_amp_filtered)
            prune_old(now_ts)

            archive_t.append(now_ts)
            archive_mic_amp_raw.append(mic_amp_suppressed_raw)
            archive_mic_amp_filtered.append(mic_amp_filtered)

            # Write CSV
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
                    0,  # mic_rate_filtered
                ]
            )
            f.flush()

            # Update plot
            sample_counter += 1
            if sample_counter % UPDATE_EVERY_N_SAMPLES != 0:
                continue

            if not series_t:
                continue

            t_rel = [t - start_ts for t in series_t]

            line_mic_amp_raw.set_data(t_rel, list(series_mic_amp_raw))
            line_mic_amp_smooth.set_data(t_rel, list(series_mic_amp_filtered))

            for ln in tip_lines_amp:
                ln.remove()
            tip_lines_amp.clear()

            for tip_ts in tip_times:
                x = tip_ts - start_ts
                tip_lines_amp.append(
                    ax_amp.axvline(
                        x=x, color="red", alpha=0.4, linewidth=1.5, linestyle="--"
                    )
                )

            if t_rel:
                x_max = t_rel[-1]
                x_min = max(0, x_max - WINDOW_SECONDS)
                if x_max - x_min < MIN_TIME_ZOOM:
                    x_max = x_min + MIN_TIME_ZOOM

                ax_amp.relim()
                ax_amp.autoscale_view()

                y_min, y_max = ax_amp.get_ylim()
                if y_max - y_min < MIN_AMPLITUDE_ZOOM:
                    y_center = (y_max + y_min) / 2
                    y_min = max(0, y_center - MIN_AMPLITUDE_ZOOM / 2)
                    y_max = y_min + MIN_AMPLITUDE_ZOOM
                    ax_amp.set_ylim(y_min, y_max)

                ax_amp.set_xlim(x_min, x_max)
                ax_time.set_xlim(x_min, x_max)

                def format_time_tick(x, pos):
                    real_time = start_ts + x
                    return datetime.fromtimestamp(real_time).strftime("%H:%M:%S")

                ax_time.xaxis.set_major_formatter(FuncFormatter(format_time_tick))

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        print("Serial port closed")

        if audio_file:
            audio_file.close()
            print(
                f"[AUDIO] Saved {audio_samples_written} samples ({audio_samples_written/AUDIO_SAMPLE_RATE:.1f}s)"
            )

            if OUTPUT_WAV.exists() and convert_wav_to_mp3(
                OUTPUT_WAV, OUTPUT_MP3, AUDIO_BITRATE
            ):
                print(f"[AUDIO] MP3 saved to {OUTPUT_MP3}")

        plt.ioff()
        plt.close(fig)
        is_closed = True

        # Save PNG
        if archive_t:
            print("Generating final plot...")
            fig_archive, ax_arch = plt.subplots(1, 1, figsize=(14, 7))
            t0 = archive_t[0]
            t_rel = [t - t0 for t in archive_t]

            ax_arch.plot(
                t_rel,
                list(archive_mic_amp_raw),
                label="RAW",
                color="green",
                alpha=0.7,
                linewidth=1.5,
            )
            ax_arch.plot(
                t_rel,
                list(archive_mic_amp_filtered),
                label="SMOOTHED",
                color="orange",
                alpha=0.85,
                linewidth=2,
            )

            ax_arch.set_title(
                "Mic Amplitude - Full Session (Audio via Serial)",
                fontsize=14,
                fontweight="bold",
            )
            ax_arch.set_xlabel("Duration (s)", fontsize=12)
            ax_arch.set_ylabel("Mic amplitude", fontsize=12)
            ax_arch.legend(loc="upper right")
            ax_arch.grid(True, alpha=0.3)

            for tip_ts in archive_tip_times:
                ax_arch.axvline(
                    x=tip_ts - t0, color="red", alpha=0.3, linewidth=1, linestyle="--"
                )

            fig_archive.tight_layout()
            fig_archive.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
            plt.close(fig_archive)
            print(f"Plot saved to {OUTPUT_PNG}")

        print(f"\nDone! CSV saved to {OUTPUT_CSV}")
        if AUDIO_ENABLE and OUTPUT_WAV.exists():
            print(f"Audio WAV saved to {OUTPUT_WAV}")
