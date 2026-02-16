import socket
import struct

import csv
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import threading

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import cv2
import numpy as np


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


# ======== CAMERA CONFIG ========
# Iriun Webcam - works with iPhone and Android
# Get your iPhone's IP address from Settings → WiFi → (i) icon
# Start Iriun app on your iPhone, then update the IP below
# Common formats:
# - Iriun: "http://192.168.1.100:8080/video"
# - IP Webcam: "http://192.168.1.100:8080/video"
# - DroidCam: "http://192.168.1.100:4747/video"
CAMERA_URL = "http://192.168.1.145:8080/video"  # CHANGE IP TO YOUR IPHONE'S IP
CAMERA_ENABLED = True  # Set to False to disable camera if troubleshooting
CAMERA_TIMEOUT = 5  # Seconds to wait for camera connection
CAMERA_RECONNECT_INTERVAL = 10  # Seconds between reconnection attempts
CAMERA_ROTATION = 90  # Rotation in degrees: 0, 90, 180, or 270
CAMERA_FPS_LIMIT = 10  # Max FPS for camera (lower = less CPU usage, smoother performance)

# ======== CONFIG ========
WINDOW_SECONDS = 30  # Plot window length (seconds)
TIP_MM_PER_TIP = 100  # Typical tipping bucket size (mm per tip)
RESET_INTERVAL = 60  # Reset tip count every N seconds for rate calculation
# Mic amplitude -> decibels (relative). Use 1.0 as reference to avoid -inf.
MIC_DB_REF = 1.0
MIC_SUPPRESSION_SECONDS = 0.35  # Suppress mic for this many seconds after a bucket tip

# === SMOOTHING PARAMETERS ===
SMOOTHING_FACTOR = 0.05  # Exponential smoothing alpha (0=max smoothing, 1=no smoothing)
# Lower values = more smoothing, prevents single spikes
# 0.15 allows gradual rain accumulation while filtering transient noise

# === PERFORMANCE OPTIMIZATIONS ===
UPDATE_EVERY_N_SAMPLES = 30  # Only update plot every N samples (reduces CPU/GPU load)
FIGURE_SIZE = (16, 7)  # Wider figure to accommodate camera and graph side-by-side

# === ZOOM LIMITS ===
MIN_TIME_ZOOM = 30.0  # Minimum x-axis range in seconds
MIN_AMPLITUDE_ZOOM = 1024.0  # Minimum y-axis range for amplitude

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
sample_counter = 0  # Performance: count samples between plot updates

# Exponentially smoothed values (initialized to zero, will converge over first few samples)
smoothed_mic_amp = 0.0
smoothed_mic_db = 0.0


# ======== CAMERA CAPTURE THREAD ========
class CameraThread:
    """Thread-safe camera capture with automatic reconnection."""

    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None
        self.last_frame_time = time.time()
        self.connection_status = "Disconnected"
        self.last_reconnect_attempt = 0

    def start(self):
        """Start the camera capture thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[CAMERA] Starting camera thread with URL: {self.camera_url}")

    def _capture_loop(self):
        """Continuously capture frames from camera."""
        while self.running:
            # Try to connect/reconnect if needed
            if self.cap is None or not self.cap.isOpened():
                current_time = time.time()
                if (
                    current_time - self.last_reconnect_attempt
                    >= CAMERA_RECONNECT_INTERVAL
                ):
                    self._connect()
                    self.last_reconnect_attempt = current_time
                else:
                    time.sleep(1)
                    continue

            # Try to read a frame
            if self.cap is None:
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Rotate frame if needed
                if CAMERA_ROTATION == 90:
                    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
                elif CAMERA_ROTATION == 180:
                    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
                elif CAMERA_ROTATION == 270:
                    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Add timestamp overlay
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Add semi-transparent background for better text readability
                overlay = frame_rgb.copy()
                cv2.rectangle(overlay, (10, 10), (450, 60), (0, 0, 0), -1)
                frame_rgb = cv2.addWeighted(overlay, 0.5, frame_rgb, 0.5, 0)

                # Add timestamp text
                cv2.putText(
                    frame_rgb,
                    timestamp_str,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),  # Green color
                    2,
                    cv2.LINE_AA,
                )

                # Update frame in thread-safe manner
                with self.lock:
                    self.frame = frame_rgb
                    self.last_frame_time = time.time()
                    self.connection_status = "Connected"
            else:
                # Failed to read frame, reconnect
                print(f"[CAMERA] Failed to read frame, reconnecting...")
                self.connection_status = "Reconnecting"
                self._disconnect()

            # Small delay to prevent CPU overload
            time.sleep(1.0 / CAMERA_FPS_LIMIT)  # Limit FPS for better performance

    def _connect(self):
        """Connect to camera stream."""
        print(f"[CAMERA] Connecting to {self.camera_url}...")
        try:
            self.cap = cv2.VideoCapture(self.camera_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

            # Test if connection successful
            ret, frame = self.cap.read()
            if ret:
                print(f"[CAMERA] Successfully connected! Frame size: {frame.shape}")
                self.connection_status = "Connected"
            else:
                print(f"[CAMERA] Connection failed - couldn't read initial frame")
                self._disconnect()
        except Exception as e:
            print(f"[CAMERA] Connection error: {e}")
            self._disconnect()

    def _disconnect(self):
        """Disconnect from camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.connection_status = "Disconnected"

    def get_frame(self):
        """Get the latest frame in thread-safe manner."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_status(self):
        """Get connection status."""
        with self.lock:
            return self.connection_status

    def stop(self):
        """Stop the camera thread."""
        print("[CAMERA] Stopping camera thread...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._disconnect()


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


print("Running CAMERA-INTEGRATED version (graph + live camera feed)")
print(f"Camera URL: {CAMERA_URL}")
print(
    "IMPORTANT: Start Iriun app on your iPhone, then update CAMERA_URL with your iPhone's IP!"
)

# Initialize camera thread
camera = None
if CAMERA_ENABLED:
    camera = CameraThread(CAMERA_URL)
    camera.start()

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

    # ----- Plot setup with camera view -----
    plt.ion()

    if CAMERA_ENABLED:
        # Create figure with 2 columns: camera (left) and graph (right)
        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], figure=fig)

        ax_camera = fig.add_subplot(gs[0, 0])
        ax_amp = fig.add_subplot(gs[0, 1])

        # Setup camera axes
        ax_camera.set_title("Live Camera Feed", fontsize=14, fontweight="bold")
        ax_camera.axis("off")

        # Initialize with placeholder
        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder_frame,
            "Connecting to camera...",
            (150, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        camera_img = ax_camera.imshow(placeholder_frame)
    else:
        # No camera - single graph only
        fig, ax_amp = plt.subplots(1, 1, figsize=(12, 6))
        camera_img = None  # Define to avoid unbound variable error

    # Setup amplitude graph
    (line_mic_amp_raw,) = ax_amp.plot(
        [], [], label="Mic amplitude RAW", color="green", linewidth=1.5
    )
    (line_mic_amp_smooth,) = ax_amp.plot(
        [], [], label="Mic amplitude SMOOTHED", color="orange", linewidth=2
    )

    ax_amp.set_title("Mic Amplitude: RAW vs SMOOTHED", fontsize=14, fontweight="bold")
    ax_amp.set_xlabel("Time (s)", fontsize=12)
    ax_amp.set_ylabel("Mic amplitude", fontsize=12)
    ax_amp.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax_amp.grid(True, alpha=0.3)
    fig.tight_layout()

    print("Reading serial... Press Ctrl+C to stop")

    start_ts = time.time()
    last_reset_time = start_ts

    tip_lines_amp = []
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
                camera_status = camera.get_status() if camera else "Disabled"
                print(
                    f"[{elapsed_s:.1f}s] amp={sound_amp}, tips={tip_count}, "
                    f"delta_tips={tips_since_reset}, camera={camera_status}"
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
            series_mic_amp_raw.append(mic_amp_unfiltered)
            series_mic_amp_filtered.append(mic_amp_filtered)
            prune_old(now_ts)  # Prune live display to 30s window

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

            # Remove old tip lines and redraw for current window
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

            # Update camera frame
            if CAMERA_ENABLED and camera and camera_img is not None:
                frame = camera.get_frame()
                if frame is not None:
                    camera_img.set_data(frame)
                else:
                    # Show connection status if no frame available
                    status = camera.get_status()
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        f"Camera: {status}",
                        (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2,
                    )
                    camera_img.set_data(placeholder)

            # Batch all axis updates together for efficiency
            if t_rel:
                x_max = t_rel[-1]
                x_min = max(0, x_max - WINDOW_SECONDS)
                # Ensure minimum x-axis range
                if x_max - x_min < MIN_TIME_ZOOM:
                    x_max = x_min + MIN_TIME_ZOOM

                ax_amp.relim()
                ax_amp.autoscale_view()

                # Ensure minimum y-axis range
                y_min, y_max = ax_amp.get_ylim()
                if y_max - y_min < MIN_AMPLITUDE_ZOOM:
                    y_center = (y_max + y_min) / 2
                    y_min = y_center - MIN_AMPLITUDE_ZOOM / 2
                    y_max = y_center + MIN_AMPLITUDE_ZOOM / 2
                    # Ensure y_min is never negative (amplitude can't be negative)
                    if y_min < 0:
                        y_min = 0
                        y_max = MIN_AMPLITUDE_ZOOM
                    ax_amp.set_ylim(y_min, y_max)

                ax_amp.set_xlim(x_min, x_max)

            plt.pause(0.01)  # Slightly longer pause for smoother updates
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop camera thread
        if camera:
            camera.stop()

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
            ax_amp_left_arch.set_xlabel("Time (s)", fontsize=12)
            ax_amp_left_arch.set_ylabel("Mic amplitude", fontsize=12)
            ax_amp_left_arch.legend(loc="upper right", framealpha=0.95, fontsize=11)
            ax_amp_left_arch.grid(True, alpha=0.3)

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
