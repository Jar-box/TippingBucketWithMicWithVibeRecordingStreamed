"""
SERIAL PURE AUDIO - Records ONLY audio via Serial (like audio_stream_player but Serial instead of UDP)
No CSV data, no plotting - just records audio to WAV
Upload: arduino/TippingBucketPureAudioSerial/TippingBucketPureAudioSerial.ino
"""

from datetime import datetime
from pathlib import Path
import struct
import time
import serial
import serial.tools.list_ports

try:
    import soundfile as sf
    import numpy as np
    from pydub import AudioSegment
except Exception:
    sf = None
    np = None
    AudioSegment = None

# ======== SERIAL CONFIG ========
SERIAL_PORT = "COM8"
BAUD_RATE = 230400
SERIAL_TIMEOUT = 1.0

# ======== AUDIO CONFIG ========
AUDIO_SAMPLE_RATE = 8000  # 8 kHz actual rate
AUDIO_BITRATE = "192k"
DC_OFFSET = 512

# ======== OUTPUT FILE ========
now = datetime.now()
stamp = now.strftime("%Y%m%d_%H%M%S")
date_folder = now.strftime("%Y-%m-%d")
hour_folder = now.strftime("%H")

output_dir = Path("data") / date_folder / hour_folder
output_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_WAV = output_dir / f"audio_{stamp}.wav"
OUTPUT_MP3 = output_dir / f"audio_{stamp}.mp3"


def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    print("\n=== Available Serial Ports ===")
    for port, desc, hwid in sorted(ports):
        print(f"  {port}: {desc}")
    print()


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str):
    if not AudioSegment:
        print("[AUDIO] pydub/ffmpeg not available; keeping WAV only")
        return False
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        return True
    except FileNotFoundError:
        print("[AUDIO] ffmpeg not found; keeping WAV only")
        return False
    except Exception as e:
        print(f"[AUDIO] MP3 conversion failed: {e}")
        return False


print("=" * 70)
print("SERIAL PURE AUDIO - Recording audio only (no CSV)")
print("=" * 70)
list_serial_ports()

# Open serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
    time.sleep(2)
except serial.SerialException as e:
    print(f"ERROR: Could not open {SERIAL_PORT}: {e}")
    list_serial_ports()
    exit(1)

# Open audio file
if not sf or not np:
    print("ERROR: soundfile and numpy required!")
    exit(1)

audio_file = sf.SoundFile(
    OUTPUT_WAV,
    mode="w",
    samplerate=AUDIO_SAMPLE_RATE,
    channels=1,
    subtype="PCM_16",
)
print(f"[AUDIO] Recording to {OUTPUT_WAV} at {AUDIO_SAMPLE_RATE} Hz")
print("Press Ctrl+C to stop...")

audio_samples_written = 0
tip_count = 0
serial_buffer = b""  # Buffer for incoming data
pcm_buffer = []  # Buffer for PCM samples before writing to file
WRITE_BATCH_SIZE = 1000  # Write samples in batches for performance

try:
    while True:
        # Read ALL available data at once (don't wait for just 2 bytes!)
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            serial_buffer += chunk

        # Process all complete 2-byte samples in buffer
        while len(serial_buffer) >= 2:
            high = serial_buffer[0]
            low = serial_buffer[1]
            serial_buffer = serial_buffer[2:]

            # Check for tip marker (0xFF 0xFF)
            if high == 0xFF and low == 0xFF:
                # Read tip count (2 more bytes needed)
                if len(serial_buffer) >= 2:
                    tip_low = serial_buffer[0]
                    tip_high = serial_buffer[1]
                    tip_count = tip_low | (tip_high << 8)
                    if tip_count < 10000:  # Validate
                        print(f"*** TIP DETECTED! Total tips: {tip_count} ***")
                        serial_buffer = serial_buffer[2:]
                        continue

            # Convert 10-bit ADC to 16-bit PCM
            adc_value = (high << 8) | low
            pcm_value = int((adc_value - DC_OFFSET) * 32.0)
            pcm_value = max(-32768, min(32767, pcm_value))

            # Add to PCM buffer
            pcm_buffer.append(pcm_value)
            audio_samples_written += 1

            # Write buffer to file in batches
            if len(pcm_buffer) >= WRITE_BATCH_SIZE:
                audio_file.write(np.array(pcm_buffer, dtype=np.int16))
                pcm_buffer = []

            # Report every 10000 samples (~1.25 seconds)
            if audio_samples_written % 10000 == 0:
                duration = audio_samples_written / AUDIO_SAMPLE_RATE
                print(
                    f"[AUDIO] Recorded {duration:.1f}s ({audio_samples_written} samples, {tip_count} tips)"
                )

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Write any remaining samples in buffer
    if pcm_buffer:
        audio_file.write(np.array(pcm_buffer, dtype=np.int16))

    ser.close()
    audio_file.close()

    duration = audio_samples_written / AUDIO_SAMPLE_RATE
    print(f"\n[AUDIO] Saved {duration:.1f}s ({audio_samples_written} samples)")
    print(f"[AUDIO] WAV: {OUTPUT_WAV}")

    if convert_wav_to_mp3(OUTPUT_WAV, OUTPUT_MP3, AUDIO_BITRATE):
        print(f"[AUDIO] MP3: {OUTPUT_MP3}")

    print("Done!")
