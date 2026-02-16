# Camera Integration Setup Guide

## Overview

`multicast_with_camera.py` adds real-time IP camera monitoring to your rainfall detection system. The camera feed displays side-by-side with the mic amplitude graph for visual correlation analysis.

---

## Prerequisites

### Python Dependencies

Install OpenCV before running:

```bash
pip install opencv-python
```

Or if you use conda:

```bash
conda install -c conda-forge opencv
```

---

## Android IP Camera Apps

Choose one of these popular apps to turn your Android phone into an IP camera:

### 1. **IP Webcam** (Recommended)

- **Download**: [Google Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam)
- **Default URL format**: `http://192.168.1.XXX:8080/video`
- **Features**: MJPEG stream, multiple resolutions, night mode
- **Setup**:
  1. Install app on Android phone
  2. Open app and scroll to bottom
  3. Tap "Start server"
  4. Note the IP address shown (e.g., `192.168.1.105:8080`)
  5. Use URL: `http://192.168.1.105:8080/video`

### 2. **DroidCam**

- **Download**: [Google Play Store](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
- **Default URL format**: `http://192.168.1.XXX:4747/video`
- **Features**: HD streaming, low latency

### 3. **RTSP Camera Server**

- **Download**: [Google Play Store](https://play.google.com/store/apps/details?id=com.spynet.camera)
- **Default URL format**: `rtsp://192.168.1.XXX:8554/unicast`
- **Features**: RTSP protocol, better for long-distance streaming

---

## Configuration Steps

### Step 1: Find Your Camera's IP Address

**On Android Phone:**

1. Connect to your local WiFi network (same network as Raspberry Pi)
2. Go to Settings → About Phone → Status → IP Address
3. Note the IP (e.g., `192.168.1.105`)

**Or use IP Webcam app:**

- The app displays the IP address when you start the server

### Step 2: Update Camera URL in Script

Open `multicast_with_camera.py` and find line ~25:

```python
CAMERA_URL = "http://192.168.1.100:8080/video"  # CHANGE THIS
```

**Update with your camera's URL:**

```python
# For IP Webcam:
CAMERA_URL = "http://192.168.1.105:8080/video"

# For DroidCam:
CAMERA_URL = "http://192.168.1.105:4747/video"

# For RTSP:
CAMERA_URL = "rtsp://192.168.1.105:8554/unicast"
```

### Step 3: Test Camera Stream

Before running the full script, test the camera connection:

```python
import cv2

# Replace with your camera URL
url = "http://192.168.1.105:8080/video"

cap = cv2.VideoCapture(url)
ret, frame = cap.read()

if ret:
    print("✓ Camera connected successfully!")
    print(f"Frame shape: {frame.shape}")
else:
    print("✗ Failed to connect to camera")

cap.release()
```

---

## Running the Script

### Basic Usage

```bash
python multicast_with_camera.py
```

### Disable Camera (Troubleshooting)

If you want to run without camera temporarily, edit line ~26:

```python
CAMERA_ENABLED = False  # Disables camera, runs graph only
```

---

## Display Layout

```
┌─────────────────────────────────────────────┐
│  Live Camera Feed  │  Mic Amplitude Graph   │
│                    │                         │
│  [Timestamp        │  [Real-time plot]      │
│   overlay]         │                         │
│                    │  • Green: RAW          │
│                    │  • Orange: SMOOTHED    │
│                    │  • Red lines: Bucket   │
│                    │    tips                │
└─────────────────────────────────────────────┘
```

---

## Performance Optimization for Raspberry Pi

### Recommended Settings

```python
# Line ~44 - Reduce update frequency
UPDATE_EVERY_N_SAMPLES = 15  # Default is 10, higher = less CPU

# Line ~51 - Smaller figure size
FIGURE_SIZE = (14, 6)  # Default is (16, 7)
```

### Camera Resolution

**In IP Webcam app:**

- Video preferences → Resolution → Use 640x480 or 800x600
- Lower resolution = smoother performance on Pi

### Additional Tips

1. **Use wired Ethernet** on Raspberry Pi if possible (reduces WiFi congestion)
2. **Close other applications** to free up CPU/memory
3. **Reduce video quality** in camera app if stream is choppy
4. **Position phone closer** to WiFi router for better signal

---

## Troubleshooting

### Camera Not Connecting

**Check 1: Same Network**

```bash
# On Raspberry Pi, ping the phone
ping 192.168.1.105
```

**Check 2: Firewall**

- Some routers block multicast/streaming
- Try connecting laptop to same network and test URL in browser

**Check 3: App Running**

- Ensure IP Webcam server is started (not just app open)
- Phone screen should show "Streaming" status

### Camera Connected But Black Screen

- Check camera permissions in Android settings
- Restart the camera app
- Try different URL path (e.g., `/video` vs `/video.mjpeg`)

### Low Frame Rate

- Reduce camera resolution in app settings
- Increase `UPDATE_EVERY_N_SAMPLES` to 20 or 30
- Use MJPEG instead of RTSP for lower latency

### Script Crashes

```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Reinstall if needed
pip install --upgrade opencv-python
```

---

## Camera URL Examples

### IP Webcam (Most Common)

```python
# Standard video feed
CAMERA_URL = "http://192.168.1.105:8080/video"

# MJPEG format (sometimes more compatible)
CAMERA_URL = "http://192.168.1.105:8080/video.mjpeg"

# With authentication (if enabled in app)
CAMERA_URL = "http://username:password@192.168.1.105:8080/video"
```

### DroidCam

```python
CAMERA_URL = "http://192.168.1.105:4747/video"
```

### RTSP Stream

```python
CAMERA_URL = "rtsp://192.168.1.105:8554/unicast"

# With authentication
CAMERA_URL = "rtsp://username:password@192.168.1.105:8554/unicast"
```

---

## Features

✓ **Side-by-side display**: Camera + real-time graph  
✓ **Timestamp overlay**: Synchronized frame timestamps  
✓ **Auto-reconnect**: Handles network interruptions  
✓ **Thread-safe**: Non-blocking camera capture  
✓ **Original features preserved**: All mic processing intact  
✓ **Raspberry Pi optimized**: Adjustable performance settings

---

## Original Files (Unchanged)

Your original working files remain untouched:

- `multicast.py` ✓
- `multicast_optimized.py` ✓

You can always revert to these if camera integration causes issues.

---

## Questions?

Feel free to adjust these settings based on your specific hardware and network conditions!
