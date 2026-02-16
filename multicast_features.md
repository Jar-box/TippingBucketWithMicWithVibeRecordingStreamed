# Multicast Rain Sensor Data Logger - Feature Documentation

## Overview
Python script that receives UDP multicast data from an Arduino-based rain sensor system with tipping bucket and microphone. Provides real-time visualization and data logging of rain events and sound amplitude.

---

## Configuration Parameters

### Timing & Window Settings
- **WINDOW_SECONDS** = 30
  - Rolling time window for live plot display (seconds)
  
- **RESET_INTERVAL** = 60
  - Interval for resetting tip count rate calculations (seconds)
  
- **MIC_SUPPRESSION_SECONDS** = 0.35
  - Duration to suppress microphone readings after a bucket tip to avoid mechanical noise

### Sensor Calibration
- **TIP_MM_PER_TIP** = 100
  - Typical tipping bucket calibration (mm of rain per tip)
  
- **MIC_DB_REF** = 1.0
  - Reference amplitude for decibel calculation (prevents -inf values)

### Network Settings
- **MCAST_GRP** = "230.138.19.201"
  - Multicast group IP address
  
- **MCAST_PORT** = 5007
  - UDP port for multicast reception
  
- **IS_ALL_GROUPS** = True
  - If True: receives ALL multicast groups on the port
  - If False: listens ONLY to MCAST_GRP

### Socket Configuration
- Timeout: 2.0 seconds
- Socket options: SO_REUSEADDR enabled
- Multicast membership: IP_ADD_MEMBERSHIP

---

## Data Structures

### Live Display Deques (30-second rolling window)
- `series_t` - Timestamps for live display
- `series_bucket` - Bucket tip rates (mm/hr)
- `series_mic_raw` - Unfiltered mic levels (dB)
- `series_mic_filtered` - Filtered mic levels (dB)
- `series_mic_amp_raw` - Unfiltered mic amplitude (raw ADC values)
- `series_mic_amp_filtered` - Filtered mic amplitude
- `tip_times` - Timestamps of bucket tips within window

### Archive Deques (ALL data for final PNG)
- `archive_t` - All timestamps
- `archive_bucket` - All bucket rates
- `archive_mic_raw` - All unfiltered mic levels (dB)
- `archive_mic_filtered` - All filtered mic levels (dB)
- `archive_mic_amp_raw` - All unfiltered mic amplitudes
- `archive_mic_amp_filtered` - All filtered mic amplitudes
- `archive_tip_times` - All tip timestamps

### State Variables
- `last_reset_time` - Timestamp of last rate reset
- `tip_count_at_reset` - Tip count at last reset
- `mic_amp_sum_at_reset` - Mic amplitude sum at reset
- `sample_count_since_reset` - Sample counter
- `previous_tip_count` - Previous tip count for edge detection
- `mic_suppressed_until` - Timestamp when mic suppression ends
- `mic_history` - Deque of recent dB values (maxlen=10)
- `mic_amp_history` - Deque of recent amplitude values (maxlen=10)

---

## Input Data Format

### Expected CSV Line from Arduino
```
ts_ms,sound_amp,tip_count,last_tip_dt_ms,reed_state
```

**Fields:**
1. `arduino_ms` (int) - Arduino timestamp in milliseconds
2. `sound_amp` (int) - Microphone amplitude (ADC value, 0-1023 typical)
3. `tip_count` (int) - Cumulative tip count since Arduino boot
4. `last_tip_dt_ms` (int) - Time since last tip (milliseconds)
5. `reed_state` (int) - Reed switch state (0 or 1)

---

## Data Processing Logic

### 1. Tip Rate Calculation
- **Spike on New Tip**: When `tip_count` increments, set `bucket_rate = 40000.0` mm/hr (fixed spike for visibility)
- **No New Tip**: `bucket_rate = 0.0`
- **Rate Reset**: Every RESET_INTERVAL seconds, reset tip counter baseline

### 2. Microphone Level (dB)
**Unfiltered:**
```python
mic_rate_unfiltered = 20.0 * log10(max(sound_amp, 1) / MIC_DB_REF)
```

**Filtered:**
- Normally equals unfiltered value
- During suppression window (MIC_SUPPRESSION_SECONDS after tip):
  - Use average of past 10 samples from `mic_history`
  - Prevents mechanical noise spikes

### 3. Microphone Amplitude
**Unfiltered:**
```python
mic_amp_unfiltered = sound_amp  # Raw ADC value
```

**Filtered:**
- Normally equals unfiltered value
- During suppression window:
  - Use average of past 10 samples from `mic_amp_history`
  - Smooth out bucket-tip mechanical transients

### 4. Suppression Mechanism
When new tip detected:
```python
mic_suppressed_until = now_ts + MIC_SUPPRESSION_SECONDS
tip_times.append(now_ts)
archive_tip_times.append(now_ts)
```

### 5. Window Pruning
- `prune_old()` function removes data older than WINDOW_SECONDS
- Only affects live display deques
- Archive deques keep ALL data

---

## Output Files

### CSV Log File
**Filename:** `rain_intensity_YYYYMMDD_HHMMSS.csv`

**Columns:**
1. `elapsed_s` - Elapsed time since start (seconds)
2. `unix_ts` - Unix timestamp (seconds)
3. `arduino_ms` - Arduino timestamp (milliseconds)
4. `sound_amp` - Raw microphone amplitude (ADC)
5. `tip_count` - Cumulative tip count
6. `last_tip_dt_ms` - Time since last tip (ms)
7. `reed_state` - Reed switch state (0/1)
8. `bucket_rate_mm_hr` - Bucket tip rate (mm/hr, spike value)
9. `mic_db_unfiltered` - Unfiltered mic level (dB)
10. `mic_db_filtered` - Filtered mic level (dB)

**Flushed:** After every data row (real-time logging)

### PNG Plot File
**Filename:** `rain_intensity_YYYYMMDD_HHMMSS.png`

**Saved:** On script exit (Ctrl+C or window close)

**Content:** Archive (full-session) data with trend lines

---

## Visualization

### Live Display (Interactive, 30s Window)
**Figure Layout:** 2x2 subplots, 14x9 inches

#### Top-Left: Unfiltered Mic (dB)
- Purple line: `mic_rate_unfiltered`
- Y-axis: symlog scale (linthresh=1.0), range 0-100
- Blue vertical lines: bucket tip events

#### Top-Right: Filtered Mic (dB)
- Orange line: `mic_rate_filtered`
- Y-axis: symlog scale (linthresh=1.0), range 0-100
- Blue vertical lines: bucket tip events

#### Bottom-Left: Unfiltered Mic (Amplitude)
- Green line: `mic_amp_unfiltered`
- Y-axis: linear scale, autoscaled
- Blue vertical lines: bucket tip events

#### Bottom-Right: Filtered Mic (Amplitude)
- Teal line: `mic_amp_filtered`
- Y-axis: linear scale, autoscaled
- Blue vertical lines: bucket tip events

**Update Rate:** `plt.pause(0.001)` - minimal pause for fast updates

**X-Axis:** Relative time (seconds) from oldest sample in window

### Archive Plot (Final PNG, Full Session)
**Figure Layout:** Same 2x2 layout as live display

**Additional Features:**
- **Trend Lines** (moving average, window=5):
  - Unfiltered dB: Purple solid line
  - Filtered dB: Orange solid line
  - Unfiltered Amplitude: Green solid line
  - Filtered Amplitude: Teal solid line
- Original data plotted with alpha=0.6 (semi-transparent)
- Trend lines plotted with linewidth=2 (bold)
- All bucket tips marked with blue vertical lines (alpha=0.35)

**Saved:** 150 DPI, tight bounding box

---

## Trend Line Calculation

### Function: `compute_trend_line(data, window=20)`
**Algorithm:** Simple moving average

```python
for i in range(len(data)):
    start = max(0, i - window // 2)
    end = min(len(data), i + window // 2 + 1)
    trend[i] = sum(data[start:end]) / (end - start)
```

**Used For:** Archive plots only (window=5)

**Purpose:** Smooth out noise and show overall trends in mic data

---

## Plot Update Cycle

1. **Receive UDP packet**
2. **Parse CSV line** (skip headers, handle errors)
3. **Process data:**
   - Detect new tips
   - Calculate rates
   - Apply mic suppression filtering
   - Compute dB levels
4. **Update state:**
   - Append to live deques
   - Append to archive deques
   - Prune old data from live deques
5. **Log to CSV** (flush immediately)
6. **Update live plot:**
   - Update line data
   - Remove old tip lines
   - Redraw tip lines for current window
   - Autoscale axes
   - Set X-axis limits
   - `plt.pause(0.001)`

---

## Error Handling

### Socket Timeout
- 2-second timeout on `sock.recv()`
- Timeout raises KeyboardInterrupt to exit gracefully

### Parse Errors
- Skip lines starting with "ts_ms" (CSV header)
- Skip lines with less than 5 fields
- Catch ValueError on int conversions
- Print error message: `[PARSE ERROR] {line}`

### Graceful Shutdown
- KeyboardInterrupt (Ctrl+C) caught
- Window close event monitored
- `is_closed` flag prevents further processing
- Final archive plot generated before exit

---

## Console Output

### Startup
```
Running.
Reading serial... Press Ctrl+C to stop
```

### Periodic Debug (every 5 seconds)
```
[{elapsed}s] amp={sound_amp}, tips={tip_count}, delta_tips={tips_since_reset}, dt={last_tip_dt_ms}ms
```

### Reset Events
```
[RESET] Tip counter reset at {elapsed}s, tips={tip_count}
```

### Shutdown
```
Stopping...
Done! CSV saved to {OUTPUT_CSV}
Plot saved to {OUTPUT_PNG}
```

---

## Key Features Summary

1. **Real-time UDP multicast data reception** from Arduino sensor
2. **Dual data storage:** Live (30s window) + Archive (full session)
3. **Tip detection:** Edge detection on cumulative tip count
4. **Mic filtering:** Automatic suppression after bucket tips to reduce mechanical noise
5. **Dual visualization:** dB (symlog) and amplitude (linear) representations
6. **Historical averaging:** 10-sample rolling average for filtered channels
7. **Interactive live plot:** 30-second rolling window, auto-scaling
8. **Final archive plot:** Full session data with trend lines
9. **Continuous CSV logging:** Real-time data export with flush
10. **Graceful shutdown:** Archive plot saved on exit

---

## Dependencies

```python
import socket
import struct
import csv
import math
import time
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
```

---

## Usage Pattern

1. **Start Arduino sensor** broadcasting on multicast address
2. **Run script:** `python multicast.py`
3. **Monitor live plot** (30-second window)
4. **Press Ctrl+C** when done
5. **Review:**
   - CSV file for detailed data
   - PNG file for visual summary with trends

---

## Notes for Optimization

When applying changes to `multicast_optimized.py`:
- Maintain all configuration parameters
- Preserve data processing logic (tip detection, dB calculation, filtering)
- Keep dual storage system (live + archive)
- Ensure CSV output format remains identical
- Archive plot with trends must be generated on exit
- Mic suppression mechanism is critical for data quality
- All console output messages should be preserved for debugging
