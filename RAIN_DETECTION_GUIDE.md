# Rain Detection System Guide

## Overview

The rain detection system uses a **state machine** that combines microphone amplitude data with tipping bucket events to automatically detect rain start and end times.

## How It Works

### State Machine

The system operates in 4 states:

1. **NO_RAIN** - Monitoring for rain, calculating baseline
2. **RAIN_DETECTED** - Mic threshold crossed, waiting for confirmation
3. **RAIN_CONFIRMED** - Active rainfall detected
4. **RAIN_ENDING** - Rain stopping, waiting for siphon drainage

### Detection Flow

```
NO_RAIN
   │
   ├─► Mic amplitude > threshold for 3+ seconds
   │
   ▼
RAIN_DETECTED
   │
   ├─► Tip occurs within 5s → RAIN_CONFIRMED
   ├─► No tip but mic elevated 10s → RAIN_CONFIRMED (light rain)
   └─► Mic drops → NO_RAIN (false alarm)
   
RAIN_CONFIRMED
   │
   ├─► Mic returns to baseline for 10s → RAIN_ENDING
   └─► Mic stays elevated → Continue monitoring
   
RAIN_ENDING
   │
   ├─► 45 seconds with no tips + mic at baseline → NO_RAIN (rain ended)
   ├─► New tip or mic rises → RAIN_CONFIRMED (rain resumed)
   └─► Countdown continues
```

## Configuration Parameters

Located in [multicast.py](multicast.py):

### Baseline Tracking
- `BASELINE_WINDOW = 60` - seconds to track for baseline calculation
- `BASELINE_MIN_SAMPLES = 20` - minimum samples needed to establish baseline

### Rain Start Detection
- `START_THRESHOLD_OFFSET = 100` - amplitude above baseline to trigger detection
- `START_THRESHOLD_MULTIPLIER = 1.5` - alternative threshold (baseline × 1.5)
- `START_MIN_ELEVATED_TIME = 3.0` - seconds mic must stay elevated
- `START_CONFIRMATION_TIME = 5.0` - seconds to wait for tip (expected ~1.7s)
- `START_NO_TIP_TIMEOUT = 10.0` - seconds before confirming light rain without tip

### Rain End Detection
- `END_THRESHOLD_OFFSET = 20` - amplitude near baseline to trigger end detection
- `END_MIC_BASELINE_TIME = 10.0` - seconds mic must be at baseline before countdown
- `END_CONFIRMATION_TIME = 45.0` - seconds to wait for siphon drainage completion

## Why These Values?

### Start Detection (3-5 seconds)
- **3s minimum**: Filters out single raindrop impacts or noise spikes
- **5s tip confirmation**: Accommodates normal 1.7s average + buffer for light rain
- **10s no-tip timeout**: Light rain may not cause tips immediately

### End Detection (45 seconds)
- **Based on hardware**: Siphon minimum tip interval = 2.74s
- **Safety margin**: 45s allows ~16 potential tip cycles to complete
- **Prevents false ends**: Accounts for intermittent rain patterns

## Output

### Console Logging

The system prints events to console:

```
[RAIN DETECTED] at 45.2s - Mic: 250 > Threshold: 180 (baseline: 80)
[RAIN CONFIRMED] at 46.8s - Tip detected, total tips: 5

[RAIN ENDING] at 128.4s - Mic at baseline, waiting for siphon drainage...

[RAIN ENDED] at 173.4s
  Duration: 126.6s (2.1 min)
  Total tips: 12
  Rainfall: 0.30 mm
```

### CSV Data

Added columns:
- `rain_state` - Current state name (NO_RAIN, RAIN_DETECTED, etc.)
- `baseline_amp` - Calculated baseline amplitude

### Visual Indicators

#### Live Plot
- **Light blue shading** - Active rain periods
- Shows current rain event and recent completed events within 30s window

#### Archive PNG
- **Light blue shading** - All rain periods
- **Text annotations** - Rainfall amount and duration on each event
- Example: `0.30mm\n2.1min`

#### Interactive HTML
- **Light blue rectangles** - Rain periods with hover annotations
- Click and drag to zoom, double-click to reset
- Hover for detailed information

### End-of-Session Summary

Printed when the program stops:

```
============================================================
RAIN EVENT SUMMARY
============================================================

Event 1:
  Start: 14:32:45
  End:   14:35:52
  Duration: 3.1 minutes (187s)
  Tips: 8
  Rainfall: 0.20 mm
  Avg Intensity: 3.9 mm/hr

Event 2:
  Start: 14:45:12
  End:   14:58:33
  Duration: 13.4 minutes (801s)
  Tips: 45
  Rainfall: 1.13 mm
  Avg Intensity: 5.1 mm/hr
============================================================
```

## Tuning the System

### If detecting too early (false positives):
1. Increase `START_THRESHOLD_OFFSET` (e.g., 150)
2. Increase `START_MIN_ELEVATED_TIME` (e.g., 5.0s)
3. Decrease `START_NO_TIP_TIMEOUT` (e.g., 8.0s)

### If detecting too late:
1. Decrease `START_THRESHOLD_OFFSET` (e.g., 75)
2. Decrease `START_MIN_ELEVATED_TIME` (e.g., 2.0s)

### If ending too early:
1. Increase `END_CONFIRMATION_TIME` (e.g., 60.0s)
2. Increase `END_MIC_BASELINE_TIME` (e.g., 15.0s)

### If ending too late:
1. Decrease `END_CONFIRMATION_TIME` (e.g., 30.0s)
2. Decrease `END_MIC_BASELINE_TIME` (e.g., 5.0s)

## Troubleshooting

### Baseline not establishing
**Symptom**: System stays in NO_RAIN but never detects rain
**Solution**: 
- Ensure at least 20 samples of quiet period at startup
- Check if ambient noise is too high
- Lower `START_THRESHOLD_OFFSET`

### False rain detection
**Symptom**: Rain detected during non-rain events (wind, vibration)
**Solution**:
- Increase `START_MIN_ELEVATED_TIME`
- Require tip confirmation by decreasing `START_NO_TIP_TIMEOUT`
- Check physical sensor mounting for vibration

### Rain doesn't end
**Symptom**: System stuck in RAIN_CONFIRMED or RAIN_ENDING
**Solution**:
- Check if baseline is accurate (printed in console)
- Verify mic is actually returning to quiet levels
- May need to recalibrate `END_THRESHOLD_OFFSET`

### Multiple short events instead of one long event
**Symptom**: Intermittent rain registers as separate events
**Solution**:
- Increase `END_MIC_BASELINE_TIME` to be more tolerant
- Increase `END_CONFIRMATION_TIME` to allow longer gaps

## Technical Details

### Baseline Calculation
- Uses **median** of recent samples (not mean) to avoid outlier influence
- Only updated during NO_RAIN state to prevent rain from affecting baseline
- Recalculated after each rain event ends

### Threshold Calculation
```python
start_threshold = max(
    baseline + START_THRESHOLD_OFFSET,
    baseline * START_THRESHOLD_MULTIPLIER
)
```
Uses whichever is larger to handle both quiet and noisy environments.

### Siphon Drainage Consideration
The 45-second end confirmation accounts for:
- Minimum tip interval: 2.74s (siphon discharge rate)
- Buffer for water still draining: ~40s
- Ensures last tip has fully drained before declaring rain ended

## Integration with Existing Features

### Compatible with:
- ✓ Exponential smoothing (uses `smoothed_mic_amp`)
- ✓ Tip suppression (works during suppression periods)
- ✓ CSV logging (adds rain state columns)
- ✓ PNG and HTML generation (adds visual shading)
- ✓ All existing multicast features

### Does not interfere with:
- ✓ Real-time plotting
- ✓ Tip line markers
- ✓ Time axis formatting
- ✓ Audio recording features

## Future Enhancements

See [TIPPING_BUCKET_SPECS.md](TIPPING_BUCKET_SPECS.md) for planned improvements:
- [ ] Dynamic siphon drainage modeling
- [ ] Rain intensity classification (light/moderate/heavy)
- [ ] Predictive rain end using mic decay patterns
- [ ] Temperature compensation for viscosity effects
