# Tipping Bucket Technical Specifications

## Hardware Configuration

### Tipping Bucket Rain Gauge

- **Volume per tip**: 7.4 mL
- **Rain depth per tip**: 0.024 mm
- **Funnel area**: 314.16 cm²
- **Funnel discharge rate**: 4.11 mL/s

## Performance Characteristics

### Tip Rate Calculations

- **Average discharge duration**: 54.71 s
  - Time for siphon to fully discharge collected water
- **Average time to first tip**: 2.01 s
  - Time from when rain starts pouring into bucket until first tip occurs
  - This represents the accumulation time needed to collect 7.4 mL
- **Minimum tip interval**: 1.80 s/tip
  - Calculation: 7.4 mL ÷ 4.11 mL/s = 1.80 s
- **Maximum tip rate**: 0.556 tip/s
  - Calculation: 1 ÷ 1.80 s = 0.556 tip/s
- **Maximum measurable intensity (theoretical)**: 48.04 mm/hr
  - Calculation: 0.556 tip/s × 0.024 mm/tip × 3600 s/hr = 48.04 mm/hr
  - OR: (3600 s/hr ÷ 1.80 s/tip) × 0.024 mm/tip = 48.00 mm/hr

## Important Considerations for Rain Detection

### Rain End Detection Complexity

**Why rain end detection is difficult:**

The tipping bucket includes a siphon mechanism that continues to drain collected water even after rainfall has stopped. This means:

1. **Delayed response**: Tips may continue for several seconds after rain actually ends
2. **Siphon drainage time**: The siphon needs time to fully drain its contents
3. **Cannot rely on tips alone**: Absence of tips doesn't immediately indicate rain has ended

**Implications for software:**

- Rain end detection must account for siphon drainage time
- Combining mic amplitude data with tip timing provides more accurate rain end detection
- A "cooldown period" after the last tip is necessary before declaring rain has ended
- Recommended cooldown: 30-60 seconds with no tips AND mic amplitude at baseline

### Rain Start Detection

Rain start detection is more straightforward:

1. **Mic amplitude rise**: Detects initial raindrops on sensor surface
2. **First tip confirmation**: Validates that sufficient water has accumulated (average 1.7s delay)
3. **Combined approach**: Mic provides early warning, tip provides confirmation

**Timing considerations:**

- Microphone detects rain immediately upon impact
- First bucket tip occurs ~2.01 seconds after rain starts (accumulation time)
- This ~2 second gap justifies using mic for early rain start detection

## Sensor Specifications

### Microphone (Sound/Vibration Sensor)

- **Type**: Analog amplitude sensor
- **Output range**: 0-1023 (Arduino ADC)
- **Purpose**: Detect raindrop impacts on bucket surface
- **Advantages**:
  - Fast response time
  - Detects rain before bucket tips
  - Sensitive to light rain that may not cause tips

### Reed Switch (Tip Counter)

- **Type**: Magnetic reed switch
- **Output**: Digital (HIGH/LOW)
- **Purpose**: Count bucket tips to measure rainfall volume
- **Advantages**:
  - Accurate volume measurement
  - No calibration drift
  - Simple and reliable

## System Limitations

1. **Maximum intensity**: 48.04 mm/hr (above this, bucket cannot drain fast enough)
2. **Minimum detectable rain**: Single raindrop (via mic), 0.024 mm (via tip)
3. **Timing resolution**: Limited by siphon discharge rate (1.80s minimum interval)
4. **End detection delay**: 30-60 seconds due to siphon drainage (average discharge duration: 54.71s)

## Calibration Notes

- **Bucket volume**: Verify 7.4 mL per tip through water volume testing
- **Siphon rate**: Measured at 4.11 mL/s under standard conditions
- **Discharge duration**: Average 54.71 seconds for complete drainage
- **Mic baseline**: Varies with ambient noise; requires dynamic baseline tracking
- **Tip validation**: Reed switch bounce filtering in Arduino code (already implemented)

## Future Enhancements

- [ ] Implement dynamic siphon drainage modeling
- [ ] Add rain intensity classification (light/moderate/heavy)
- [ ] Develop predictive rain end algorithm using mic decay patterns
- [ ] Add temperature compensation for viscosity effects on siphon rate
