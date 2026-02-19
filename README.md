# TippingBucketWithMicWithVibeRecordingStreamed

## Important Files

### Main Python Scripts

- **multicast.py** - For good/high-performance machines. Full-featured multicast implementation.
- **multicast_optimized.py** - For machines that can't handle multicast.py. Optimized version with reduced resource usage.
- **serial_optimized.py** - Based on multicast_optimized.py but connects directly to Arduino via USB instead of using multicast from RPi. **Currently includes MP3 recording from MAX44466** (this feature needs to be applied to other files in the future).

All three files have the same function but differ in execution method.

### Additional Files

- **multicast_with_camera.py** - Based on multicast_optimized.py with camera integration. Currently not in use due to high resource requirements for current machines.
- **TippingBucketWithMicWithVibeRecording.ino** - Arduino code. Does not require frequent modification.
- **requirements.txt** - Python dependencies for the project.
- **CAMERA_SETUP_GUIDE.md** - Camera setup documentation (not currently in use).

### Experimental Audio Recording Files (Currently Dropped)

The following files were created for an attempt to record audio and save MP3 and WAV files:

- **serial_pure_audio.py** - Pure audio recording without graph plotting
- **serial_with_audio.py** - Audio recording with graph plotting
- **arduino/TippingBucketPureAudioSerial/TippingBucketPureAudioSerial.ino** - Arduino code for pure audio recording
- **arduino/TippingBucketSerialAudio/TippingBucketSerialAudio.ino** - Arduino code for audio recording with data

**Difference between "pure" and "non-pure" versions:** The "pure" files only record audio without plotting graphs, while the "non-pure" versions include graph plotting functionality.

**Status:** This feature has been dropped for now due to:

- Poor audio quality from the microphone (crunchy sound)
- Time constraints - requires more development time to polish
- Need to prioritize documentation finalization
- May be revisited in the future if required

## Output Files

The system currently records data in the following formats:

- PNG (images)
- HTML (reports)
- CSV (data)
- MP3 (audio recordings)

## Data Organization

Data is stored in the `data/` folder with the following structure:

```
data/
├── YYYY-MM-DD/          (date folders)
│   ├── HH/              (hour folders)
│   │   ├── rain_intensity_*.csv
│   │   ├── rain_intensity_*.html
│   │   ├── rain_intensity_*.png
│   │   └── rain_intensity_*.mp3
```

## TODO

- [ ] Apply MP3 recording feature from serial_optimized.py to multicast.py and multicast_optimized.py
