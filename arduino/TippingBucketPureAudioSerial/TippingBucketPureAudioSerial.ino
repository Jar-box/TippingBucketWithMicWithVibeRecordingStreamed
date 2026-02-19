// ============================================================================
// PURE AUDIO STREAM - Serial version of TippingBucketWithMic_AudioStream.ino
// ============================================================================
// Streams ONLY audio via Serial (no CSV data)
// Tip markers: 0xFF 0xFF + tip count (2 bytes)
// Python: serial_pure_audio.py

// -------- Sound Sensor --------
const int SOUND_PIN = A2;   // Micro MAX4466

// --- Sound streaming settings ---
const unsigned long SOUND_SAMPLE_PERIOD_US = 100;   // ~10 kHz sampling (actual ~8kHz with overhead)
unsigned long lastSoundSampleMicros = 0;

// -------- Tipping Bucket --------
const int REED_SWITCH_PIN = 2;
volatile int bucketTips = 0;
volatile unsigned long lastTipTime = 0;

// Reed switch interrupt handler
void reedSwitchISR() {
  unsigned long now = millis();
  if (now - lastTipTime > 200) {  // Debounce 200ms
    bucketTips++;
    lastTipTime = now;
    // Send tip event in binary format: 0xFF 0xFF (marker) + tip count
    Serial.write(0xFF);
    Serial.write(0xFF);
    Serial.write(bucketTips & 0xFF);
    Serial.write((bucketTips >> 8) & 0xFF);
  }
}

void setup() {
  Serial.begin(230400);      // High speed for audio streaming

  // Setup reed switch with interrupt
  pinMode(REED_SWITCH_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(REED_SWITCH_PIN), reedSwitchISR, FALLING);
  
  delay(500);
}

void loop() {
  unsigned long nowMicros = micros();
  
  // Sample and stream audio at ~8 kHz  
  if (nowMicros - lastSoundSampleMicros >= SOUND_SAMPLE_PERIOD_US) {
    lastSoundSampleMicros = nowMicros;
    
    // Read and send raw audio sample (0-1023, 10-bit ADC)
    int sample = analogRead(SOUND_PIN);
    
    // Send as 2 bytes: high byte, low byte
    Serial.write((sample >> 8) & 0xFF);  // High byte
    Serial.write(sample & 0xFF);         // Low byte
  }
}
