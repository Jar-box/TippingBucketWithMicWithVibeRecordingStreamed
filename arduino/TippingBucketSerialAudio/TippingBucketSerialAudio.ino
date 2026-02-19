// ============================================================================
// SERIAL AUDIO VERSION - Sends audio samples via USB/Serial
// ============================================================================
// This version streams both sensor data (CSV) and raw audio samples via Serial
// Requires higher baud rate: 230400 baud
// Python script: serial_with_audio.py

// -------- Sound Sensor --------
const int SOUND_PIN = A2;   // Micro MAX4466

// --- Audio Streaming ---
const unsigned long SOUND_SAMPLE_PERIOD_US = 100;   // ~10 kHz sampling rate (stable for UNO, actual ~8kHz with overhead)
const unsigned long STATUS_REPORT_MS = 50;          // Send status every 50ms (20 Hz updates)
const int AUDIO_BUFFER_SIZE = 400;                  // Buffer 400 samples before sending (~50ms at 8kHz)

int audioBuffer[AUDIO_BUFFER_SIZE];
int audioBufferIndex = 0;
unsigned long lastSoundSampleMicros = 0;
unsigned long lastStatusReportMs = 0;

// For amplitude calculation (running min/max)
int signalMin = 512;
int signalMax = 512;
unsigned long amplitudeWindowStart = 0;

// -------- Tipping Bucket --------
const int REED_SWITCH_PIN = 2;
volatile int bucketTips = 0;
volatile unsigned long lastTipTime = 0;
volatile unsigned long lastTipIntervalMs = 0;
int lastReedState = HIGH;

// Reed switch interrupt handler
void reedSwitchISR() {
  unsigned long now = millis();
  if (now - lastTipTime > 200) {  // Debounce 200ms
    if (lastTipTime > 0) {
      lastTipIntervalMs = now - lastTipTime;
    }
    bucketTips++;
    lastTipTime = now;
  }
}

// Send audio buffer as binary data with marker
void sendAudioBuffer() {
  if (audioBufferIndex == 0) return;
  
  // Protocol: "AUDIO:" + 2-byte sample count + binary samples (2 bytes each, big-endian)
  Serial.print("AUDIO:");
  Serial.write((byte)(audioBufferIndex >> 8));  // High byte of count
  Serial.write((byte)(audioBufferIndex & 0xFF)); // Low byte of count
  
  // Send each sample as 2 bytes (10-bit ADC value)
  for (int i = 0; i < audioBufferIndex; i++) {
    int sample = audioBuffer[i];
    Serial.write((byte)(sample >> 8));   // High byte
    Serial.write((byte)(sample & 0xFF));  // Low byte
  }
  Serial.println();  // Newline delimiter
  
  audioBufferIndex = 0;
}

void setup() {
  Serial.begin(230400);  // Higher baud rate for audio streaming

  // Setup reed switch with interrupt
  pinMode(REED_SWITCH_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(REED_SWITCH_PIN), reedSwitchISR, FALLING);
  
  delay(500);
  
  signalMin = 512;
  signalMax = 512;
  amplitudeWindowStart = millis();
  lastStatusReportMs = millis();

  // Identification header
  Serial.println("SERIAL_AUDIO_MODE:8000Hz");
  Serial.println("ts_ms,sound_amp,tip_count,last_tip_dt_ms,reed_state");
}

void loop() {
  unsigned long nowMillis = millis();
  unsigned long nowMicros = micros();

  // --- Sound sampling at ~8 kHz ---
  if (nowMicros - lastSoundSampleMicros >= SOUND_SAMPLE_PERIOD_US) {
    lastSoundSampleMicros = nowMicros;
    int sample = analogRead(SOUND_PIN);
    
    // Buffer audio sample
    if (audioBufferIndex < AUDIO_BUFFER_SIZE) {
      audioBuffer[audioBufferIndex++] = sample;
    }
    
    // Track min/max for amplitude calculation
    if (sample < signalMin) signalMin = sample;
    if (sample > signalMax) signalMax = sample;
    
    // Send buffer when full
    if (audioBufferIndex >= AUDIO_BUFFER_SIZE) {
      sendAudioBuffer();
    }
  }

  // --- Status report every 100ms ---
  if (nowMillis - lastStatusReportMs >= STATUS_REPORT_MS) {
    lastStatusReportMs = nowMillis;
    
    // Calculate amplitude from min/max
    int amplitude = signalMax - signalMin;
    
    // Reset amplitude window
    signalMin = 512;
    signalMax = 512;
    
    // Get tip data
    noInterrupts();
    int tipsSnapshot = bucketTips;
    unsigned long lastTipIntervalSnapshot = lastTipIntervalMs;
    int reedState = digitalRead(REED_SWITCH_PIN);
    interrupts();
    
    // Send any pending audio before status
    sendAudioBuffer();
    
    // Send CSV status line
    Serial.print(nowMillis);
    Serial.print(',');
    Serial.print(amplitude);
    Serial.print(',');
    Serial.print(tipsSnapshot);
    Serial.print(',');
    Serial.print(lastTipIntervalSnapshot);
    Serial.print(',');
    Serial.println(reedState == HIGH ? 1 : 0);
  }
}
