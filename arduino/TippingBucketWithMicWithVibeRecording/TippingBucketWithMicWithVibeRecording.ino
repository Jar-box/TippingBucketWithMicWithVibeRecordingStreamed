// -------- Sound Sensor --------
const int SOUND_PIN = A2;   // Micro MAX4466

// --- Sound ---
unsigned long soundWindowStart      = 0;      // début fenêtre 1 s
unsigned long lastSoundSampleMicros = 0;      // pour ~10 kHz

const unsigned long SOUND_SAMPLE_PERIOD_US = 100;   // 10 kHz sampling rate
const unsigned long SOUND_WINDOW_MS       = 33;   // 50 ms window (20 Hz updates) for faster streaming

unsigned long soundSamples = 0;
int signalMin;
int signalMax;

void resetSoundWindow(unsigned long nowMillis) {
  soundWindowStart = nowMillis;
  soundSamples = 0;
  signalMin = 1023;
  signalMax = 0;
}

// -------- Tipping Bucket --------
const int REED_SWITCH_PIN = 2;  // Direct connection to tipping bucket reed switch
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

void setup() {
  Serial.begin(115200);      // Debug USB

  // Setup reed switch with interrupt
  pinMode(REED_SWITCH_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(REED_SWITCH_PIN), reedSwitchISR, FALLING);
  
  delay(500);
  resetSoundWindow(millis());

  // CSV header
  Serial.println("ts_ms,sound_amp,tip_count,last_tip_dt_ms,reed_state");
}

void loop() {
  unsigned long nowMillis = millis();

  // Check for reed switch state changes
  int currentReedState = digitalRead(REED_SWITCH_PIN);
  if (currentReedState != lastReedState) {
    Serial.print("Pin state changed: ");
    Serial.println(currentReedState == HIGH ? "HIGH (open/no magnet)" : "LOW (closed/magnet detected)");
    lastReedState = currentReedState;
  }

  // --- Sound sampling ---
  unsigned long nowMicros = micros();
  if (nowMicros - lastSoundSampleMicros >= SOUND_SAMPLE_PERIOD_US) {
    lastSoundSampleMicros = nowMicros;
    int sample = analogRead(SOUND_PIN);
    soundSamples++;
    if (sample < signalMin) signalMin = sample;
    if (sample > signalMax) signalMax = sample;
  }

  // --- Sound amplitude calculation + CSV output ---
  if (nowMillis - soundWindowStart >= SOUND_WINDOW_MS) {
    int amplitude = (soundSamples > 0) ? (signalMax - signalMin) : 0;

    noInterrupts();
    int tipsSnapshot = bucketTips;
    unsigned long lastTipIntervalSnapshot = lastTipIntervalMs;
    interrupts();

    Serial.print(nowMillis);
    Serial.print(',');
    Serial.print(amplitude);
    Serial.print(',');
    Serial.print(tipsSnapshot);
    Serial.print(',');
    Serial.print(lastTipIntervalSnapshot);
    Serial.print(',');
    Serial.println(currentReedState == HIGH ? 1 : 0);

    resetSoundWindow(nowMillis);
  }
}
