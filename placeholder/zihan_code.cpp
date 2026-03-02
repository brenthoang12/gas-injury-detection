#include <Wire.h>
#include <Adafruit_HDC302x.h>
 
#define PIN_HCHO      34
#define PIN_VOC       35
#define PIN_NH3       32
#define PIN_BUTTON    18
#define PIN_ETOH_GAS  26
 
#define DISCONNECTED_V  0.1f
#define WARMUP_MS       5000UL    // change to 900000UL (15 min) for real use
#define READ_INTERVAL   2000UL
 
const float ETOH_SENS_CODE = -30.0f;
const float ETOH_TIA       = 249000.0f;
 
Adafruit_HDC302x hdc = Adafruit_HDC302x();
bool  hdc_connected  = false;
float etoh_Vgas0     = 0.0f;
 
enum State { WARMUP, READY, COLLECTING };
State state = WARMUP;
 
unsigned long warmupStart  = 0;
unsigned long lastReadTime = 0;
bool lastButtonState       = HIGH;
 
// ── Helpers ───────────────────────────────────────────
float toVolts(int raw) {
  return raw * (3.3f / 4095.0f);
}
 
float computeM() {
  return pow(10.0f, ETOH_SENS_CODE / 100.0f) * ETOH_TIA * 1e-9f;
}
 
String readMEMS(int pin) {
  float v = toVolts(analogRead(pin));
  if (v < DISCONNECTED_V) return "null";
  return String(v, 3) + " V";
}
 
String readAlcohol() {
  float Vgas = toVolts(analogRead(PIN_ETOH_GAS));
  if (Vgas < DISCONNECTED_V) return "null";
  float ppm = constrain((Vgas - etoh_Vgas0) / computeM(), 0.0f, 200.0f);
  return String(ppm, 2) + " ppm (Vgas=" + String(Vgas, 4) + "V)";
}
 
void captureBaseline() {
  float v = toVolts(analogRead(PIN_ETOH_GAS));
  if (v > DISCONNECTED_V) {
    etoh_Vgas0 = v;
    Serial.print("Alcohol baseline: ");
    Serial.print(etoh_Vgas0, 4);
    Serial.println(" V");
  } else {
    Serial.println("Alcohol sensor not detected — check wiring.");
  }
}
 
bool buttonJustPressed() {
  bool current = digitalRead(PIN_BUTTON);
  if (current == LOW && lastButtonState == HIGH) {
    lastButtonState = LOW;
    return true;
  }
  if (current == HIGH) lastButtonState = HIGH;
  return false;
}
 
// ── Setup ─────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
 
  Wire.begin(21, 22);
  if (hdc.begin(0x44, &Wire)) {
    hdc_connected = true;
    Serial.println("HDC3022 ready.");
  } else {
    Serial.println("HDC3022 not found.");
  }
 
  warmupStart = millis();
  Serial.println("Warming up...");
}
 
// ── Loop ──────────────────────────────────────────────
void loop() {
  bool pressed = buttonJustPressed();
 
  // ── WARMUP ───────────────────────────────────────────
  if (state == WARMUP) {
    if (millis() - warmupStart >= WARMUP_MS) {
      captureBaseline();
      state = READY;
      Serial.println("Warm-up done! Press button to start collecting.");
    }
  }
 
  // ── READY ────────────────────────────────────────────
  else if (state == READY) {
    if (pressed) {
      state = COLLECTING;
      lastReadTime = millis();
      Serial.println("Collecting... Press button to stop.");
    }
  }
 
  // ── COLLECTING ───────────────────────────────────────
  else if (state == COLLECTING) {
    if (pressed) {
      state = READY;
      Serial.println("Stopped. Press button to collect again.");
      return;
    }
 
    if (millis() - lastReadTime >= READ_INTERVAL) {
      lastReadTime = millis();
 
      String s_temp = "null";
      String s_hum  = "null";
      if (hdc_connected) {
        double temp = 0.0, RH = 0.0;
        if (hdc.readTemperatureHumidityOnDemand(temp, RH, TRIGGERMODE_LP0)) {
          s_temp = String(temp, 2) + " C";
          s_hum  = String(RH, 2)  + " %RH";
        }
      }
 
      Serial.println("--- Readings ---");
      Serial.print("  Temp:    "); Serial.println(s_temp);
      Serial.print("  RH:      "); Serial.println(s_hum);
      Serial.print("  HCHO:    "); Serial.println(readMEMS(PIN_HCHO));
      Serial.print("  VOC:     "); Serial.println(readMEMS(PIN_VOC));
      Serial.print("  NH3:     "); Serial.println(readMEMS(PIN_NH3));
      Serial.print("  Alcohol: "); Serial.println(readAlcohol());
      Serial.println("----------------");
      Serial.println();
    }
  }
}