// TODO: might have to switch everything to resistance based reading

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_HDC302x.h>

#define PIN_BUTTON  32
#define PIN_HCHO    39
#define PIN_VOC     35
#define PIN_NH3     34
#define PIN_LED     2

// Expected clean-air baseline voltage ranges derived from datasheet long-term stability curves:
//   SEN0566 VOC  (Fig 7): V0 ~1.9 V in clean air
//   SEN0567 NH3  (Fig 7): V0 ~1.2 V in clean air
//   SEN0563 HCHO (Fig 4): V0 ~1.7 V in clean air
#define GLOBAL_BASELINE_MIN   0.01f
#define VOC_BASELINE_MIN      1.0f
#define VOC_BASELINE_MAX      2.8f
#define NH3_BASELINE_MIN      0.5f
#define NH3_BASELINE_MAX      2.2f
#define HCHO_BASELINE_MIN     0.8f
#define HCHO_BASELINE_MAX     2.5f

// Sensor fault thresholds (3.3 V rail, ADC_11db attenuation):
//   < FAULT_LOW  -> open circuit or heater failure
//   > FAULT_HIGH -> short circuit or ADC saturation
#define VOLTAGE_FAULT_LOW  0.2f
#define VOLTAGE_FAULT_HIGH 3.2f

Adafruit_HDC302x hdc = Adafruit_HDC302x();
bool hdcReady             = false;
bool hchoEnabled          = true;
bool checkVoltageEnabled  = false;

// Set before flashing: true = raw voltage (V),  false = estimated PPM
const bool outputVoltage = true;

bool measuringMode    = false;
bool lastButtonState  = HIGH;

float v0_hcho = -1;
float v0_voc  = -1;
float v0_nh3  = -1;

const unsigned long MEASURE_INTERVAL_MS = 1000;
unsigned long lastMeasureTime = 0;

void     handleButton();
float    getAverageVoltage(int pin);
bool     checkVoltage(float voltage, const char* name);
bool     checkBaseline(float voltage, float minV, float maxV, const char* name);
void     takeMeasurement();
float    estimatePPM_HCHO(float ratio);
float    estimatePPM_VOC(float voltage);
float    estimatePPM_NH3(float voltage);

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);         // 12-bit: 0–4095
  analogSetAttenuation(ADC_11db);   // read up to ~3.3 V
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  Serial.println("=== WARMING UP MODE ===");
  Serial.println("Press button when ready to measure.");
  if (!hchoEnabled) {
    Serial.println("[INFO] HCHO sensor disabled (not installed).");
  }

  Wire.begin(21, 22);
  hdcReady = hdc.begin(0x44, &Wire);
  Serial.println(hdcReady ? "HDC3022 ready." : "HDC3022 not found.");
}

void loop() {
  handleButton();

  if (measuringMode) {
    unsigned long now = millis();
    if (now - lastMeasureTime >= MEASURE_INTERVAL_MS) {
      lastMeasureTime = now;
      takeMeasurement();
    }
  }

  delay(50);
}

void handleButton() {
  // Switches between "Measuring Mode" and "Warming Up Mode"
  bool currentButtonState = digitalRead(PIN_BUTTON);

  if (lastButtonState == HIGH && currentButtonState == LOW) {
    delay(50); // debounce

    if (!measuringMode) {
      Serial.println("Calibrating baselines in clean air...");

      v0_voc = getAverageVoltage(PIN_VOC);
      Serial.print("V0 VOC:  "); Serial.print(v0_voc,  3); Serial.println(" V");
      checkVoltage(v0_voc, "VOC");
      checkBaseline(v0_voc, GLOBAL_BASELINE_MIN, VOC_BASELINE_MAX, "VOC");

      v0_nh3 = getAverageVoltage(PIN_NH3);
      Serial.print("V0 NH3:  "); Serial.print(v0_nh3,  3); Serial.println(" V");
      checkVoltage(v0_nh3, "NH3");
      checkBaseline(v0_nh3, GLOBAL_BASELINE_MIN, NH3_BASELINE_MAX, "NH3");

      if (hchoEnabled) {
        v0_hcho = getAverageVoltage(PIN_HCHO);
        Serial.print("V0 HCHO: "); Serial.print(v0_hcho, 3); Serial.println(" V");
        checkVoltage(v0_hcho, "HCHO");
        checkBaseline(v0_hcho, GLOBAL_BASELINE_MIN, HCHO_BASELINE_MAX, "HCHO");
      }


      measuringMode = true;
      lastMeasureTime = 0;   // trigger an immediate first reading
      digitalWrite(PIN_LED, HIGH);
      Serial.println("=== MEASURING MODE ===");
    } else {
      measuringMode = false;
      digitalWrite(PIN_LED, LOW);
      Serial.println("=== WARMING UP MODE ===");
      Serial.println("Press button when ready to measure.");
    }
  }

  lastButtonState = currentButtonState;
}


float getAverageVoltage(int pin) {
  // Average 64 samples
  long sum = 0;
  for (int i = 0; i < 64; i++) {
    sum += analogRead(pin);
    delay(2);
  }
  return (sum / 64.0) * (3.3 / 4095.0);
}

bool checkVoltage(float voltage, const char* name) {
  // Returns false and prints a fault message when voltage is outside the plausible
  // operating range of the sensor (same thresholds for all sensors on the 3.3 V rail).
  if (!checkVoltageEnabled) return true;
  if (voltage < VOLTAGE_FAULT_LOW) {
    Serial.print("[FAULT] "); Serial.print(name);
    Serial.println(": near 0 V — open circuit or heater failure");
    return false;
  }
  if (voltage > VOLTAGE_FAULT_HIGH) {
    Serial.print("[FAULT] "); Serial.print(name);
    Serial.println(": near rail — short circuit or ADC saturation");
    return false;
  }
  return true;
}


bool checkBaseline(float voltage, float minV, float maxV, const char* name) {
  // Warns when a calibration baseline falls outside the expected clean-air range
  // from the datasheet stability curves.  Does not block measuring mode.
  if (voltage < minV || voltage > maxV) {
    Serial.print("[WARN]  "); Serial.print(name);
    Serial.print(": baseline "); Serial.print(voltage, 3);
    Serial.print(" V outside expected [");
    Serial.print(minV, 1); Serial.print(", "); Serial.print(maxV, 1);
    Serial.println("] V — sensor may need more warm-up or could be faulty");
    return false;
  }
  return true;
}

void takeMeasurement() {
  char tempStr[8], rhStr[8], vocStr[8], nh3Str[8], hchoStr[8];

  // HDC302x — always physical units
  if (hdcReady) {
    double temp, rh;
    if (hdc.readTemperatureHumidityOnDemand(temp, rh, TRIGGERMODE_LP0)) {
      snprintf(tempStr, sizeof(tempStr), "%.1f", temp);
      snprintf(rhStr,   sizeof(rhStr),   "%.1f", rh);
    } else {
      Serial.println("[FAULT] HDC3022: read failed (CRC or I2C fault)");
      strcpy(tempStr, "NAN"); strcpy(rhStr, "NAN");
    }
  } else {
    strcpy(tempStr, "NAN"); strcpy(rhStr, "NAN");
  }

  // VOC
  float vs_voc = getAverageVoltage(PIN_VOC);
  if (checkVoltage(vs_voc, "VOC")) {
    if (outputVoltage) snprintf(vocStr, sizeof(vocStr), "%.3f", vs_voc);
    else               snprintf(vocStr, sizeof(vocStr), "%.1f", estimatePPM_VOC(vs_voc));
  } else {
    strcpy(vocStr, "NAN");
  }

  // NH3
  float vs_nh3 = getAverageVoltage(PIN_NH3);
  if (checkVoltage(vs_nh3, "NH3")) {
    if (outputVoltage) snprintf(nh3Str, sizeof(nh3Str), "%.3f", vs_nh3);
    else               snprintf(nh3Str, sizeof(nh3Str), "%.1f", estimatePPM_NH3(vs_nh3));
  } else {
    strcpy(nh3Str, "NAN");
  }

  // HCHO
  if (hchoEnabled) {
    float vs_hcho = getAverageVoltage(PIN_HCHO);
    if (checkVoltage(vs_hcho, "HCHO")) {
      if (outputVoltage) snprintf(hchoStr, sizeof(hchoStr), "%.3f", vs_hcho);
      else               snprintf(hchoStr, sizeof(hchoStr), "%.2f", estimatePPM_HCHO(vs_hcho / v0_hcho));
    } else {
      strcpy(hchoStr, "NAN");
    }
  } else {
    strcpy(hchoStr, "NAN");
  }

  // Build and emit frame.
  // Format: $DATA,<millis_ms>,<temp_C>,<rh_pct>,<mode>,<voc>,<nh3>,<hcho>*<CRC>
  // <mode> is 'V' (voltage) or 'P' (PPM)
  char buf[96];
  snprintf(buf, sizeof(buf), "DATA,%lu,%s,%s,%c,%s,%s,%s",
           millis(), tempStr, rhStr, outputVoltage ? 'V' : 'P',
           vocStr, nh3Str, hchoStr);

  uint8_t crc = 0;
  for (int i = 0; buf[i]; i++) crc ^= (uint8_t)buf[i];

  Serial.print('$');
  Serial.print(buf);
  Serial.printf("*%02X\r\n", crc);
}

float estimatePPM_VOC(float voltage) {
  // VOC: uses absolute voltage mapped to linearity curve (ethanol reference)
  float curve[][2] = {
    {2.0, 0.0},
    {2.5, 20.0},
    {3.0, 40.0},
    {3.5, 70.0},
    {4.0, 100.0}
  };
  int points = 5;

  if (voltage <= curve[0][0]) return 0.0;
  if (voltage >= curve[points-1][0]) return curve[points-1][1];

  for (int i = 0; i < points - 1; i++) {
    if (voltage >= curve[i][0] && voltage <= curve[i+1][0]) {
      float t = (voltage - curve[i][0]) / (curve[i+1][0] - curve[i][0]);
      return curve[i][1] + t * (curve[i+1][1] - curve[i][1]);
    }
  }
  return -1;
}

float estimatePPM_NH3(float voltage) {
  // NH3: uses absolute voltage mapped to linearity curve
  float curve[][2] = {
    {1.2, 0.0},
    {2.0, 20.0},
    {2.5, 35.0},
    {3.0, 50.0},
    {3.5, 75.0},
    {4.0, 100.0}
  };
  int points = 6;

  if (voltage <= curve[0][0]) return 0.0;
  if (voltage >= curve[points-1][0]) return curve[points-1][1];

  for (int i = 0; i < points - 1; i++) {
    if (voltage >= curve[i][0] && voltage <= curve[i+1][0]) {
      float t = (voltage - curve[i][0]) / (curve[i+1][0] - curve[i][0]);
      return curve[i][1] + t * (curve[i+1][1] - curve[i][1]);
    }
  }
  return -1;
}


float estimatePPM_HCHO(float ratio) {
  // HCHO: uses Vs/V0 ratio mapped to sensitivity curve from datasheet
  // {ratio, ppm} pairs read from the sensitivity curve
  float curve[][2] = {
    {1.0,  0.0},
    {1.2,  0.1},
    {1.45, 0.2},
    {1.75, 0.4},
    {1.85, 0.6},
    {2.1,  0.8},
    {2.65, 1.2}
  };
  int points = 7;

  // Below minimum or above maximum
  if (ratio <= curve[0][0]) return 0.0;
  if (ratio >= curve[points-1][0]) return curve[points-1][1];

  // Find the segment and interpolate
  for (int i = 0; i < points - 1; i++) {
    if (ratio >= curve[i][0] && ratio <= curve[i+1][0]) {
      float t = (ratio - curve[i][0]) / (curve[i+1][0] - curve[i][0]);
      return curve[i][1] + t * (curve[i+1][1] - curve[i][1]);
    }
  }

  return -1; // should never reach here
}