// TODO: add temperature sensor

#include <Arduino.h>

#define PIN_BUTTON  32
#define PIN_HCHO    34
#define PIN_VOC     35
#define PIN_NH3     39
#define PIN_LED     2

bool measuringMode = false;
bool lastButtonState = HIGH;

float v0_hcho = -1;
float v0_voc  = -1;
float v0_nh3  = -1;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);         // 12-bit: 0-4095
  analogSetAttenuation(ADC_11db);   // read up to ~3.3V
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  Serial.println("=== WARMING UP MODE ===");
  Serial.println("Press button when ready to measure.");
}

void loop() {
  handleButton();

  delay(50);
}

void handleButton() {
  // Toggle button between "Warming Up Mode" and "Measuring Mode"
  bool currentButtonState = digitalRead(PIN_BUTTON);

  if (lastButtonState == HIGH && currentButtonState == LOW) {
    delay(50); // debounce

    if (!measuringMode) {
      // Calibrate all baselines before measuring mode
      Serial.println("Calibrating baselines in clean air...");
      v0_hcho = getAverageVoltage(PIN_HCHO);
      v0_voc  = getAverageVoltage(PIN_VOC);
      v0_nh3  = getAverageVoltage(PIN_NH3);

      Serial.print("V0 HCHO: "); Serial.print(v0_hcho, 3); Serial.println(" V");
      Serial.print("V0 VOC:  "); Serial.print(v0_voc,  3); Serial.println(" V");
      Serial.print("V0 NH3:  "); Serial.print(v0_nh3,  3); Serial.println(" V");

      measuringMode = true;
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
  // TODO: add in a better mechanism than average to reduce outlier
  long sum = 0;
  for (int i = 0; i < 64; i++) {
    sum += analogRead(pin);
    delay(2);
  }
  return (sum / 64.0) * (3.3 / 4095.0);
}

float estimatePPM_HCHO(float ratio) {
  // HCHO: uses Vs/V0 ratio mapped to sensitivity curve from datasheet
  // Based on Figure 1 data points — linear interpolation between known points
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

float estimatePPM_VOC(float voltage) {
  // VOC: uses absolute voltage mapped to linearity curve (ethanol reference)
  // From Figure 6: ~2.0V = 0ppm, ~4.0V = 100ppm
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