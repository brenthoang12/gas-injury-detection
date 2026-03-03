// TODO: add v_offset
// TODO: add temperature compensation

#include <Arduino.h>

#define PIN_ETOH_GAS  26
#define PIN_ETOH_REF  27
#define PIN_H2S_GAS   25
#define PIN_H2S_REF   14
#define PIN_BUTTON    18
#define PIN_LED       2

#define ETOH_SENSITIVITY_CODE  21.5   // nA/ppm - from sensor label
#define ETOH_TIA_GAIN          249.0  // kV/A   

#define H2S_SENSITIVITY_CODE   4.94   // nA/ppm - from sensor label [change here]
#define H2S_TIA_GAIN           49.9   // kV/A   

bool measuringMode = false;
bool lastButtonState = HIGH;

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    pinMode(PIN_LED, OUTPUT);
    digitalWrite(PIN_LED, LOW);
    Serial.println("=== WARMING UP MODE ===");
    Serial.println("EtOH: warm up 15 min | H2S: warm up 60 min");
    Serial.println("Press button when ready to measure.");
}

void loop() {
    handleButton();

    if (measuringMode) {
        float vgas_etoh = getAverageVoltage(PIN_ETOH_GAS);
        float vref_etoh = getAverageVoltage(PIN_ETOH_REF);
        float vgas_h2s  = getAverageVoltage(PIN_H2S_GAS);
        float vref_h2s  = getAverageVoltage(PIN_H2S_REF);

        float m_etoh = calcM(ETOH_SENSITIVITY_CODE, ETOH_TIA_GAIN);
        float m_h2s  = calcM(H2S_SENSITIVITY_CODE,  H2S_TIA_GAIN);

        float ppm_etoh = (vgas_etoh - vref_etoh) / m_etoh;
        float ppm_h2s  = (vgas_h2s  - vref_h2s)  / m_h2s;

        if (ppm_etoh < 0) ppm_etoh = 0;
        if (ppm_h2s  < 0) ppm_h2s  = 0;

        Serial.println("--- Sensor Readings ---");
        Serial.print("[EtOH] Vgas: "); Serial.print(vgas_etoh, 4);
        Serial.print(" V | Vref: ");   Serial.print(vref_etoh, 4);
        Serial.print(" V | Est: ");    Serial.print(ppm_etoh, 2);
        Serial.println(" ppm");

        Serial.print("[H2S]  Vgas: "); Serial.print(vgas_h2s, 4);
        Serial.print(" V | Vref: ");   Serial.print(vref_h2s, 4);
        Serial.print(" V | Est: ");    Serial.print(ppm_h2s, 2);
        Serial.println(" ppm");
        Serial.println();

        delay(1000);
    }

    delay(50);
}

void handleButton() {
    bool currentButtonState = digitalRead(PIN_BUTTON);

    if (lastButtonState == HIGH && currentButtonState == LOW) {
        delay(50); // debounce

        if (!measuringMode) {
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
    long sum = 0;
    for (int i = 0; i < 64; i++) {
        sum += analogRead(pin);
        delay(2);
    }
    return (sum / 64.0) * (3.3 / 4095.0);
}

float calcM(float sensCode, float tiaGain) {
    // M (V/ppm) = SensCode(nA/ppm) * TIAGain(kV/A) * 1e-6    
    return sensCode * tiaGain * 1e-6;
}