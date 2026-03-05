#include <Arduino.h>

#define PIN_H2S_GAS   25
#define PIN_H2S_REF   14

#define H2S_SENSITIVITY_CODE   216.09 // nA/ppm - from sensor label
#define H2S_TIA_GAIN           49.9   // kV/A

float getAverageVoltage(int pin) {
    long sum = 0;
    for (int i = 0; i < 64; i++) {
        sum += analogRead(pin);
        delay(2);
    }
    return (sum / 64.0) * (3.3 / 4095.0);
}

float calcM(float sensCode, float tiaGain) {
    return sensCode * tiaGain * 1e-6;
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
}

void loop() {
    float vgas_h2s  = getAverageVoltage(PIN_H2S_GAS);
    float vref_h2s  = getAverageVoltage(PIN_H2S_REF);

    float m_h2s  = calcM(H2S_SENSITIVITY_CODE,  H2S_TIA_GAIN);

    float ppm_h2s  = (vgas_h2s  - vref_h2s)  / m_h2s;

    if (ppm_h2s < 0)  ppm_h2s = 0;
    if (ppm_h2s > 50) ppm_h2s = 50;

    Serial.print("[H2S]  Vgas: "); Serial.print(vgas_h2s, 4);
    Serial.print(" V | Vref: ");   Serial.print(vref_h2s, 4);
    Serial.print(" V | Est: ");    Serial.print(ppm_h2s, 2);
    Serial.println(" ppm");

    delay(1000);
}