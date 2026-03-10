#include <Arduino.h>

#define PIN_ETOH_GAS  25
#define PIN_ETOH_REF  14

#define ETOH_SENSITIVITY_CODE  21.5   // nA/ppm - from sensor label
#define ETOH_TIA_GAIN          249.0  // kV/A   


float getAverageVoltage(int pin) {
    long sum = 0;
    for (int i = 0; i < 64; i++) {
        sum += analogRead(pin);
        delay(2);
    }
    return (sum / 64.0) * (3.3 / 4095.0);
}

float calcM(float sensCode, float tiaGain) {
    // sensCode in nA/ppm, tiaGain in kΩ
    // nA × kΩ = 1e-9 × 1e3 = 1e-6 → result in V/ppm
    return sensCode * tiaGain * 1e-6;
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
}

void loop() {
    float vgas_etoh  = getAverageVoltage(PIN_ETOH_GAS);
    float vref_etoh  = getAverageVoltage(PIN_ETOH_REF);

    float m_etoh  = calcM(ETOH_SENSITIVITY_CODE,  ETOH_TIA_GAIN);

    float ppm_etoh  = (vgas_etoh  - vref_etoh)  / m_etoh;

    if (ppm_etoh < 0)  ppm_etoh = 0;
    if (ppm_etoh > 200) ppm_etoh = 200;

    Serial.print("[ETOH]  Vgas: "); Serial.print(vgas_etoh, 4);
    Serial.print(" V | Vref: ");   Serial.print(vref_etoh, 4);
    Serial.print(" V | Est: ");    Serial.print(ppm_etoh, 2);
    Serial.println(" ppm");

    delay(1000);
}