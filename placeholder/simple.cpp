#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_HDC302x.h>

#define PIN_BUTTON  32
#define PIN_HCHO    39
#define PIN_VOC     35
#define PIN_NH3     34
#define PIN_LED     2

Adafruit_HDC302x hdc = Adafruit_HDC302x();
bool hdcReady = false;

int vocRawValue = 0;
int hchoRawValue = 0;
int nh3RawValue = 0;



void setup() {
    Serial.begin(115200);
    analogReadResolution(12);         // 12-bit: 0–4095
    analogSetAttenuation(ADC_11db);   // read up to ~3.3 V
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    pinMode(PIN_LED, OUTPUT);
    digitalWrite(PIN_LED, LOW);
    Wire.begin(21, 22);
    hdcReady = hdc.begin(0x44, &Wire);
    Serial.println(hdcReady ? "HDC3022 ready." : "HDC3022 not found.");
}

void loop() {
    sensorValue = analogRead(PIN_VOC);
    Serial.println(sensorValue);
    delay(100);
}

void readAnalog() {

}