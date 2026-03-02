const int buttonPin = 32;
const int ledPin = 2;
bool measuringMode = false;
bool lastButtonState = HIGH;

void setup() {
  Serial.begin(115200);
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  Serial.println("=== WARMING UP MODE ===");
  Serial.println("Press button when ready to measure.");
}

void loop() {
  handleButton();
  delay(50);
}

void handleButton() {
  bool currentButtonState = digitalRead(buttonPin);

  if (lastButtonState == HIGH && currentButtonState == LOW) {
    delay(50); // debounce

    if (!measuringMode) {
      measuringMode = true;
      Serial.println("=== MEASURING MODE ===");
      digitalWrite(ledPin, HIGH); // LED on = measuring
    } else {
      measuringMode = false;
      Serial.println("=== WARMING UP MODE ===");
      digitalWrite(ledPin, LOW);  // LED off = warming up
    }
  }

  lastButtonState = currentButtonState;
}