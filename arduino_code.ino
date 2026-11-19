#if defined(ESP32)
  const int RELAY_PIN = 26;
#elif defined(ESP8266)
  const int RELAY_PIN = 5;   // D1
#else
  const int RELAY_PIN = 7;   // Arduino Uno Pin 7
#endif

const bool ACTIVE_HIGH = true; // Change to false if your relay is active-LOW

void setBulb(bool on) {
  int level = ACTIVE_HIGH ? (on ? HIGH : LOW) : (on ? LOW : HIGH);
  digitalWrite(RELAY_PIN, level);
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  setBulb(false);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd == "ON") {
      setBulb(true);
      Serial.println("ACK:ON");
    } else if (cmd == "OFF") {
      setBulb(false);
      Serial.println("ACK:OFF");
    }
  }
}