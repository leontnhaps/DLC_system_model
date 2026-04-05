#include <Arduino.h>

const uint8_t PIN_LED_R = 2;
const uint8_t PIN_LED_B = 3;
const uint8_t PIN_LED_G = 4;

// 1초 간격 테스트용, 필요시 10000ms로 바꾸면 10초 간격
const unsigned long STATE_INTERVAL_MS = 10000UL;

unsigned long g_started_ms = 0;
unsigned long g_last_step_ms = 0;
int g_stage = 0;

void setBatteryLedBits(int stage) {
  if (stage < 0) stage = 0;
  if (stage > 7) stage = 7;

  digitalWrite(PIN_LED_R, (stage & 0b100) ? HIGH : LOW);
  digitalWrite(PIN_LED_B, (stage & 0b010) ? HIGH : LOW);
  digitalWrite(PIN_LED_G, (stage & 0b001) ? HIGH : LOW);
}

const char* stageToBits(int stage) {
  switch (stage) {
    case 0: return "000";
    case 1: return "001";
    case 2: return "010";
    case 3: return "011";
    case 4: return "100";
    case 5: return "101";
    case 6: return "110";
    default: return "111";
  }
}

void printState(int stage) {
  unsigned long elapsed_s = (millis() - g_started_ms) / 1000UL;
  Serial.print("t=");
  Serial.print(elapsed_s);
  Serial.print("s stage=");
  Serial.print(stage);
  Serial.print(" bits=");
  Serial.println(stageToBits(stage));
}

void setup() {
  pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_LED_B, OUTPUT);
  pinMode(PIN_LED_G, OUTPUT);

  Serial.begin(9600);
  g_started_ms = millis();
  g_last_step_ms = g_started_ms;
  g_stage = 0;

  setBatteryLedBits(g_stage);
  printState(g_stage);
}

void loop() {
  unsigned long now = millis();
  if (now - g_last_step_ms < STATE_INTERVAL_MS) {
    return;
  }

  g_last_step_ms = now;
  g_stage = (g_stage + 1) % 8;
  setBatteryLedBits(g_stage);
  printState(g_stage);
}
