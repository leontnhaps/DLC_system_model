#include <Arduino.h>
#include <EEPROM.h>

// =====================================================
// 핀 설정
// =====================================================
const uint8_t PIN_BAT   = A1;
const uint8_t PIN_LED_R = 2;
const uint8_t PIN_LED_B = 3;
const uint8_t PIN_LED_G = 4;

// =====================================================
// 분압 설정
// BAT+ --- 39k --- A1 --- 12k --- GND
// =====================================================
const double R_TOP = 39000.0;
const double R_BOTTOM = 12000.0;
const double DIV_GAIN = (R_TOP + R_BOTTOM) / R_BOTTOM;   // 4.25
const double REF_VOLTAGE_mV = 1100.0;                    // INTERNAL 1.1V

// =====================================================
// 배터리 범위
// =====================================================
const long V_FULL_mV  = 4200;   // 4.2V
const long V_EMPTY_mV = 3000;   // 3.0V

// LED 구간
const long V_LOW_mV = 3400;     // 3.0 ~ 3.4V -> RED
const long V_MID_mV = 3800;     // 3.4 ~ 3.8V -> BLUE / 3.8 ~ 4.2V -> GREEN
const long LED_HYS_mV = 30;     // 히스테리시스

// =====================================================
// 측정 / 저장 주기
// =====================================================
const unsigned long SAMPLE_INTERVAL_MS = 1000UL;   // 1초마다 측정
const uint8_t AVG_COUNT = 10;                      // 10개 평균 -> 10초 평균 저장

// =====================================================
// EEPROM 저장 포맷
// =====================================================
const uint16_t VMIN_LOG_mV = 3000;
const uint8_t  STEP_mV     = 5;

// 헤더를 너무 자주 저장하면 특정 주소가 빨리 닳으므로
// 60개 로그마다 한 번만 저장 (10초 * 60 = 10분)
const uint8_t HEADER_SAVE_EVERY = 60;

// =====================================================
// 상태 정의
// =====================================================
enum LedState {
  ST_RED = 0,
  ST_BLUE,
  ST_GREEN
};

LedState g_state = ST_RED;

// =====================================================
// EEPROM 헤더
// =====================================================
const uint16_t MAGIC = 0xBEEF;
const uint8_t  VER   = 1;

struct Header {
  uint16_t magic;
  uint8_t  ver;
  uint8_t  step_mV;
  uint16_t vmin_mV;
  uint16_t interval_s;
  uint16_t write_idx;
  uint32_t total_written;
};

Header H;
uint8_t pendingHeaderLogs = 0;

// =====================================================
// 시간 / 누적 변수
// =====================================================
unsigned long lastSampleMs = 0;

long sampleSum_mV = 0;      // 최근 10초 누적합
uint8_t sampleCount = 0;    // 최근 10초 샘플 개수

// =====================================================
// 유틸
// =====================================================
uint16_t dataOffset() {
  return (uint16_t)sizeof(Header);
}

uint16_t dataCapacity() {
  uint16_t len = (uint16_t)EEPROM.length();
  uint16_t off = dataOffset();

  if (len <= off) return 0;
  return (uint16_t)(len - off);
}

void saveHeader() {
  EEPROM.put(0, H);
}

void syncHeaderIfNeeded() {
  if (pendingHeaderLogs > 0) {
    saveHeader();
    pendingHeaderLogs = 0;
  }
}

void loadOrInitHeader() {
  EEPROM.get(0, H);

  bool ok = (H.magic == MAGIC &&
             H.ver == VER &&
             H.step_mV == STEP_mV &&
             H.vmin_mV == VMIN_LOG_mV &&
             H.interval_s == (AVG_COUNT * (SAMPLE_INTERVAL_MS / 1000UL)));

  if (!ok) {
    H.magic = MAGIC;
    H.ver = VER;
    H.step_mV = STEP_mV;
    H.vmin_mV = VMIN_LOG_mV;
    H.interval_s = (uint16_t)(AVG_COUNT * (SAMPLE_INTERVAL_MS / 1000UL)); // 10초
    H.write_idx = 0;
    H.total_written = 0;
    saveHeader();
  }
}

// =====================================================
// LED 제어
// =====================================================
void setOneLed(LedState st) {
  digitalWrite(PIN_LED_R, (st == ST_RED)   ? HIGH : LOW);
  digitalWrite(PIN_LED_B, (st == ST_BLUE)  ? HIGH : LOW);
  digitalWrite(PIN_LED_G, (st == ST_GREEN) ? HIGH : LOW);
}

// =====================================================
// 배터리 전압 읽기
// - 한 번 측정할 때 내부적으로 8번 평균
// =====================================================
long readBattery_mV() {
  long sum = 0;

  for (uint8_t i = 0; i < 8; i++) {
    sum += analogRead(PIN_BAT);
    delay(2);
  }

  double adc = sum / 8.0;
  double vBat_mV = adc * (REF_VOLTAGE_mV / 1023.0) * DIV_GAIN;

  return (long)(vBat_mV + 0.5);
}

// =====================================================
// 전압 -> 퍼센트
// =====================================================
int voltageToPercent(long vbat_mV) {
  if (vbat_mV < V_EMPTY_mV) vbat_mV = V_EMPTY_mV;
  if (vbat_mV > V_FULL_mV)  vbat_mV = V_FULL_mV;

  long num = (vbat_mV - V_EMPTY_mV) * 100L;
  long den = (V_FULL_mV - V_EMPTY_mV);

  return (int)((num + den / 2) / den);
}

// =====================================================
// LED 상태 갱신
// runningAvg_mV 기준으로 LED 표시
// =====================================================
void updateBatteryLed(long vbat_mV) {
  if (g_state == ST_RED) {
    if (vbat_mV >= (V_LOW_mV + LED_HYS_mV)) {
      g_state = ST_BLUE;
    }
  }
  else if (g_state == ST_BLUE) {
    if (vbat_mV < (V_LOW_mV - LED_HYS_mV)) {
      g_state = ST_RED;
    }
    else if (vbat_mV >= (V_MID_mV + LED_HYS_mV)) {
      g_state = ST_GREEN;
    }
  }
  else { // ST_GREEN
    if (vbat_mV < (V_MID_mV - LED_HYS_mV)) {
      g_state = ST_BLUE;
    }
  }

  setOneLed(g_state);
}

// =====================================================
// 인코딩 / 디코딩
// =====================================================
uint8_t encodeVbat(long vbat_mV) {
  long x = vbat_mV - (long)VMIN_LOG_mV;
  if (x < 0) x = 0;

  long code = (x + (STEP_mV / 2)) / (long)STEP_mV;
  if (code > 255) code = 255;

  return (uint8_t)code;
}

long decodeVbat(uint8_t code) {
  return (long)VMIN_LOG_mV + (long)code * (long)STEP_mV;
}

// =====================================================
// EEPROM 샘플 저장
// =====================================================
void appendSample(long vbat_mV) {
  uint16_t cap = dataCapacity();
  if (cap == 0) return;

  uint8_t code = encodeVbat(vbat_mV);
  uint16_t addr = dataOffset() + H.write_idx;

  EEPROM.update(addr, code);

  H.write_idx = (uint16_t)((H.write_idx + 1) % cap);

  if (H.total_written < 0xFFFFFFFFUL) {
    H.total_written++;
  }

  pendingHeaderLogs++;

  if (pendingHeaderLogs >= HEADER_SAVE_EVERY) {
    saveHeader();
    pendingHeaderLogs = 0;
  }
}

// =====================================================
// CSV 출력
// =====================================================
void dumpCsv() {
  syncHeaderIfNeeded();

  uint16_t cap = dataCapacity();
  if (cap == 0) {
    Serial.println("ERR,EEPROM capacity=0");
    return;
  }

  uint32_t total = H.total_written;
  uint32_t n = (total < cap) ? total : cap;
  uint16_t start = (total < cap) ? 0 : H.write_idx;

  Serial.println("idx,time_s,vbat_mV,percent");

  for (uint32_t i = 0; i < n; i++) {
    uint16_t idx = (uint16_t)((start + i) % cap);
    uint8_t code = EEPROM.read(dataOffset() + idx);

    long v = decodeVbat(code);
    int pct = voltageToPercent(v);
    uint32_t t = i * (uint32_t)H.interval_s;

    Serial.print(i);
    Serial.print(",");
    Serial.print(t);
    Serial.print(",");
    Serial.print(v);
    Serial.print(",");
    Serial.println(pct);
  }

  Serial.println("END");
}

// =====================================================
// 로그 초기화
// =====================================================
void clearLog() {
  H.write_idx = 0;
  H.total_written = 0;
  pendingHeaderLogs = 0;
  saveHeader();

  sampleSum_mV = 0;
  sampleCount = 0;

  Serial.println("OK,CLEARED");
}

// =====================================================
// setup
// =====================================================
void setup() {
  pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_LED_B, OUTPUT);
  pinMode(PIN_LED_G, OUTPUT);

  setOneLed(ST_RED);

  Serial.begin(9600);

  analogReference(INTERNAL);
  delay(5);

  // 기준전압 전환 직후 더미 리드
  analogRead(PIN_BAT);
  delay(5);

  loadOrInitHeader();

  Serial.println("Battery EEPROM Logger Ready");
  Serial.println("Commands: D=dump CSV, C=clear");
}

// =====================================================
// loop
// =====================================================
void loop() {
  // -------------------------
  // 시리얼 명령 처리
  // -------------------------
  if (Serial.available()) {
    char c = (char)Serial.read();

    if (c == 'D' || c == 'd') {
      dumpCsv();
    }
    else if (c == 'C' || c == 'c') {
      clearLog();
    }
  }

  unsigned long now = millis();

  // -------------------------
  // 1초마다 측정
  // -------------------------
  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;

    long vbat = readBattery_mV();

    sampleSum_mV += vbat;
    sampleCount++;

    // LED는 현재까지의 누적 평균값 기준으로 표시
    long runningAvg_mV = sampleSum_mV / sampleCount;
    updateBatteryLed(runningAvg_mV);

    Serial.print("SAMPLE,");
    Serial.println(vbat);

    // -------------------------
    // 10개 모이면 평균 저장
    // -------------------------
    if (sampleCount >= AVG_COUNT) {
      long avgVbat = sampleSum_mV / sampleCount;

      appendSample(avgVbat);

      Serial.print("LOG,");
      Serial.print(avgVbat);
      Serial.print(",");
      Serial.println(voltageToPercent(avgVbat));

      sampleSum_mV = 0;
      sampleCount = 0;
    }
  }

  delay(10);
}