#include <Arduino.h>
#include <EEPROM.h>

// ===================== 핀 설정 =====================
const uint8_t PIN_BAT_A0 = A0;

// LED (각각 100Ω 직렬 후 LED->GND, 핀 HIGH면 켜짐)
const uint8_t PIN_LED_R = 2;
const uint8_t PIN_LED_B = 3;
const uint8_t PIN_LED_G = 4;

// ===================== 분압 설정 =====================
// 1k/1k => A0 = Vbat/2 => Vbat = VA0 * 2
const float DIV_GAIN = 2.0f;

// ===================== 배터리 % 매핑 (대략) =====================
const long V_FULL_mV  = 4200;  // 4.20V
const long V_EMPTY_mV = 3300;  // 3.30V (보수적)

// LED 구간(%)
const int PCT_LOW = 35;
const int PCT_MID = 70;
const int PCT_HYS = 2;   // 깜빡임 방지 히스테리시스

// ===================== 로깅 간격(초) =====================
// EEPROM 1KB 기준, 데이터 1바이트/샘플이라 대략 1000샘플 저장 가능
// 30초 => 약 8.3시간, 60초 => 약 16.7시간
const uint16_t LOG_INTERVAL_S = 10;

// ===================== EEPROM 저장 포맷(1바이트 압축) =====================
// vbat_mV를 5mV step으로 1바이트에 압축 (3.000~4.275V 커버)
const uint16_t VMIN_LOG_mV = 3000;
const uint8_t  STEP_mV     = 5;

// ===================== 상태(숫자 상수로) =====================
const uint8_t ST_RED   = 0;
const uint8_t ST_BLUE  = 1;
const uint8_t ST_GREEN = 2;

uint8_t g_state = ST_RED;

// ===================== EEPROM 헤더 =====================
const uint16_t MAGIC = 0xBEEF;
const uint8_t  VER   = 1;

struct Header {
  uint16_t magic;
  uint8_t  ver;
  uint8_t  step_mV;
  uint16_t vmin_mV;
  uint16_t interval_s;
  uint16_t write_idx;       // 다음에 쓸 위치(0..cap-1)
  uint32_t total_written;   // 총 기록 횟수
};

Header H;

static uint16_t dataOffset() {
  return (uint16_t)sizeof(Header);
}

static uint16_t dataCapacity() {
  uint16_t len = (uint16_t)EEPROM.length();
  uint16_t off = dataOffset();
  if (len <= off) return 0;
  return (uint16_t)(len - off);
}

// ===================== LED 제어 =====================
static void setOneLed(uint8_t st) {
  digitalWrite(PIN_LED_R, (st == ST_RED)   ? HIGH : LOW);
  digitalWrite(PIN_LED_B, (st == ST_BLUE)  ? HIGH : LOW);
  digitalWrite(PIN_LED_G, (st == ST_GREEN) ? HIGH : LOW);
}

// ===================== VCC 측정(밴드갭) =====================
// Pro Mini RAW 사용 시 VCC가 정확히 3.30이 아닐 수 있어서 보정용
static long readVcc_mV() {
#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
  // AVcc 기준, 내부 1.1V 밴드갭 측정
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  delay(2);
  ADCSRA |= _BV(ADSC);
  while (bit_is_set(ADCSRA, ADSC)) {}
  uint16_t adc = ADC;
  // 1.1V * 1023 * 1000 = 1125300
  long vcc = 1125300L / (long)adc;
  return vcc;
#else
  return 3300; // fallback
#endif
}

// ===================== 배터리 전압 읽기(mV) =====================
static long readBattery_mV() {
  long vcc = readVcc_mV();

  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(PIN_BAT_A0);
    delay(2);
  }
  float adc = sum / 8.0f;

  float vA0_mV  = (adc * (float)vcc) / 1023.0f;
  float vBat_mV = vA0_mV * DIV_GAIN;

  return (long)(vBat_mV + 0.5f);
}

static int voltageToPercent(long vbat_mV) {
  float p = ((float)(vbat_mV - V_EMPTY_mV) * 100.0f) / (float)(V_FULL_mV - V_EMPTY_mV);
  if (p < 0) p = 0;
  if (p > 100) p = 100;
  return (int)(p + 0.5f);
}

// ===================== EEPROM 로드/초기화 =====================
static void loadOrInitHeader() {
  EEPROM.get(0, H);

  bool ok = (H.magic == MAGIC &&
             H.ver == VER &&
             H.step_mV == STEP_mV &&
             H.vmin_mV == VMIN_LOG_mV &&
             H.interval_s == LOG_INTERVAL_S);

  if (!ok) {
    H.magic = MAGIC;
    H.ver = VER;
    H.step_mV = STEP_mV;
    H.vmin_mV = VMIN_LOG_mV;
    H.interval_s = LOG_INTERVAL_S;
    H.write_idx = 0;
    H.total_written = 0;
    EEPROM.put(0, H);
  }
}

static void saveHeader() {
  EEPROM.put(0, H); // put은 바뀐 바이트만 update로 기록(Arduino EEPROM 구현)
}

// ===================== 인코딩/디코딩 =====================
static uint8_t encodeVbat(long vbat_mV) {
  long x = vbat_mV - (long)VMIN_LOG_mV;
  if (x < 0) x = 0;
  long code = x / (long)STEP_mV;
  if (code > 255) code = 255;
  return (uint8_t)code;
}

static long decodeVbat(uint8_t code) {
  return (long)VMIN_LOG_mV + (long)code * (long)STEP_mV;
}

// ===================== 샘플 추가(링버퍼) =====================
static void appendSample(long vbat_mV) {
  uint16_t cap = dataCapacity();
  if (cap == 0) return;

  uint8_t code = encodeVbat(vbat_mV);
  uint16_t addr = dataOffset() + H.write_idx;

  EEPROM.update(addr, code);

  H.write_idx = (uint16_t)((H.write_idx + 1) % cap);
  H.total_written++;
  saveHeader();
}

// ===================== 덤프/클리어 =====================
static void dumpCsv() {
  uint16_t cap = dataCapacity();
  if (cap == 0) {
    Serial.println("ERR,EEPROM capacity=0");
    return;
  }

  uint32_t total = H.total_written;
  uint32_t n = (total < cap) ? total : cap;

  // 링버퍼에서 가장 오래된 데이터 시작점
  uint16_t start = (total < cap) ? 0 : H.write_idx;

  Serial.println("idx,time_s,vbat_mV,percent");

  for (uint32_t i = 0; i < n; i++) {
    uint16_t idx = (uint16_t)((start + i) % cap);
    uint8_t code = EEPROM.read(dataOffset() + idx);
    long v = decodeVbat(code);
    int pct = voltageToPercent(v);
    uint32_t t = i * (uint32_t)H.interval_s;

    Serial.print(i); Serial.print(",");
    Serial.print(t); Serial.print(",");
    Serial.print(v); Serial.print(",");
    Serial.println(pct);
  }
  Serial.println("END");
}

static void clearLog() {
  H.write_idx = 0;
  H.total_written = 0;
  saveHeader();
  Serial.println("OK,CLEARED");
}

// ===================== 메인 =====================
unsigned long lastLogMs = 0;
unsigned long lastLedMs = 0;

void setup() {
  pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_LED_B, OUTPUT);
  pinMode(PIN_LED_G, OUTPUT);

  setOneLed(ST_RED);

  Serial.begin(19200);
  delay(200);

  loadOrInitHeader();

  Serial.println("Battery EEPROM Logger Ready");
  Serial.println("Commands: D=dump CSV, C=clear");
}

void loop() {
  // ---- 시리얼 명령 ----
  if (Serial.available()) {
    char c = (char)Serial.read();
    // 입력이 들어오면 BLUE LED 50ms 깜빡 (수신 확인용)
    digitalWrite(PIN_LED_B, HIGH);
    delay(50);
    digitalWrite(PIN_LED_B, LOW);
    if (c == 'D' || c == 'd') dumpCsv();
    if (c == 'C' || c == 'c') clearLog();
  }

  unsigned long now = millis();

  // ---- LED 상태 업데이트(0.5초마다) ----
  if (now - lastLedMs >= 500) {
    lastLedMs = now;

    long vbat = readBattery_mV();
    int pct = voltageToPercent(vbat);

    // 히스테리시스
    if (g_state == ST_RED) {
      if (pct > (PCT_LOW + PCT_HYS)) g_state = ST_BLUE;
    } else if (g_state == ST_BLUE) {
      if (pct < (PCT_LOW - PCT_HYS)) g_state = ST_RED;
      else if (pct > (PCT_MID + PCT_HYS)) g_state = ST_GREEN;
    } else { // GREEN
      if (pct < (PCT_MID - PCT_HYS)) g_state = ST_BLUE;
    }

    setOneLed(g_state);
  }

  // ---- EEPROM 로깅(LOG_INTERVAL_S마다) ----
  if (now - lastLogMs >= (unsigned long)LOG_INTERVAL_S * 1000UL) {
    lastLogMs = now;
    long vbat = readBattery_mV();
    int pct = voltageToPercent(vbat); // 퍼센트 계산 추가
    
    appendSample(vbat);
    
    // 시리얼 모니터에 현재 상태 출력 (로깅 주기에 맞춰서)
    Serial.print("[LOG] Vbat: ");
    Serial.print(vbat);
    Serial.print(" mV, ");
    Serial.print(pct);
    Serial.println(" %");
  }

  delay(10);
}