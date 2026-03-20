  #include <Arduino.h>
  #include <EEPROM.h>

  // ===================== 핀 설정 =====================
  const uint8_t PIN_BAT_A0 = A1;

  // LED (각각 100Ω 직렬 후 LED->GND, 핀 HIGH면 켜짐)
  const uint8_t PIN_LED_R = 2;
  const uint8_t PIN_LED_B = 3;
  const uint8_t PIN_LED_G = 4;

  // ===================== 분압 설정 =====================
  // 100k / 20k 저항 분배 => (100 + 20) / 20 = 6배
  const float DIV_GAIN = 6.0f;
  const float REF_VOLTAGE_mV = 1100.0f; // 아두이노 내부 1.1V (1100mV) 기준

  // ===================== 배터리 % 매핑 (대략) =====================
  const long V_FULL_mV  = 4200;  // 4.20V
  const long V_EMPTY_mV = 3000;  // 3.0V 

  // LED 구간(%)
  const int PCT_LOW = 35;
  const int PCT_MID = 70;
  const int PCT_HYS = 2;   // 깜빡임 방지 히스테리시스

  // ===================== 로깅 간격(초) =====================
  const uint16_t LOG_INTERVAL_S = 10;

  // ===================== EEPROM 저장 포맷(1바이트 압축) =====================
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
    uint16_t write_idx;       
    uint32_t total_written;   
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

  // ===================== 배터리 전압 읽기(mV) =====================
  static long readBattery_mV(bool printDebug = false) {
    long sum = 0;
    for (int i = 0; i < 8; i++) {
      sum += analogRead(PIN_BAT_A0);
      delay(2);
    }
    float adc = sum / 8.0f;

    // ★ 1.1V(1100mV) 내부 기준과 6배율 적용 계산식
    // Vbat(mV) = ADC * (1100mV / 1023) * 6
    float vBat_mV = adc * (REF_VOLTAGE_mV / 1023.0f) * DIV_GAIN;

    if (printDebug) {
      Serial.print("[DEBUG] 내부 1.1V 기준 작동 중 | 아날로그 리드(평균): ");
      Serial.print(adc);
      Serial.print(" -> 계산된 배터리 전압: ");
      Serial.print(vBat_mV);
      Serial.println(" mV");
    }

    return (long)(vBat_mV + 0.5f); // 반올림하여 정수(mV)로 반환
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
    EEPROM.put(0, H); 
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

    Serial.begin(9600); // 19200 보드레이트 유지
    delay(200);

    // ★ 1.1V 내부 기준 전압 사용 설정
    analogReference(INTERNAL);

    loadOrInitHeader();

    Serial.println("Battery EEPROM Logger Ready (1.1V Ref / 5.7x Divider)");
    Serial.println("Commands: D=dump CSV, C=clear");
  }

  void loop() {
    // ---- 시리얼 명령 ----
    if (Serial.available()) {
      char c = (char)Serial.read();
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

      if (g_state == ST_RED) {
        if (pct > (PCT_LOW + PCT_HYS)) g_state = ST_BLUE;
      } else if (g_state == ST_BLUE) {
        if (pct < (PCT_LOW - PCT_HYS)) g_state = ST_RED;
        else if (pct > (PCT_MID + PCT_HYS)) g_state = ST_GREEN;
      } else { 
        if (pct < (PCT_MID - PCT_HYS)) g_state = ST_BLUE;
      }

      setOneLed(g_state);
    }

    // ---- EEPROM 로깅(LOG_INTERVAL_S마다) ----
    if (now - lastLogMs >= (unsigned long)LOG_INTERVAL_S * 1000UL) {
      lastLogMs = now;
      long vbat = readBattery_mV(true); // true로 설정하여 디버그 메시지 출력
      int pct = voltageToPercent(vbat); 
      
      appendSample(vbat);
      
      Serial.print("[LOG] Vbat: ");
      Serial.print(vbat);
      Serial.print(" mV, ");
      Serial.print(pct);
      Serial.println(" %");
    }

    delay(10);
  }