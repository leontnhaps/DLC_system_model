#include <EEPROM.h>

int count = 0;       
float sum_aR = 0.0;  

int recordCount = 0; 
int maxRecords = 125; // 2개씩 저장하므로 최대 개수가 125개로 줄어듭니다.

void setup() {
  Serial.begin(19200);
  
  // ★ 핵심 1: 아두이노의 측정 잣대를 내부 1.1V로 단단히 고정합니다.
  analogReference(INTERNAL); 
  
  EEPROM.get(0, recordCount);
  
  if (recordCount < 0 || recordCount > maxRecords) {
    recordCount = 0;
  }
}

void loop() {
  // 1. 파이썬 및 명령어 처리
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == 'c') { 
      recordCount = 0;
      EEPROM.put(0, recordCount);
      Serial.println("CLEARED"); 
    }
    else if (cmd == 'D') { 
      // 파이썬 덤프 요청 시 두 가지 값을 쉼표로 구분하여 전송
      for (int i = 0; i < recordCount; i++) {
        float v_3v3, v_1v1; // 변수명 변경 (v_wrong -> v_1v1)
        int addr = 4 + (i * 8); 
        
        EEPROM.get(addr, v_3v3);
        EEPROM.get(addr + 4, v_1v1); 
        
        Serial.print(v_3v3);
        Serial.print(",");
        Serial.println(v_1v1);
      }
      Serial.println("END"); 
    }
  }

  // 2. A0핀 읽기 누적
  float aR = analogRead(A1);
  sum_aR = sum_aR + aR; 
  count++;

  // 3. 10초마다 계산 및 동시 저장
  if (count >= 10) {
    float avg_aR = sum_aR / 10.0;
    
    // [주의] 잣대가 1.1V로 바뀌었으므로, 이 식은 이제 의미 없는 엉터리 값이 나옵니다.
    float voltage_3v3 = avg_aR * (3.3 / 1023.0) * 2.0; 
    
    // ★ 핵심 2: 새로운 계산식 (1.1V 잣대 * 5.7배 뻥튀기)
    // 470k와 100k 분배 비율: (470+100)/100 = 5.7
    float voltage_1v1 = avg_aR * (1.1 / 1023.0) * 5.7; 

    // EEPROM에 두 값 모두 저장
    if (recordCount < maxRecords) {
      int addr = 4 + (recordCount * 8); 
      EEPROM.put(addr, voltage_3v3);
      EEPROM.put(addr + 4, voltage_1v1); // 5V 자리에 진짜 전압 기록
      
      recordCount++;
      EEPROM.put(0, recordCount);
      
      Serial.print("Saved -> 3.3V식(무시): ");
      Serial.print(voltage_3v3);
      Serial.print("V / 진짜 배터리(1.1V+저항): ");
      Serial.print(voltage_1v1);
      Serial.println("V");
    }
    
    sum_aR = 0.0;
    count = 0;
  }
  
  delay(1000); 
}