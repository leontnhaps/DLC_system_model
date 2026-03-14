#include <EEPROM.h>

int count = 0;       
float sum_aR = 0.0;  

int recordCount = 0; 
int maxRecords = 125; // 2개씩(8바이트) 저장하므로 최대 125개

void setup() {
  Serial.begin(9600);
  
  // ★ 아두이노의 측정 잣대를 내부 1.1V로 고정
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
        float v_3v3, v_1v1; 
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

  // 2. A1핀 읽기 누적 (두 번째 코드의 배선에 맞춰 A1 사용)
  float aR = analogRead(A1);
  sum_aR = sum_aR + aR; 
  count++;

  // 3. 10초마다 계산 및 동시 저장
  if (count >= 10) {
    float avg_aR = sum_aR / 10.0;
    
    // 3.3V 기준 계산식 (잣대가 1.1V로 고정되었으므로 실제 전압과 오차가 있음)
    float voltage_3v3 = avg_aR * (3.3 / 1023.0) * 2.0; 
    
    // 1.1V 내부 기준 + 저항 분배(470k, 100k) 계산식 -> 실제 배터리 전압
    float voltage_1v1 = avg_aR * (1.1 / 1023.0) * 5.7; 

    // EEPROM에 두 값 모두 저장
    if (recordCount < maxRecords) {
      int addr = 4 + (recordCount * 8); 
      EEPROM.put(addr, voltage_3v3);     // 첫 번째 4바이트: 3.3V식
      EEPROM.put(addr + 4, voltage_1v1); // 두 번째 4바이트: 1.1V식
      
      recordCount++;
      EEPROM.put(0, recordCount);
      
      Serial.print("Saved -> 3.3V식: ");
      Serial.print(voltage_3v3);
      Serial.print("V / 1.1V식: ");
      Serial.print(voltage_1v1);
      Serial.println("V");
    }
    
    sum_aR = 0.0;
    count = 0;
  }
  
  delay(1000); 
}