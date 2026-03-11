#include <EEPROM.h>

int count = 0;       
float sum_aR = 0.0;  

int recordCount = 0; 
int maxRecords = 125; // 2개씩 저장하므로 최대 개수가 125개로 줄어듭니다.

void setup() {
  Serial.begin(9600);
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
        float v_3v3, v_wrong;
        int addr = 4 + (i * 8); // 8바이트 간격으로 읽기
        
        EEPROM.get(addr, v_3v3);
        EEPROM.get(addr + 4, v_wrong);
        
        Serial.print(v_3v3);
        Serial.print(",");
        Serial.println(v_wrong);
      }
      Serial.println("END"); 
    }
  }

  // 2. A0핀 읽기 누적
  float aR = analogRead(A0);
  sum_aR = sum_aR + aR; 
  count++;

  // 3. 10초마다 계산 및 동시 저장
  if (count >= 10) {
    float avg_aR = sum_aR / 10.0;
    
    // 두 가지 방식으로 전압 계산
    float voltage_3v3 = avg_aR * (3.3 / 1023.0) * 2.0; // 올바른 계산 (3.3V 보드)
    float voltage_wrong = avg_aR * (5.0 / 1023.0) * 2.0; // 예전 방식 또는 RAW를 수식에 넣었을 때의 잘못된 계산

    // EEPROM에 두 값 모두 저장
    if (recordCount < maxRecords) {
      int addr = 4 + (recordCount * 8); // 한 세트당 8바이트(4+4) 차지
      EEPROM.put(addr, voltage_3v3);
      EEPROM.put(addr + 4, voltage_wrong);
      
      recordCount++;
      EEPROM.put(0, recordCount);
      
      Serial.print("Saved -> 3.3V기준: ");
      Serial.print(voltage_3v3);
      Serial.print("V / 잘못된기준: ");
      Serial.print(voltage_wrong);
      Serial.println("V");
    }
    
    sum_aR = 0.0;
    count = 0;
  }
  
  delay(1000); 
}