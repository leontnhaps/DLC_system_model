#include <Wire.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

void setup(void) 
{
  Serial.begin(9600);
  while (!Serial) {
      // 시리얼 연결 기다림
      delay(1);
  }

  if (! ina219.begin()) {
    Serial.println("INA219 칩을 찾을 수 없습니다. 배선을 확인하세요!");
    while (1) { delay(10); }
  }

  Serial.println("측정 시작...");
}

void loop(void) 
{
  float current_mA = 0;
  float loadvoltage = 0;

  // 전류 측정 (mA)
  current_mA = ina219.getCurrent_mA();
  
  // 전압 측정 (V) - PV가 주는 전압
  loadvoltage = ina219.getBusVoltage_V() + (ina219.getShuntVoltage_mV() / 1000);

  Serial.print("생성 전압: "); Serial.print(loadvoltage); Serial.print(" V  |  ");
  Serial.print("충전 전류: "); Serial.print(current_mA); Serial.println(" mA");

  delay(1000); // 1초마다 갱신
}