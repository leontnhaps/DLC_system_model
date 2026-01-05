void setup() {
  Serial.begin(9600);
}

void loop() {
  // A0핀 읽어서 바로 전압(V)으로 변환 후 출력
  // 5.0은 아두이노 기준 전압, 1023은 분해능
  float voltage = analogRead(A0) * (5.0 / 1023.0);
  voltage = voltage * 2 ; // 전압분배로 2배로확인
  Serial.println(voltage);
  delay(1000);
}