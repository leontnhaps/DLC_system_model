const int ledPin = 8; // LED가 연결된 핀 번호를 8번으로 설정

void setup() {
  pinMode(ledPin, OUTPUT); // 8번 핀을 출력 모드로 설정
  digitalWrite(ledPin, HIGH); // 8번 핀에 전압을 가해 LED를 켬
}

void loop() {
  // LED를 켜둔 상태로 유지하므로 loop에는 아무것도 넣지 않아도 됩니다.
}