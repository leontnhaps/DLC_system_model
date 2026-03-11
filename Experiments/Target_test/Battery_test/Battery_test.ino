int count = 0;       // 측정 횟수를 세는 변수
float sum_aR = 0.0;  // 10번의 아날로그 값을 누적해서 더할 변수

void setup() {
  Serial.begin(9600);
}

void loop() {
  // A0핀 읽기
  float aR = analogRead(A0);
  
  // 읽은 값을 누적해서 더하고 횟수를 1 증가
  sum_aR = sum_aR + aR; 
  count++;

  // 측정 횟수가 10번(10초)이 되면 평균을 계산하고 출력
  if (count >= 10) {
    // 평균 아날로그 값 계산
    float avg_aR = sum_aR / 10.0;
    
    // 평균 아날로그 값을 전압으로 변환 (전압분배 2배 적용)
    float voltage = avg_aR * (5.0 / 1023.0) * 2.0; 

    Serial.println("=== 10초 평균 데이터 ===");
    Serial.print("평균 analogRead : ");
    Serial.println(avg_aR);

    Serial.print("평균 최종 전압 : ");
    Serial.println(voltage);
    Serial.println("========================");
    
    // 다음 10초 측정을 위해 누적 값과 카운트를 초기화
    sum_aR = 0.0;
    count = 0;
  }
  
  delay(1000); // 1초 대기
}