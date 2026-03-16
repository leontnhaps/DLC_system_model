# Com_refactor 테스트 실행 방법

아래 명령을 저장소 루트에서 실행하세요.

```bash
uv run python -m compileall Com_refactor
uv run python Com_refactor/tests/test_naming.py
uv run python Com_refactor/tests/test_router.py
```

---

### 새로 분리한 것
- **app**: 앱 상태 / 설정 / 메인 윈도우 / 이벤트 핸들러 / 헬퍼
- **infra**: 이벤트 버스 / 이미지 라우터 / 네트워크 클라이언트 / 프로토콜 상수
- **ui**: 프리뷰 프레임 및 탭 UI
- **utils**: 범용 유틸리티 (예: naming, threading)
- **vision**: YOLO / MOT / scan controller / LED filter
- **workflows**: Pointing / Scan / Scheduling workflow 경계
- **scheduling**: 향후 scheduling 알고리즘 확장을 위한 기반

---

## 1. 폴더 구조

```text
Com_refactor/
  app/
    config.py
    state.py
    helpers.py
    event_handlers.py
    window.py

  infra/
    event_bus.py
    image_router.py
    network_client.py
    protocol.py

  ui/
    preview_frame.py
    scan_tab.py
    test_settings_tab.py
    pointing_tab.py
    scheduling_tab.py

  utils/
    naming.py
    threading.py

  vision/
    yolo_utils.py
    mot.py
    scan_controller.py
    led_filter.py

  workflows/
    pointing_workflow.py
    scan_workflow.py
    scheduling_workflow.py

  scheduling/
    base.py
    round_robin.py

  tests/
    ... stdlib-only smoke / wrapper tests ...

  # compatibility wrappers
  Com_main.py
  app_config.py
  app_state.py
  app_helpers.py
  event_handlers.py
  image_router.py
  infra_event_bus.py
  led_filter.py
  mot.py
  naming.py
  network.py
  pointing_handler.py
  scan_controller.py
  ui_components.py
  yolo_utils.py
```

---

## 2. 패키지별 역할

### `app/`
애플리케이션 레벨 로직을 담당합니다.

- `config.py` : GUI/네트워크/저장 관련 기본 설정
- `state.py` : `AppState` 등 현재 상태 보관
- `helpers.py` : 앱 보조 로직
- `event_handlers.py` : 이벤트 폴링 / 이벤트 처리
- `window.py` : 메인 GUI 조립 및 엔트리 로직 본체

### `infra/`
입출력/연결/프로토콜 관련 인프라 계층입니다.

- `event_bus.py` : UI 이벤트 전달용 버스
- `image_router.py` : 수신 이미지 라우팅
- `network_client.py` : GUI ↔ Server 통신 클라이언트
- `protocol.py` : 프로토콜 문자열/빌더 정의

### `ui/`
Tkinter 기반 UI 컴포넌트를 담당합니다.

- `preview_frame.py`
- `scan_tab.py`
- `test_settings_tab.py`
- `pointing_tab.py`
- `scheduling_tab.py`

### `utils/`
범용 유틸리티입니다.

- `naming.py` : 이미지 이름 규칙 파싱/재사용
- `threading.py` : UI thread-safe helper

### `vision/`
비전 처리 계층입니다.

- `yolo_utils.py` : YOLO 로딩/타일링/NMS
- `mot.py` : Multi-Object Tracking
- `scan_controller.py` : 스캔 세션/CSV/워커 처리
- `led_filter.py` : LED 상태 판정 보조

### `workflows/`
실행 흐름 경계를 분리하기 위한 계층입니다.

- `pointing_workflow.py`
- `scan_workflow.py`
- `scheduling_workflow.py`

> 주의: workflow는 기존 기능을 깨지 않기 위해 점진적으로 연결되었으며,  
> 실제 하드웨어 실행 시 세부 동작 확인이 필요합니다.

### `scheduling/`
향후 스케줄링 알고리즘 확장을 위한 영역입니다.

- `base.py` : 공통 인터페이스
- `round_robin.py` : 기본 라운드로빈 구현

> 현재 목표는 **알고리즘 확장 기반 준비**이며,  
> 알고리즘 변경/추가는 하드웨어 실험 및 검증 이후 진행하는 것이 권장됩니다.


---

## 3. 실행 방법

### 전체 실행 순서

#### 1) Relay Server 실행
```bash
python Server/Server_main.py
```

#### 2) Raspberry Pi Agent 실행
```bash
python Raspberrypi/Rasp_main.py
```

#### 3) GUI 실행 (`Com_refactor` 기준)
```bash
python Com_refactor/Com_main.py
```

### 실행 목적
- 기존 `Com/Com_main.py` 대신
- 리팩토링된 구조의 GUI 후보를 실제로 검증하기 위함

---


## 4. 하드웨어 적용 전 체크리스트

### GUI
- [ ] `python Com_refactor/Com_main.py`로 정상 실행
- [ ] 모든 탭이 정상 표시
- [ ] 버튼 바인딩 정상

### Preview
- [ ] Server 연결 정상
- [ ] Pi Agent 연결 정상
- [ ] Preview 프레임 수신 정상

### Scan
- [ ] Scan 시작/중지 정상
- [ ] progress 업데이트 정상
- [ ] 이미지 저장 / CSV 저장 정상

### Pointing
- [ ] 타깃 계산 정상
- [ ] 이미지 라우팅 (`pointing_*`) 정상
- [ ] aiming loop 정상

### Scheduling
- [ ] 기존 Scheduling 동작 정상
- [ ] 기존 Round Robin 흐름이 깨지지 않음
- [ ] `SchedulingWorkflow` wiring이 기능을 해치지 않음

### 의존성
- [ ] `cv2`, `numpy`, `scipy`, `PIL`, `ultralytics` 설치 확인
- [ ] 경로/권한 문제 없음
- [ ] `captures/` 저장 가능

---
