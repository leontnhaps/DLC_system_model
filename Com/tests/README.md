# Com 테스트 실행 방법

아래 명령을 저장소 루트에서 실행하세요.

```bash
uv run python -m compileall Com
uv run python -m unittest discover -s Com/tests
```

개별 smoke/wrapper 테스트만 확인할 때는 필요한 파일을 직접 실행할 수 있습니다.

```bash
uv run python Com/tests/test_naming.py
uv run python Com/tests/test_router.py
uv run python Com/tests/test_scheduling_proposed.py
```

---

## 현재 `Com/` 구조

`Com/Com_main.py`는 GUI 실행 진입점이고, 대부분의 루트 모듈은 기존 import를 유지하기 위한 compatibility wrapper입니다. 실제 구현은 아래 패키지들에 나뉘어 있습니다.

- `app/`: 앱 상태, 설정, 메인 윈도우, 이벤트 핸들러, helper
- `infra/`: 이벤트 버스, 이미지 라우터, 네트워크 클라이언트, 프로토콜
- `ui/`: Tkinter preview frame 및 각 탭 UI
- `utils/`: 이름 파싱, UI thread helper 등 범용 유틸리티
- `vision/`: YOLO, MOT, scan controller, LED filter
- `workflows/`: Scan, Pointing, Scheduling workflow 경계
- `scheduling/`: RoundRobin/Proposed scheduling 알고리즘
- `tests/`: stdlib 기반 테스트

---

## 주요 확인 포인트

- GUI wrapper가 실제 구현 모듈을 그대로 re-export하는지
- `infra`, `ui`, `utils`, `vision`, `workflows`, `scheduling` 패키지 import가 깨지지 않는지
- `ScanWorkflow`, `SchedulingWorkflow`가 기존 controller/algorithm을 감싸는 방식으로 연결되어 있는지
- `RoundRobinScheduler`, `ProposedScheduler`의 기본 선택/할당 로직이 기대대로 동작하는지
- 테스트는 하드웨어 없이 실행 가능한 범위의 smoke/unit 검증에 집중하는지

---

## 하드웨어 적용 전 체크리스트

- [ ] `python Com/Com_main.py`로 GUI 실행
- [ ] Server 연결 정상
- [ ] Pi Agent 연결 정상
- [ ] Preview 프레임 수신 정상
- [ ] Scan 시작/중지 및 progress 업데이트 정상
- [ ] 이미지 저장, CSV 저장, similarity log 저장 정상
- [ ] Pointing target 계산 및 aiming loop 정상
- [ ] RoundRobin scheduling 동작 정상
- [ ] Proposed scheduling 동작 정상
- [ ] `cv2`, `numpy`, `scipy`, `PIL`, `ultralytics` 설치 확인
- [ ] `captures/` 저장 권한 확인
