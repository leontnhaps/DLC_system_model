# Com_refactor 테스트 실행 방법

아래 명령을 저장소 루트에서 실행하세요.

```bash
uv run python -m compileall Com_refactor
uv run python Com_refactor/tests/test_naming.py
uv run python Com_refactor/tests/test_router.py
```
