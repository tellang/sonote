"""PyInstaller / python -m src 부트스트랩 엔트리포인트."""
import multiprocessing
import os
import sys

from src.runtime.context import is_frozen

multiprocessing.freeze_support()

# PyInstaller console=False 빌드에서 sys.stdout/stderr가 None이 되는 문제 방지
# uvicorn 로깅이 isatty() 호출 시 AttributeError 발생하므로 devnull로 대체
if is_frozen():
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

if __name__ == "__main__":
    from src.desktop import main

    main()
