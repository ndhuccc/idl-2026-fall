"""pytest 設定:決定要測試 starter 或 solution。

用法:
    pytest labs/lab01/tests                    # 預設測試 starter(學生自己的實作)
    LAB_TARGET=solution pytest labs/lab01/tests   # 教師驗證解答
"""

import importlib.util
import os
import pathlib

import pytest

LAB_DIR = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def lab():
    """動態載入 lab01 模組(依環境變數 LAB_TARGET 選擇資料夾)。"""
    target = os.environ.get("LAB_TARGET", "starter")
    path = LAB_DIR / target / "lab01.py"
    spec = importlib.util.spec_from_file_location("lab01_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
