"""测试夹具：绕开本机 `tmp_path` 的可用性问题。

**为什么要覆盖 pytest 的 `tmp_path`**

pytest 的 `tmp_path` 走 `tempfile.mkdtemp()`，而在本机沙箱环境下：

- `mkdtemp` 创建的目录**不继承父目录 ACL** ⇒ 往里写文件会被拒
- 系统 `%TEMP%` 同样不可用

所以 `tmp_path` 在这里根本建不出来（`PermissionError: [WinError 5]`）。

**为什么手工 mkdir 反而可以**

`Path.mkdir()` 不指定权限位，新目录**继承父目录 ACL** ⇒ 可写。
差别就在这个继承上，与「有没有权限」无关。

**为什么不改测试**

测试里用 `tmp_path` 是 pytest 的通用写法，改掉会让这些用例在别的环境里
变得不标准。用一个 fixture 覆盖它，测试代码保持原样 —— 这也是
`echo-health` 仓里已经验证过的做法。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / ".test_tmp"


@pytest.fixture
def tmp_path(request) -> Path:
    """每个用例一个独立目录，落在仓库内的 `.test_tmp/`（已在 .gitignore）。"""
    safe_name = request.node.name.replace("/", "_").replace("::", "_")
    path = TMP_ROOT / safe_name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # 清理失败不影响结果（沙箱可能禁止删除；下次运行会先 rmtree）
    shutil.rmtree(path, ignore_errors=True)