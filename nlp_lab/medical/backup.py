"""Consistent SQLite snapshots and safe restore copies."""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
from contextlib import closing

from .store import Store


REQUIRED_TABLES = {"documents", "extractions", "reviews", "evidence_search"}


def validate_database(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite 文件不存在：{path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=20)) as db:
            results = [row[0] for row in db.execute("PRAGMA quick_check")]
            if not results or any(value != "ok" for value in results):
                raise ValueError("SQLite 完整性校验失败")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > Store.schema_version:
                raise ValueError(f"数据库版本 {version} 高于当前程序支持的版本 {Store.schema_version}")
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
            missing = REQUIRED_TABLES - tables
            if missing:
                raise ValueError("SQLite 缺少工作台所需数据表：" + ", ".join(sorted(missing)))
    except sqlite3.DatabaseError as exc:
        raise ValueError("文件不是可读取的 SQLite 文献数据库") from exc
    return {"integrity": "ok", "schema_version": version}


def create_snapshot(source: Path, destination: Path) -> dict:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("文献数据库或备份文件不存在")
    if source == destination:
        raise ValueError("目标不能与源数据库相同")
    if destination.exists():
        raise FileExistsError("目标文件已存在；请使用新的、未使用的文件名")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with closing(sqlite3.connect(source, timeout=20)) as original, closing(sqlite3.connect(temporary)) as snapshot:
            original.backup(snapshot)
        metadata = validate_database(temporary)
        if os.name == "nt":
            # Same-volume rename is atomic and fails rather than overwriting.
            os.rename(temporary, destination)
        else:
            # Atomic no-clobber publication on the same filesystem.
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), **metadata}
