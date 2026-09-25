import os
import tempfile

# 必须在导入 app.database / main 之前指定数据库，避免应用建表时在工作目录生成 robot_data.db
_TEMP_DIR = tempfile.mkdtemp(prefix="robot-data-tests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEMP_DIR}/default.db")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

import app.database as db_module
from app.database import Base, apply_sqlite_pragmas
from main import app


@pytest.fixture
def db_engine(tmp_path):
    """每个测试一个独立的临时 SQLite 文件库（启用与生产一致的外键/锁设置）。"""
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    apply_sqlite_pragmas(engine)
    Base.metadata.create_all(bind=engine)

    original_engine = db_module.engine
    # 重绑现有工厂，而不是替换：测试中 `from app.database import SessionLocal`
    # 拿到的必须始终是同一个工厂对象
    db_module.engine = engine
    db_module.SessionLocal.configure(bind=engine)
    try:
        yield engine
    finally:
        db_module.SessionLocal.configure(bind=original_engine)
        db_module.engine = original_engine
        engine.dispose()


@pytest.fixture
def db_session(db_engine):
    from app.database import SessionLocal
    with SessionLocal() as session:
        yield session


@pytest.fixture
def client(db_engine):
    with TestClient(app) as test_client:
        yield test_client
