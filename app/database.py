from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite")


IS_SQLITE = is_sqlite_url(settings.DATABASE_URL)

# SQLite 写互斥时的等待上限：竞争事务在此时间内排队，而不是立刻报 locked
BUSY_TIMEOUT_MS = 8000

_connect_args = {"check_same_thread": False} if IS_SQLITE else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def apply_sqlite_pragmas(target_engine) -> None:
    """为 SQLite 引擎开启外键约束并接管事务开始方式。

    - foreign_keys=ON：删除被引用的基础资源时由数据库兜底，历史数据不会留下悬空外键；
    - busy_timeout：写锁冲突时排队等待，而非立即失败；
    - 关闭 pysqlite 的隐式 BEGIN，改由 begin 事件显式发起，使关键写事务可以用
      BEGIN IMMEDIATE 在第一时间取得 RESERVED 锁，消除“先检查后删除”的竞争窗口。
    """

    @event.listens_for(target_engine, "connect")
    def _sqlite_connect(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        cursor.close()
        dbapi_connection.isolation_level = None

    @event.listens_for(target_engine, "begin")
    def _sqlite_begin(conn):
        # immediate_session 会在连接上设置 IMMEDIATE 标记并在会话结束时清除，
        # 普通会话（无标记）使用 DEFERRED，互不影响
        mode = conn.info.get("transaction_begin_mode", "DEFERRED")
        conn.exec_driver_sql(f"BEGIN {mode}")


if IS_SQLITE:
    apply_sqlite_pragmas(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def immediate_session(bind=None):
    """提供一个以 BEGIN IMMEDIATE 开始的独立会话。

    会话内每次开启事务都使用 IMMEDIATE（数据集创建等会多次提交重开事务），
    退出时清除标记，保证连接归还连接池后不影响普通的 DEFERRED 请求。
    """
    bind = bind or engine
    with bind.connect() as connection:
        if is_sqlite_url(str(bind.url)):
            connection.info["transaction_begin_mode"] = "IMMEDIATE"
        try:
            with Session(bind=connection) as session:
                yield session
        finally:
            if is_sqlite_url(str(bind.url)):
                connection.info.pop("transaction_begin_mode", None)


def get_immediate_db():
    """写接口依赖：从事务一开始即取得写锁。

    “先读存在性、后插入”的接口若以普通 DEFERRED 事务运行，会先拿到 SHARED 锁，
    随后升级写锁时与并发写事务形成立即失败的锁升级冲突（busy_timeout 也不等待）。
    统一以 IMMEDIATE 开始后，写冲突在起点排队串行，既不产生 500，也不留悬空数据。
    """
    with immediate_session() as db:
        yield db
