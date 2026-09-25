"""基础资源删除保护的接口测试。

覆盖：
- 三类资源（机型/场景/技能）无引用时正常删除、重复删除返回 404；
- 多个引用来源（现役作业、数据集及版本快照、历史复用统计）返回 409 与有限摘要；
- 检查与删除之间的并发新增作业竞争不会绕过保护；
- 重启（重新建客户端/引擎指向同一文件）后历史数据仍可读，统计名称不缺失。
"""

import threading
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event

import app.database as db_module
from app.database import Base, SessionLocal, apply_sqlite_pragmas
from app.models import OperationData, RobotModel
from main import app

API = "/api/v1"


# ---------------------------------------------------------------- helpers


def _create_base_resources(client, prefix=""):
    model = client.post(
        f"{API}/robot-models",
        json={"name": f"机型-{prefix}", "manufacturer": "Acme"},
    ).json()
    scene = client.post(
        f"{API}/scenes",
        json={"name": f"场景-{prefix}", "category": "生产制造"},
    ).json()
    skill = client.post(
        f"{API}/skills",
        json={"name": f"技能-{prefix}", "category": "操作"},
    ).json()
    return model, scene, skill


def _create_operation(client, model_id, scene_id, skill_id):
    resp = client.post(
        f"{API}/operations",
        json={
            "robot_model_id": model_id,
            "scene_id": scene_id,
            "skill_id": skill_id,
            "motion_trajectory": {"waypoints": [{"x": 1}]},
            "perception_records": {"camera_images_captured": 3},
            "grasp_result": {"attempted": True, "success": True},
            "timestamp_start": "2025-01-01T00:00:00Z",
            "timestamp_end": "2025-01-01T00:01:00Z",
            "duration_ms": 60000,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _create_dataset(client, model_id, scene_id, skill_id, name="数据集"):
    resp = client.post(
        f"{API}/datasets",
        json={
            "name": name,
            "robot_model_id": model_id,
            "scene_id": scene_id,
            "skill_id": skill_id,
            "owner_team": "数据组",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------- no references


@pytest.mark.parametrize("kind", ["robot-models", "scenes", "skills"])
def test_delete_without_references_succeeds(client, kind):
    model, scene, skill = _create_base_resources(client, "free")
    resource_id = {"robot-models": model["id"], "scenes": scene["id"], "skills": skill["id"]}[kind]

    resp = client.delete(f"{API}/{kind}/{resource_id}")
    assert resp.status_code == 200
    assert resp.json() == {"message": "删除成功"}

    # 重复删除沿用既有未找到语义
    again = client.delete(f"{API}/{kind}/{resource_id}")
    assert again.status_code == 404


@pytest.mark.parametrize("kind", ["robot-models", "scenes", "skills"])
def test_delete_missing_returns_404(client, kind):
    resp = client.delete(f"{API}/{kind}/99999")
    assert resp.status_code == 404


# ------------------------------------------------- operation references


@pytest.mark.parametrize("kind,key", [
    ("robot-models", "robot_model_id"),
    ("scenes", "scene_id"),
    ("skills", "skill_id"),
])
def test_delete_referenced_by_operation_conflicts(client, kind, key):
    model, scene, skill = _create_base_resources(client, "op")
    op = _create_operation(client, model["id"], scene["id"], skill["id"])
    resource_id = {"robot-models": model["id"], "scenes": scene["id"], "skills": skill["id"]}[kind]

    resp = client.delete(f"{API}/{kind}/{resource_id}")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "resource_in_use"
    assert detail["resource_type"] == {"robot-models": "robot_model",
                                       "scenes": "scene",
                                       "skills": "skill"}[kind]
    assert detail["references"]["operation_count"] == 1
    assert detail["references"]["dataset_count"] == 0
    assert op["id"] in detail["sample_operation_ids"]
    assert detail["resource_name"]

    # 资源仍在
    assert client.get(f"{API}/{kind}/{resource_id}").status_code == 200


# -------------------------------------------------- dataset references


@pytest.mark.parametrize("kind", ["robot-models", "scenes", "skills"])
def test_delete_referenced_only_by_dataset_conflicts(client, kind):
    # 技能被数据集引用此前会被误删（可空外键），这里三类统一拦截
    model, scene, skill = _create_base_resources(client, "ds")
    dataset = _create_dataset(client, model["id"], scene["id"], skill["id"], name="引用数据集")
    resource_id = {"robot-models": model["id"], "scenes": scene["id"], "skills": skill["id"]}[kind]

    resp = client.delete(f"{API}/{kind}/{resource_id}")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    refs = detail["references"]
    assert refs["operation_count"] == 0
    assert refs["dataset_count"] == 1
    # 创建数据集即产生初始版本快照
    assert refs["dataset_version_count"] == 1
    assert refs["reuse_count"] == 0
    assert {"id": str(dataset["id"]), "name": "引用数据集"} in detail["sample_datasets"]

    # 数据集本身完好，列表接口不再缺名/悬空
    listed = client.get(f"{API}/datasets").json()
    assert any(item["id"] == dataset["id"] for item in listed)


def test_conflict_summary_includes_versions_and_reuse_history(client):
    model, scene, skill = _create_base_resources(client, "hist")
    op = _create_operation(client, model["id"], scene["id"], skill["id"])
    dataset = _create_dataset(client, model["id"], scene["id"], skill["id"], name="历史数据集")

    # 将作业加入数据集，使版本快照统计包含作业
    added = client.post(
        f"{API}/datasets/{dataset['id']}/items",
        json={"operation_data_ids": [op["id"]]},
    )
    assert added.status_code == 200
    # 提交审核并通过 -> 产生新版本快照
    assert client.post(
        f"{API}/datasets/{dataset['id']}/review", json={"action": "submit", "reviewer": "r"}
    ).status_code == 200
    assert client.post(
        f"{API}/datasets/{dataset['id']}/review", json={"action": "approve", "reviewer": "r"}
    ).status_code == 200
    # 产生一条复用历史（历史统计来源）
    reuse = client.post(
        f"{API}/dataset-reuses",
        json={"dataset_id": dataset["id"], "reusing_team": "复用组", "purpose": "训练"},
    )
    assert reuse.status_code == 200, reuse.text

    resp = client.delete(f"{API}/skills/{skill['id']}")
    assert resp.status_code == 409
    refs = resp.json()["detail"]["references"]
    assert refs["operation_count"] == 1
    assert refs["dataset_count"] == 1
    assert refs["dataset_version_count"] == 2
    assert refs["reuse_count"] == 1


def test_summary_is_bounded(client):
    """影响摘要只包含有限样本，不做全量导出。"""
    model, scene, skill = _create_base_resources(client, "bound")
    for i in range(8):
        _create_operation(client, model["id"], scene["id"], skill["id"])

    resp = client.delete(f"{API}/robot-models/{model['id']}")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["references"]["operation_count"] == 8
    assert len(detail["sample_operation_ids"]) <= 5


# ------------------------------------------------- multiple sources


def test_all_three_reference_sources_block_each_resource(client):
    """一个机型同时被作业、数据集、历史统计引用时，三类计数同时呈现。"""
    model, scene, skill = _create_base_resources(client, "multi")
    op = _create_operation(client, model["id"], scene["id"], skill["id"])
    dataset = _create_dataset(client, model["id"], scene["id"], skill["id"])
    client.post(f"{API}/datasets/{dataset['id']}/items", json={"operation_data_ids": [op["id"]]})
    client.post(f"{API}/datasets/{dataset['id']}/review", json={"action": "submit"})
    client.post(f"{API}/datasets/{dataset['id']}/review", json={"action": "approve"})
    client.post(f"{API}/dataset-reuses", json={"dataset_id": dataset["id"], "reusing_team": "t"})

    for resource_id in (model["id"], scene["id"], skill["id"]):
        kind = {model["id"]: "robot-models", scene["id"]: "scenes", skill["id"]: "skills"}[resource_id]
        resp = client.delete(f"{API}/{kind}/{resource_id}")
        assert resp.status_code == 409
        refs = resp.json()["detail"]["references"]
        assert refs["operation_count"] >= 1
        assert refs["dataset_count"] >= 1
        assert refs["dataset_version_count"] >= 2
        assert refs["reuse_count"] >= 1


# ------------------------------------------------------- concurrency


def test_concurrent_insert_cannot_slip_between_check_and_delete(client):
    """确定性证明检查与删除之间的窗口被写锁关闭。

    删除事务持有 BEGIN IMMEDIATE 的 RESERVED 锁期间，并发作业插入只能排队；
    删除提交后，排队的插入被外键约束拒绝，不会产生悬空作业。
    """
    from app.database import immediate_session
    from app.services.resource_guard import delete_resource

    target_model = client.post(
        f"{API}/robot-models", json={"name": "race-target", "manufacturer": "Acme"}
    ).json()
    # 同一批有效的场景/技能供并发作业使用
    scene = client.post(f"{API}/scenes", json={"name": "race-scene", "category": "c"}).json()
    skill = client.post(f"{API}/skills", json={"name": "race-skill", "category": "c"}).json()

    insert_outcome = {}

    def insert_operation():
        # 与 POST /operations 相同：以 IMMEDIATE 开始，首条写语句即在写锁上排队
        try:
            with immediate_session() as session:
                session.add(OperationData(
                    robot_model_id=target_model["id"],
                    scene_id=scene["id"],
                    skill_id=skill["id"],
                    motion_trajectory={"w": 1},
                    perception_records={"p": 1},
                    timestamp_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    timestamp_end=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
                ))
                session.commit()  # 删除事务提交前在此排队
                insert_outcome["result"] = "inserted"
        except Exception as exc:
            insert_outcome["result"] = "rejected"
            insert_outcome["error"] = type(exc).__name__

    with immediate_session() as deleter:
        # 首条语句即取得 RESERVED 写锁，必须在启动并发写入之前完成
        assert deleter.get(RobotModel, target_model["id"]) is not None

        # 删除事务已持有写锁；此时启动并发插入，它必须等待
        worker = threading.Thread(target=insert_operation)
        worker.start()
        worker.join(timeout=1.0)
        assert worker.is_alive(), "并发插入应被写锁阻塞，而不是提前写入"

        # 检查（0 引用）与删除都在锁保护内完成
        delete_resource(deleter, "robot_model", target_model["id"])
        deleter.commit()

        worker.join(timeout=10)

    assert not worker.is_alive()
    assert insert_outcome["result"] == "rejected"
    assert insert_outcome["error"] == "IntegrityError"

    with SessionLocal() as session:
        assert session.get(RobotModel, target_model["id"]) is None
        assert session.query(OperationData).filter(
            OperationData.robot_model_id == target_model["id"]
        ).count() == 0


def test_insert_then_delete_under_lock_reports_conflict(client):
    """串行化的另一方向：并发插入先取得写锁并提交，随后的删除必须检出引用并返回 409。"""
    from app.database import immediate_session
    from app.services.resource_guard import (
        ResourceInUse,
        delete_resource,
    )

    target_model = client.post(
        f"{API}/robot-models", json={"name": "race-target-2", "manufacturer": "Acme"}
    ).json()
    scene = client.post(f"{API}/scenes", json={"name": "race-scene-2", "category": "c"}).json()
    skill = client.post(f"{API}/skills", json={"name": "race-skill-2", "category": "c"}).json()

    # 插入事务先取得写锁并提交作业
    with immediate_session() as writer:
        writer.add(OperationData(
            robot_model_id=target_model["id"],
            scene_id=scene["id"],
            skill_id=skill["id"],
            motion_trajectory={"w": 1},
            perception_records={"p": 1},
            timestamp_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            timestamp_end=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
        ))
        writer.commit()

    # 删除在自己的写事务内必然看到已提交的作业
    with immediate_session() as deleter:
        with pytest.raises(ResourceInUse) as exc_info:
            delete_resource(deleter, "robot_model", target_model["id"])
        deleter.rollback()

    assert exc_info.value.summary.references.operation_count == 1
    with SessionLocal() as session:
        assert session.get(RobotModel, target_model["id"]) is not None
        assert session.query(OperationData).filter(
            OperationData.robot_model_id == target_model["id"]
        ).count() == 1


def test_concurrent_delete_and_http_insert_end_state_consistent(db_engine):
    """经由真实 HTTP 接口的并发：删除与作业创建竞争，最终状态始终自洽。"""
    client = TestClient(app)
    target_model, scene, skill = _create_base_resources(client, "http-race")
    outcome = {}
    worker_holder = {}

    def create_operation():
        c = TestClient(app)
        resp = c.post(
            f"{API}/operations",
            json={
                "robot_model_id": target_model["id"],
                "scene_id": scene["id"],
                "skill_id": skill["id"],
                "motion_trajectory": {"w": 1},
                "perception_records": {"p": 1},
                "timestamp_start": "2025-01-01T00:00:00Z",
                "timestamp_end": "2025-01-01T00:01:00Z",
            },
        )
        outcome["status"] = resp.status_code
        c.close()

    fired = threading.Event()

    def fire_on_begin(conn):
        # 删除事务一开始（已取得写锁）即放出并发插入，不在这里等待，避免死锁
        if not fired.is_set():
            fired.set()
            thread = threading.Thread(target=create_operation)
            thread.start()
            worker_holder["thread"] = thread

    event.listen(db_engine, "begin", fire_on_begin)
    delete_resp = client.delete(f"{API}/robot-models/{target_model['id']}")
    # 必须先等 worker 结束所有数据库活动，再摘除监听器，避免并发修改监听器列表
    worker_holder["thread"].join(timeout=15)
    event.remove(db_engine, "begin", fire_on_begin)

    assert fired.is_set()
    with SessionLocal() as session:
        model_exists = session.get(RobotModel, target_model["id"]) is not None
        op_count = session.query(OperationData).filter(
            OperationData.robot_model_id == target_model["id"]
        ).count()

    # 不允许悬空：资源在则作业在（409），资源删则作业被拒（400）
    assert op_count == (1 if model_exists else 0)
    if model_exists:
        assert delete_resp.status_code == 409
        assert outcome["status"] == 200
    else:
        assert delete_resp.status_code == 200
        assert op_count == 0
        assert outcome["status"] == 400

    client.close()


# ------------------------------------------------- persistence/restart


def test_history_remains_readable_after_restart(tmp_path, monkeypatch):
    """进程重启（新建引擎指向同一库文件）后，被保留的历史数据与统计名称仍可读。"""
    db_path = tmp_path / "restart.db"

    def bind_engine():
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        apply_sqlite_pragmas(engine)
        Base.metadata.create_all(bind=engine)
        monkeypatch.setattr(db_module, "engine", engine)
        db_module.SessionLocal.configure(bind=engine)
        return engine

    engine_one = bind_engine()

    with TestClient(app) as first:
        model, scene, skill = _create_base_resources(first, "keep")
        op = _create_operation(first, model["id"], scene["id"], skill["id"])
        assert first.post(
            f"{API}/annotations",
            json={"operation_data_id": op["id"], "is_success": True, "annotator": "张工"},
        ).status_code == 200
        dataset = _create_dataset(first, model["id"], scene["id"], skill["id"], name="留存数据集")

        free_model, free_scene, free_skill = _create_base_resources(first, "disposable")
        assert first.delete(f"{API}/robot-models/{free_model['id']}").status_code == 200
        assert first.delete(f"{API}/scenes/{free_scene['id']}").status_code == 200
        assert first.delete(f"{API}/skills/{free_skill['id']}").status_code == 200

    # 模拟进程重启：释放旧引擎，用新连接打开同一个数据库文件
    engine_one.dispose()
    engine_two = bind_engine()

    with TestClient(app) as restarted:
        assert restarted.get(f"{API}/robot-models/{model['id']}").status_code == 200
        op_resp = restarted.get(f"{API}/operations/{op['id']}")
        assert op_resp.status_code == 200
        assert op_resp.json()["robot_model_id"] == model["id"]

        datasets = restarted.get(f"{API}/datasets").json()
        target = next(item for item in datasets if item["id"] == dataset["id"])
        assert target["name"] == "留存数据集"
        assert target["skill_id"] == skill["id"]

        stats = restarted.get(f"{API}/stats/by-robot-model").json()
        row = next(item for item in stats if item["robot_model_id"] == model["id"])
        assert row["robot_model_name"] == "机型-keep"
        assert row["total_data_count"] == 1
        assert row["dataset_count"] == 1

        scene_stats = restarted.get(f"{API}/stats/by-scene").json()
        scene_row = next(item for item in scene_stats if item["scene_id"] == scene["id"])
        assert scene_row["scene_name"] == "场景-keep"

        # 已删除资源重启后重复删除仍为 404
        assert restarted.delete(f"{API}/robot-models/{free_model['id']}").status_code == 404
        assert restarted.delete(f"{API}/scenes/{free_scene['id']}").status_code == 404
        assert restarted.delete(f"{API}/skills/{free_skill['id']}").status_code == 404

    engine_two.dispose()
