"""基础资源（机型/场景/技能）删除保护的接口测试。

覆盖：三类资源、多种引用来源（现役作业、数据集版本、历史统计）、
并发新增与删除竞争、以及重启后历史数据仍可读。
"""

import os
import tempfile

# 必须在导入应用之前指向临时数据库，测试只使用临时 SQLite 文件。
_DB_DIR = tempfile.mkdtemp(prefix="robot_resource_delete_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_DB_DIR, 'test.db')}"

import threading

import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import settings
from app.database import SessionLocal, engine
from app.models import (
    Annotation,
    Dataset,
    DatasetItem,
    DatasetReuse,
    DatasetReview,
    DatasetSubscription,
    DatasetVersion,
    OperationData,
    RobotModel,
    Scene,
    Skill,
)

PREFIX = settings.API_V1_PREFIX

REFERENCE_KEYS = {
    "operations",
    "datasets",
    "dataset_versions",
    "annotations",
    "dataset_reuses",
}


@pytest.fixture(autouse=True)
def clean_tables():
    db = SessionLocal()
    try:
        for model in (
            DatasetReview,
            DatasetSubscription,
            DatasetReuse,
            DatasetItem,
            DatasetVersion,
            Annotation,
            Dataset,
            OperationData,
            RobotModel,
            Scene,
            Skill,
        ):
            db.query(model).delete()
        db.commit()
    finally:
        db.close()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _create_trio(client, suffix=""):
    rm = client.post(
        f"{PREFIX}/robot-models",
        json={"name": f"机型{suffix}", "manufacturer": "ACME"},
    ).json()
    sc = client.post(
        f"{PREFIX}/scenes", json={"name": f"场景{suffix}", "category": "生产制造"}
    ).json()
    sk = client.post(
        f"{PREFIX}/skills", json={"name": f"技能{suffix}", "category": "操作"}
    ).json()
    return rm, sc, sk


def _operation_payload(rm_id, sc_id, sk_id):
    return {
        "robot_model_id": rm_id,
        "scene_id": sc_id,
        "skill_id": sk_id,
        "motion_trajectory": {"waypoints": []},
        "perception_records": {"camera_images_captured": 1},
        "timestamp_start": "2026-01-01T00:00:00Z",
        "timestamp_end": "2026-01-01T00:01:00Z",
    }


def _create_operation(client, rm, sc, sk):
    resp = client.post(
        f"{PREFIX}/operations", json=_operation_payload(rm["id"], sc["id"], sk["id"])
    )
    assert resp.status_code == 200
    return resp.json()


def _create_dataset(client, rm, sc, sk, suffix="", operation_ids=None):
    payload = {
        "name": f"数据集{suffix}",
        "robot_model_id": rm["id"],
        "scene_id": sc["id"],
        "skill_id": sk["id"],
        "owner_team": "数据组",
    }
    if operation_ids:
        payload["operation_data_ids"] = operation_ids
    resp = client.post(f"{PREFIX}/datasets", json=payload)
    assert resp.status_code == 200
    return resp.json()


def _assert_conflict_shape(body, resource_type, resource_id):
    detail = body["detail"]
    assert detail["resource_type"] == resource_type
    assert detail["resource_id"] == resource_id
    assert detail["message"]
    assert set(detail["references"].keys()) == REFERENCE_KEYS
    assert all(isinstance(v, int) and v >= 0 for v in detail["references"].values())
    return detail


# ---------------------------------------------------------------------------
# 三类资源：被现役作业引用时返回稳定的 409
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resource_type,path_segment,label",
    [
        ("robot_model", "robot-models", "机型"),
        ("scene", "scenes", "场景"),
        ("skill", "skills", "技能"),
    ],
)
def test_delete_referenced_by_operation_returns_conflict(
    client, resource_type, path_segment, label
):
    rm, sc, sk = _create_trio(client, suffix=f"-{resource_type}")
    _create_operation(client, rm, sc, sk)

    resource = {"robot_model": rm, "scene": sc, "skill": sk}[resource_type]
    resp = client.delete(f"{PREFIX}/{path_segment}/{resource['id']}")

    assert resp.status_code == 409
    detail = _assert_conflict_shape(resp.json(), resource_type, resource["id"])
    assert detail["references"]["operations"] == 1
    assert detail["references"]["datasets"] == 0

    # 资源未被删除，历史查询仍能看到名称
    assert client.get(f"{PREFIX}/{path_segment}/{resource['id']}").status_code == 200


def test_conflict_response_shape_is_stable_across_resource_types(client):
    rm, sc, sk = _create_trio(client, suffix="-stable")
    _create_operation(client, rm, sc, sk)

    shapes = []
    for path_segment, resource in (
        ("robot-models", rm),
        ("scenes", sc),
        ("skills", sk),
    ):
        resp = client.delete(f"{PREFIX}/{path_segment}/{resource['id']}")
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        shapes.append((set(detail.keys()), set(detail["references"].keys())))

    assert shapes[0] == shapes[1] == shapes[2]


# ---------------------------------------------------------------------------
# 多个引用来源：数据集、数据集版本、标注与复用等历史统计
# ---------------------------------------------------------------------------


def test_delete_skill_referenced_only_by_dataset_returns_conflict(client):
    """技能只被数据集引用时（历史上下载/统计仍需要它），同样禁止删除。"""
    rm, sc, sk = _create_trio(client, suffix="-dataset-skill")
    _create_dataset(client, rm, sc, sk, suffix="-dataset-skill")

    resp = client.delete(f"{PREFIX}/skills/{sk['id']}")

    assert resp.status_code == 409
    detail = _assert_conflict_shape(resp.json(), "skill", sk["id"])
    assert detail["references"]["operations"] == 0
    assert detail["references"]["datasets"] == 1
    assert detail["references"]["dataset_versions"] == 1

    # 数据集仍引用该技能，技能记录必须保留
    assert client.get(f"{PREFIX}/skills/{sk['id']}").status_code == 200


def test_impact_summary_covers_versions_annotations_and_reuses(client):
    rm, sc, sk = _create_trio(client, suffix="-full")
    op = _create_operation(client, rm, sc, sk)

    annotation = client.post(
        f"{PREFIX}/annotations",
        json={"operation_data_id": op["id"], "is_success": True},
    )
    assert annotation.status_code == 200

    dataset = _create_dataset(client, rm, sc, sk, suffix="-full", operation_ids=[op["id"]])
    submit = client.post(f"{PREFIX}/datasets/{dataset['id']}/review", json={"action": "submit"})
    assert submit.status_code == 200
    approve = client.post(
        f"{PREFIX}/datasets/{dataset['id']}/review",
        json={"action": "approve", "reviewer": "审核员"},
    )
    assert approve.status_code == 200
    reuse = client.post(
        f"{PREFIX}/dataset-reuses",
        json={"dataset_id": dataset["id"], "reusing_team": "算法组"},
    )
    assert reuse.status_code == 200

    resp = client.delete(f"{PREFIX}/robot-models/{rm['id']}")

    assert resp.status_code == 409
    detail = _assert_conflict_shape(resp.json(), "robot_model", rm["id"])
    refs = detail["references"]
    assert refs["operations"] == 1
    assert refs["datasets"] == 1
    assert refs["dataset_versions"] >= 2  # 初始版本 + 审核通过快照
    assert refs["annotations"] == 1
    assert refs["dataset_reuses"] == 1

    # 场景同样被数据集与作业引用，删除也被拒绝
    scene_resp = client.delete(f"{PREFIX}/scenes/{sc['id']}")
    assert scene_resp.status_code == 409
    scene_refs = scene_resp.json()["detail"]["references"]
    assert scene_refs["operations"] == 1
    assert scene_refs["datasets"] == 1


# ---------------------------------------------------------------------------
# 无引用正常删除、重复删除按未找到处理、引用解除后可删除
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_segment,label",
    [("robot-models", "机型"), ("scenes", "场景"), ("skills", "技能")],
)
def test_delete_unreferenced_succeeds_and_repeat_returns_not_found(
    client, path_segment, label
):
    rm, sc, sk = _create_trio(client, suffix=f"-free-{path_segment}")
    resource = {"robot-models": rm, "scenes": sc, "skills": sk}[path_segment]

    resp = client.delete(f"{PREFIX}/{path_segment}/{resource['id']}")
    assert resp.status_code == 200
    assert resp.json() == {"message": "删除成功"}

    assert client.get(f"{PREFIX}/{path_segment}/{resource['id']}").status_code == 404

    again = client.delete(f"{PREFIX}/{path_segment}/{resource['id']}")
    assert again.status_code == 404
    assert again.json()["detail"] == f"{label}不存在"


def test_delete_succeeds_after_references_removed(client):
    rm, sc, sk = _create_trio(client, suffix="-cleanup")
    op = _create_operation(client, rm, sc, sk)
    dataset = _create_dataset(client, rm, sc, sk, suffix="-cleanup")

    assert client.delete(f"{PREFIX}/robot-models/{rm['id']}").status_code == 409

    assert client.delete(f"{PREFIX}/datasets/{dataset['id']}").status_code == 200
    assert client.delete(f"{PREFIX}/operations/{op['id']}").status_code == 200

    resp = client.delete(f"{PREFIX}/robot-models/{rm['id']}")
    assert resp.status_code == 200
    assert client.delete(f"{PREFIX}/scenes/{sc['id']}").status_code == 200
    assert client.delete(f"{PREFIX}/skills/{sk['id']}").status_code == 200


# ---------------------------------------------------------------------------
# 并发竞争：检查与删除之间不能被并发新增作业绕过
# ---------------------------------------------------------------------------


def test_concurrent_create_operation_cannot_bypass_delete():
    for round_no in range(6):
        setup = TestClient(app)
        rm, sc, sk = _create_trio(setup, suffix=f"-race-{round_no}")

        barrier = threading.Barrier(2)
        outcome = {}

        def do_delete():
            thread_client = TestClient(app)
            barrier.wait(timeout=10)
            outcome["delete"] = thread_client.delete(
                f"{PREFIX}/robot-models/{rm['id']}"
            )

        def do_create():
            thread_client = TestClient(app)
            barrier.wait(timeout=10)
            outcome["create"] = thread_client.post(
                f"{PREFIX}/operations",
                json=_operation_payload(rm["id"], sc["id"], sk["id"]),
            )

        threads = [
            threading.Thread(target=do_delete),
            threading.Thread(target=do_create),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert all(not thread.is_alive() for thread in threads)

        delete_resp = outcome["delete"]
        create_resp = outcome["create"]

        # 不允许出现服务器错误
        assert delete_resp.status_code in (200, 409)
        assert create_resp.status_code in (200, 400)
        # 结果必须互补：删除成功则新增失败，新增成功则删除冲突
        assert (delete_resp.status_code == 200) == (create_resp.status_code == 400)

        verify = TestClient(app)
        if delete_resp.status_code == 200:
            assert verify.get(f"{PREFIX}/robot-models/{rm['id']}").status_code == 404
            # 不存在删除成功后留下的悬空作业引用
            ops = verify.get(
                f"{PREFIX}/operations", params={"robot_model_id": rm["id"]}
            ).json()
            assert ops["total"] == 0
        else:
            assert verify.get(f"{PREFIX}/robot-models/{rm['id']}").status_code == 200
            detail = delete_resp.json()["detail"]
            assert detail["references"]["operations"] >= 1
            # 冲突结果稳定：再次删除仍是 409
            assert (
                verify.delete(f"{PREFIX}/robot-models/{rm['id']}").status_code == 409
            )


# ---------------------------------------------------------------------------
# 重启后历史数据仍可读
# ---------------------------------------------------------------------------


def test_historical_data_remains_readable_after_restart(client):
    rm, sc, sk = _create_trio(client, suffix="-restart")
    op = _create_operation(client, rm, sc, sk)
    client.post(
        f"{PREFIX}/annotations",
        json={"operation_data_id": op["id"], "is_success": False, "failure_category": "感知异常"},
    )
    dataset = _create_dataset(client, rm, sc, sk, suffix="-restart", operation_ids=[op["id"]])

    # 删除被引用保护拦截
    assert client.delete(f"{PREFIX}/robot-models/{rm['id']}").status_code == 409
    assert client.delete(f"{PREFIX}/scenes/{sc['id']}").status_code == 409
    assert client.delete(f"{PREFIX}/skills/{sk['id']}").status_code == 409

    # 模拟重启：丢弃全部数据库连接，后续请求重新建连读取同一数据库文件
    engine.dispose()
    restarted = TestClient(app)

    stats = restarted.get(f"{PREFIX}/stats/by-robot-model").json()
    entry = next(item for item in stats if item["robot_model_id"] == rm["id"])
    assert entry["robot_model_name"] == rm["name"]
    assert entry["total_data_count"] == 1
    assert entry["annotated_count"] == 1

    scene_stats = restarted.get(f"{PREFIX}/stats/by-scene").json()
    scene_entry = next(item for item in scene_stats if item["scene_id"] == sc["id"])
    assert scene_entry["scene_name"] == sc["name"]
    assert scene_entry["total_data_count"] == 1

    reloaded_op = restarted.get(f"{PREFIX}/operations/{op['id']}").json()
    assert reloaded_op["robot_model_id"] == rm["id"]
    assert reloaded_op["scene_id"] == sc["id"]
    assert reloaded_op["skill_id"] == sk["id"]

    reloaded_dataset = restarted.get(f"{PREFIX}/datasets/{dataset['id']}").json()
    assert reloaded_dataset["name"] == dataset["name"]
    versions = restarted.get(f"{PREFIX}/datasets/{dataset['id']}/versions").json()
    assert len(versions) >= 1

    # 重启后引用保护仍然生效（新连接同样开启外键约束）
    assert restarted.delete(f"{PREFIX}/robot-models/{rm['id']}").status_code == 409
    assert restarted.delete(f"{PREFIX}/skills/{sk['id']}").status_code == 409
