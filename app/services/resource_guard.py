"""基础资源（机型/场景/技能）的删除保护。

删除入口必须在同一条串行化写事务内完成“引用检查 + 删除”，避免检查与删除之间
被并发新增的作业或数据集绕过。引用来源统一为：

- 现役作业：operation_data 的直接外键；
- 数据集版本：datasets 行及其 dataset_versions 版本快照（经数据集挂接）；
- 历史统计：dataset_reuses 复用记录（历史统计数字由数据集携带）。

任一来源存在引用即拒绝删除并返回有限的影响摘要；无引用才物理删除。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from sqlalchemy.orm import Session

from app.models import (
    Dataset,
    DatasetReuse,
    DatasetVersion,
    OperationData,
    RobotModel,
    Scene,
    Skill,
)


class ResourceInUse(Exception):
    """资源仍被引用，无法删除。"""

    def __init__(self, resource_type: str, resource_name: Optional[str], summary: "ResourceReferenceSummary"):
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.summary = summary
        super().__init__(f"{resource_type} 仍被引用，无法删除")


class ResourceNotFound(Exception):
    """资源不存在（含已被删除的重复删除请求）。"""

    def __init__(self, resource_type: str):
        self.resource_type = resource_type
        super().__init__(f"{resource_type} 不存在")


# 摘要中逐类列出的样本数量上限：摘要只给出有限影响面，不做全量导出
SAMPLE_LIMIT = 5


@dataclass
class ReferenceSourceCount:
    # 现役作业引用数
    operation_count: int = 0
    # 数据集（及其版本快照）引用数
    dataset_count: int = 0
    # 历史版本快照数量
    dataset_version_count: int = 0
    # 历史复用记录数
    reuse_count: int = 0

    @property
    def total(self) -> int:
        return self.operation_count + self.dataset_count

    def as_dict(self) -> Dict[str, int]:
        return {
            "operation_count": self.operation_count,
            "dataset_count": self.dataset_count,
            "dataset_version_count": self.dataset_version_count,
            "reuse_count": self.reuse_count,
        }


@dataclass
class ResourceReferenceSummary:
    resource_type: str
    resource_id: int
    resource_name: Optional[str]
    references: ReferenceSourceCount = field(default_factory=ReferenceSourceCount)
    # 有限的影响样本：作业ID、数据集名称
    sample_operation_ids: List[int] = field(default_factory=list)
    sample_datasets: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "references": self.references.as_dict(),
            "sample_operation_ids": self.sample_operation_ids,
            "sample_datasets": self.sample_datasets,
        }


# 资源类型与（作业外键字段、数据集外键字段、中文标签）的映射
_RESOURCE_SPECS: Dict[str, Dict[str, object]] = {
    "robot_model": {
        "model": RobotModel,
        "operation_column": OperationData.robot_model_id,
        "dataset_column": Dataset.robot_model_id,
        "label": "机型",
    },
    "scene": {
        "model": Scene,
        "operation_column": OperationData.scene_id,
        "dataset_column": Dataset.scene_id,
        "label": "场景",
    },
    "skill": {
        "model": Skill,
        "operation_column": OperationData.skill_id,
        "dataset_column": Dataset.skill_id,
        "label": "技能",
    },
}


def _build_summary(
    db: Session,
    resource_type: str,
    resource_id: int,
    resource_name: Optional[str],
    operation_column,
    dataset_column,
) -> ResourceReferenceSummary:
    summary = ResourceReferenceSummary(
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource_name,
    )

    operation_ids = [
        row_id for (row_id,) in db.query(OperationData.id)
        .filter(operation_column == resource_id)
        .order_by(OperationData.id.asc())
        .limit(SAMPLE_LIMIT)
        .all()
    ]
    operation_count = db.query(OperationData.id).filter(operation_column == resource_id).count()
    summary.references.operation_count = operation_count
    summary.sample_operation_ids = operation_ids

    dataset_rows = (
        db.query(Dataset.id, Dataset.name)
        .filter(dataset_column == resource_id)
        .order_by(Dataset.id.asc())
        .all()
    )
    dataset_ids = [row.id for row in dataset_rows]
    summary.references.dataset_count = len(dataset_rows)
    summary.sample_datasets = [
        {"id": str(row.id), "name": row.name} for row in dataset_rows[:SAMPLE_LIMIT]
    ]

    if dataset_ids:
        summary.references.dataset_version_count = (
            db.query(DatasetVersion.id)
            .filter(DatasetVersion.dataset_id.in_(dataset_ids))
            .count()
        )
        summary.references.reuse_count = (
            db.query(DatasetReuse.id)
            .filter(DatasetReuse.dataset_id.in_(dataset_ids))
            .count()
        )

    return summary


def delete_resource(
    db: Session,
    resource_type: str,
    resource_id: int,
) -> Dict[str, object]:
    """在调用方给定的事务内执行引用检查与删除。

    调用方必须保证该事务以写锁开始（SQLite 下为 BEGIN IMMEDIATE），从而与并发的
    作业/数据集新增互相串行：对方要么在本事务提交后再写入（届时因外键失败），
    要么先写入、本事务的引用检查必然能看到。
    """
    spec = _RESOURCE_SPECS[resource_type]
    model: Type = spec["model"]  # type: ignore[assignment]

    resource = db.query(model).filter(model.id == resource_id).first()
    if resource is None:
        raise ResourceNotFound(str(spec["label"]))

    summary = _build_summary(
        db,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource.name,
        operation_column=spec["operation_column"],
        dataset_column=spec["dataset_column"],
    )
    if summary.references.total > 0:
        raise ResourceInUse(str(spec["label"]), resource.name, summary)

    db.delete(resource)
    return {"message": "删除成功", "resource_type": resource_type, "resource_id": resource_id}
