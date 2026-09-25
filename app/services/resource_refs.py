"""基础资源（机型/场景/技能）删除前的引用统计与冲突摘要。

三类资源共用同一套引用模型：

- 现役作业：``operation_data`` 直接引用资源；
- 数据集与数据集版本：``datasets`` 直接引用资源，其版本快照随之关联；
- 历史统计来源：作业的标注记录、数据集的复用记录，均由上述记录派生。

删除入口在事务内调用 :func:`collect_resource_references` 得到有限（固定字段、
仅计数）的影响摘要；存在引用时返回稳定的 409 响应，而不是让数据库约束异常
直接变成 500。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import (
    Annotation,
    Dataset,
    DatasetReuse,
    DatasetVersion,
    OperationData,
    RobotModel,
    Scene,
    Skill,
)


@dataclass(frozen=True)
class ResourceSpec:
    """一类基础资源的删除保护配置。"""

    resource_type: str
    label: str
    model: type
    operation_fk: object
    dataset_fk: object


RESOURCE_SPECS: Dict[str, ResourceSpec] = {
    "robot_model": ResourceSpec(
        resource_type="robot_model",
        label="机型",
        model=RobotModel,
        operation_fk=OperationData.robot_model_id,
        dataset_fk=Dataset.robot_model_id,
    ),
    "scene": ResourceSpec(
        resource_type="scene",
        label="场景",
        model=Scene,
        operation_fk=OperationData.scene_id,
        dataset_fk=Dataset.scene_id,
    ),
    "skill": ResourceSpec(
        resource_type="skill",
        label="技能",
        model=Skill,
        operation_fk=OperationData.skill_id,
        dataset_fk=Dataset.skill_id,
    ),
}


@dataclass(frozen=True)
class ResourceReferenceSummary:
    """一次删除前检查的有限影响摘要（仅计数，字段固定）。"""

    resource_type: str
    resource_id: int
    operations: int
    datasets: int
    dataset_versions: int
    annotations: int
    dataset_reuses: int

    @property
    def has_references(self) -> bool:
        return any(
            (
                self.operations,
                self.datasets,
                self.dataset_versions,
                self.annotations,
                self.dataset_reuses,
            )
        )

    def references_dict(self) -> Dict[str, int]:
        return {
            "operations": self.operations,
            "datasets": self.datasets,
            "dataset_versions": self.dataset_versions,
            "annotations": self.annotations,
            "dataset_reuses": self.dataset_reuses,
        }

    def conflict_detail(self, label: str) -> dict:
        """构造三类资源一致的 409 响应体。"""
        return {
            "message": f"{label}仍被引用，无法删除",
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "references": self.references_dict(),
        }


def collect_resource_references(
    db: Session, resource_type: str, resource_id: int
) -> ResourceReferenceSummary:
    """在当前事务内统计指定资源被作业、数据集及历史统计引用的情况。"""
    spec = RESOURCE_SPECS[resource_type]

    operations = (
        db.query(func.count(OperationData.id))
        .filter(spec.operation_fk == resource_id)
        .scalar()
        or 0
    )

    dataset_ids = [
        row[0]
        for row in db.query(Dataset.id).filter(spec.dataset_fk == resource_id).all()
    ]

    dataset_versions = 0
    dataset_reuses = 0
    if dataset_ids:
        dataset_versions = (
            db.query(func.count(DatasetVersion.id))
            .filter(DatasetVersion.dataset_id.in_(dataset_ids))
            .scalar()
            or 0
        )
        dataset_reuses = (
            db.query(func.count(DatasetReuse.id))
            .filter(DatasetReuse.dataset_id.in_(dataset_ids))
            .scalar()
            or 0
        )

    annotations = 0
    if operations:
        operation_ids = db.query(OperationData.id).filter(
            spec.operation_fk == resource_id
        )
        annotations = (
            db.query(func.count(Annotation.id))
            .filter(Annotation.operation_data_id.in_(operation_ids))
            .scalar()
            or 0
        )

    return ResourceReferenceSummary(
        resource_type=resource_type,
        resource_id=resource_id,
        operations=operations,
        datasets=len(dataset_ids),
        dataset_versions=dataset_versions,
        annotations=annotations,
        dataset_reuses=dataset_reuses,
    )
