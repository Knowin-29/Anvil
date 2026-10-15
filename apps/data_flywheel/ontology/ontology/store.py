"""离线词表存储与序列化。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from apps.data_flywheel.ontology.schema import AliasEntry, ObjectClass, ObjectSubtype


@dataclass
class OntologyStore:
    classes: Dict[str, ObjectClass] = field(default_factory=dict)
    subtypes: Dict[str, ObjectSubtype] = field(default_factory=dict)
    aliases: List[AliasEntry] = field(default_factory=list)

    def add_class(self, object_class: ObjectClass) -> None:
        if object_class.class_id in self.classes:
            raise ValueError(f"class_id already exists: {object_class.class_id}")
        self.classes[object_class.class_id] = object_class

    def add_subtype(self, subtype: ObjectSubtype) -> None:
        if subtype.class_id not in self.classes:
            raise KeyError(f"Unknown class_id: {subtype.class_id}")
        if subtype.subtype_id in self.subtypes:
            raise ValueError(f"subtype_id already exists: {subtype.subtype_id}")
        self.subtypes[subtype.subtype_id] = subtype

    def add_alias(self, alias: AliasEntry) -> bool:
        for existing in self.aliases:
            if (
                existing.normalized_alias == alias.normalized_alias
                and existing.class_id == alias.class_id
                and existing.subtype_id == alias.subtype_id
            ):
                return False
            if (
                existing.normalized_alias == alias.normalized_alias
                and existing.status in {"approved", "auto", "auto_approved"}
                and alias.status in {"approved", "auto", "auto_approved"}
                and (existing.class_id != alias.class_id or existing.subtype_id != alias.subtype_id)
            ):
                raise ValueError(
                    f"alias already approved for another target: {alias.alias} -> "
                    f"{existing.class_id}/{existing.subtype_id or '-'}"
                )
        self.aliases.append(alias)
        return True

    def get_class(self, class_id: str) -> Optional[ObjectClass]:
        return self.classes.get(class_id)

    def get_subtype(self, subtype_id: Optional[str]) -> Optional[ObjectSubtype]:
        if not subtype_id:
            return None
        return self.subtypes.get(subtype_id)

    def list_subtypes(self, class_id: str) -> List[ObjectSubtype]:
        return [item for item in self.subtypes.values() if item.class_id == class_id]

    def to_dict(self) -> Dict[str, object]:
        return {
            "classes": {key: value.to_dict() for key, value in self.classes.items()},
            "subtypes": {key: value.to_dict() for key, value in self.subtypes.items()},
            "aliases": [item.to_dict() for item in self.aliases],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OntologyStore":
        classes = {
            key: ObjectClass.from_dict(value)
            for key, value in (data.get("classes") or {}).items()
        }
        subtypes = {
            key: ObjectSubtype.from_dict(value)
            for key, value in (data.get("subtypes") or {}).items()
        }
        aliases = [AliasEntry.from_dict(item) for item in (data.get("aliases") or [])]
        return cls(classes=classes, subtypes=subtypes, aliases=aliases)

    @classmethod
    def load(cls, path: str) -> "OntologyStore":
        if not path or not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
