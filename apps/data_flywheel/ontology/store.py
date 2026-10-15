import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from apps.data_flywheel.ontology.schema import (
    AliasEntry,
    ObjectClass,
    ObjectSubtype,
    PlacementPreference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_STORE = PROJECT_ROOT / "data" / "output" / "flywheel" / "ontology_store.json"
DEFAULT_SEED_STORE = PROJECT_ROOT / "data" / "ontology" / "ontology_store.json"


def resolve_store_path(path: Optional[str]) -> str:
    candidates = [
        path,
        os.environ.get("REX_ONTOLOGY_STORE_PATH", "").strip(),
        os.environ.get("ONTOLOGY_STORE_PATH", "").strip(),
        str(DEFAULT_RUNTIME_STORE),
        str(DEFAULT_SEED_STORE),
    ]
    fallback = str(DEFAULT_SEED_STORE)
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        fallback = normalized
        if os.path.exists(normalized):
            return normalized
    return fallback


@dataclass
class QueryEntry:
    query: str
    level: str
    class_id: str
    subtype_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "level": self.level,
            "class_id": self.class_id,
            "subtype_id": self.subtype_id,
        }


@dataclass
class OntologyStore:
    classes: Dict[str, ObjectClass] = field(default_factory=dict)
    subtypes: Dict[str, ObjectSubtype] = field(default_factory=dict)
    aliases: List[AliasEntry] = field(default_factory=list)
    preferences: List[PlacementPreference] = field(default_factory=list)

    def add_class(self, obj_class: ObjectClass) -> None:
        if obj_class.class_id in self.classes:
            raise ValueError(f"class_id already exists: {obj_class.class_id}")
        self.classes[obj_class.class_id] = obj_class

    def add_subtype(self, obj_subtype: ObjectSubtype) -> None:
        if obj_subtype.class_id not in self.classes:
            raise KeyError(f"Unknown class_id: {obj_subtype.class_id}")
        if obj_subtype.subtype_id in self.subtypes:
            raise ValueError(f"subtype_id already exists: {obj_subtype.subtype_id}")
        self.subtypes[obj_subtype.subtype_id] = obj_subtype

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
                and (
                    existing.class_id != alias.class_id
                    or existing.subtype_id != alias.subtype_id
                )
            ):
                raise ValueError(
                    f"alias already approved for another target: {alias.alias} -> "
                    f"{existing.class_id}/{existing.subtype_id or '-'}"
                )
        self.aliases.append(alias)
        return True

    def get_class(self, class_id: str) -> Optional[ObjectClass]:
        return self.classes.get(class_id)

    def resolve_observation_name(self, name: str, context: Optional[object] = None):
        from apps.data_flywheel.ontology.matcher import resolve_observation_name

        return resolve_observation_name(name, self, context=context)

    def get_operation(
        self,
        name: str,
        context: Optional[object] = None,
    ) -> Optional[str]:
        return self.resolve_observation_name(name, context=context).operation

    def get_preferred_operation(
        self,
        name: str,
        context: Optional[object] = None,
    ) -> Optional[str]:
        return self.resolve_observation_name(name, context=context).preferred_operation

    def get_role_ids(
        self,
        name: str,
        context: Optional[object] = None,
    ) -> List[str]:
        return list(self.resolve_observation_name(name, context=context).role_ids)

    def get_allowed_operations(
        self,
        name: str,
        context: Optional[object] = None,
    ) -> List[str]:
        return list(
            self.resolve_observation_name(name, context=context).allowed_operations
        )

    def get_subtype(self, subtype_id: Optional[str]) -> Optional[ObjectSubtype]:
        if not subtype_id:
            return None
        return self.subtypes.get(subtype_id)

    def list_subtypes(self, class_id: str) -> List[ObjectSubtype]:
        return [item for item in self.subtypes.values() if item.class_id == class_id]

    def export_query_list(self, level: str = "all") -> List[QueryEntry]:
        resolved_level = "alias" if level == "all" else level
        if resolved_level not in {"class", "subtype", "alias"}:
            raise ValueError(f"Invalid level: {level}")

        queries: List[QueryEntry] = []
        approved_class_ids = {
            class_id
            for class_id, obj_class in self.classes.items()
            if obj_class.metadata.status == "approved"
        }

        for class_id in sorted(approved_class_ids):
            obj_class = self.classes[class_id]
            queries.append(QueryEntry(query=obj_class.name, level="class", class_id=class_id))

        if resolved_level in {"subtype", "alias"}:
            for subtype in self.subtypes.values():
                if subtype.class_id not in approved_class_ids:
                    continue
                if subtype.metadata.status != "approved":
                    continue
                queries.append(
                    QueryEntry(
                        query=subtype.name,
                        level="subtype",
                        class_id=subtype.class_id,
                        subtype_id=subtype.subtype_id,
                    )
                )

        if resolved_level == "alias":
            for alias in self.aliases:
                if alias.class_id not in approved_class_ids:
                    continue
                if alias.status not in {"approved", "auto_approved"}:
                    continue
                queries.append(
                    QueryEntry(
                        query=alias.alias,
                        level="alias",
                        class_id=alias.class_id,
                        subtype_id=alias.subtype_id,
                    )
                )

        if resolved_level == "class":
            return [q for q in queries if q.level == "class"]
        if resolved_level == "subtype":
            return [q for q in queries if q.level in {"class", "subtype"}]
        return queries

    def get_preferred_container(
        self,
        object_class_id: str,
        visible_container_ids: Set[int],
    ) -> Optional[int]:
        if object_class_id not in self.classes:
            return None
        ranked = sorted(
            [p for p in self.preferences if p.object_class_id == object_class_id],
            key=lambda p: p.rank,
        )
        for pref in ranked:
            if pref.container_id not in visible_container_ids:
                continue
            container = self.get_container_by_id(pref.container_id)
            if container is None or container.container_profile is None:
                continue
            if not container.container_profile.enabled:
                continue
            return pref.container_id
        return None

    def list_containers(self, enabled_only: bool = True) -> List[ObjectClass]:
        containers = [
            obj_class
            for obj_class in self.classes.values()
            if obj_class.container_profile is not None
        ]
        if not enabled_only:
            return containers
        return [obj_class for obj_class in containers if obj_class.container_profile.enabled]

    def get_container_by_id(self, container_id: int) -> Optional[ObjectClass]:
        for obj_class in self.classes.values():
            profile = obj_class.container_profile
            if profile and profile.container_id == container_id:
                return obj_class
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": {k: v.to_dict() for k, v in self.classes.items()},
            "subtypes": {k: v.to_dict() for k, v in self.subtypes.items()},
            "aliases": [alias.to_dict() for alias in self.aliases],
            "preferences": [p.to_dict() for p in self.preferences],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyStore":
        classes: Dict[str, ObjectClass] = {}
        for key, raw in (data.get("classes") or {}).items():
            obj = ObjectClass.from_dict(raw or {})
            if not obj.class_id:
                obj.class_id = key
            classes[obj.class_id] = obj

        subtypes: Dict[str, ObjectSubtype] = {}
        for key, raw in (data.get("subtypes") or {}).items():
            obj = ObjectSubtype.from_dict(raw or {})
            if not obj.subtype_id:
                obj.subtype_id = key
            subtypes[obj.subtype_id] = obj

        return cls(
            classes=classes,
            subtypes=subtypes,
            aliases=[
                AliasEntry.from_dict(item)
                for item in (data.get("aliases") or [])
            ],
            preferences=[
                PlacementPreference.from_dict(item)
                for item in (data.get("preferences") or [])
            ],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "OntologyStore":
        if not path or not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
