"""数据源注册表 — @register_source 装饰器 + 工厂函数。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

from apps.data_flywheel.sources.base import DataSource

_REGISTRY: Dict[str, Type[DataSource]] = {}


def register_source(name: str) -> Callable:
    """装饰器：注册数据源类型。"""
    def decorator(cls: Type[DataSource]) -> Type[DataSource]:
        _REGISTRY[name] = cls
        cls.source_type = name
        return cls
    return decorator


def get_source(name: str, **kwargs: Any) -> DataSource:
    """根据名称创建数据源实例。"""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown source type '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_sources() -> List[str]:
    """列出所有已注册的数据源类型。"""
    return sorted(_REGISTRY.keys())
