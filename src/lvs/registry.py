from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .contracts import ModuleManifest


class ModuleRegistry:
    """Tiny future-proof registry.

    Today only the lightning module is registered. Future modules must register
    a manifest and should consume/produce stable contract objects.
    """

    def __init__(self) -> None:
        self._modules: Dict[str, ModuleManifest] = {}

    def register(self, manifest: ModuleManifest) -> None:
        if manifest.module_id in self._modules:
            raise ValueError(f"Duplicate module_id: {manifest.module_id}")
        self._modules[manifest.module_id] = manifest

    def list(self) -> List[ModuleManifest]:
        return list(self._modules.values())

    def as_dicts(self) -> List[dict]:
        return [asdict(m) for m in self.list()]


def default_registry() -> ModuleRegistry:
    reg = ModuleRegistry()
    reg.register(ModuleManifest(
        module_id="lightning_reel",
        name="Lightning Viral Reel Builder",
        version="1.0.0",
        description="Detects lightning events, trims dark footage, crops strikes, and exports viral-ready reels.",
        enabled=True,
    ))
    return reg
