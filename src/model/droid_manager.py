import os
import shutil
from pathlib import Path
from typing import List, Optional
from src.model.droid import DroidEngine

# ponytail: Simple folder-backed Droid profile manager.

class DroidManager:
    """Manages independent named Droid profiles under droids/<name>/."""

    def __init__(self, base_dir: str = "droids"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cached_instances = {}

    def list_droids(self) -> List[str]:
        """List all available Droid names."""
        if not self.base_dir.exists():
            return []
        dirs = [d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        return sorted(dirs)

    def create_droid(self, name: str) -> DroidEngine:
        """Create a new Droid profile."""
        clean_name = name.strip().lower().replace(" ", "-")
        droid_dir = self.base_dir / clean_name
        droid_dir.mkdir(parents=True, exist_ok=True)

        engine = DroidEngine(name=clean_name)
        engine.save_profile(str(droid_dir))
        self.cached_instances[clean_name] = engine
        return engine

    def get_droid(self, name: str) -> DroidEngine:
        """Load an existing Droid profile or create one if it doesn't exist."""
        clean_name = name.strip().lower().replace(" ", "-")
        if clean_name in self.cached_instances:
            return self.cached_instances[clean_name]

        droid_dir = self.base_dir / clean_name
        engine = DroidEngine(name=clean_name)
        if (droid_dir / "config.json").exists():
            engine.load_profile(str(droid_dir))
        else:
            droid_dir.mkdir(parents=True, exist_ok=True)
            engine.save_profile(str(droid_dir))

        self.cached_instances[clean_name] = engine
        return engine

    def save_droid(self, name: str):
        """Save a specific Droid profile to disk."""
        clean_name = name.strip().lower().replace(" ", "-")
        if clean_name in self.cached_instances:
            droid_dir = self.base_dir / clean_name
            self.cached_instances[clean_name].save_profile(str(droid_dir))

    def delete_droid(self, name: str) -> bool:
        """Delete a Droid profile directory."""
        clean_name = name.strip().lower().replace(" ", "-")
        droid_dir = self.base_dir / clean_name
        if clean_name in self.cached_instances:
            del self.cached_instances[clean_name]
        if droid_dir.exists():
            shutil.rmtree(droid_dir)
            return True
        return False
