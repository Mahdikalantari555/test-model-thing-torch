
import os
import shutil
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.model.droid import DroidEngine

# Try imports for new modules
try:
    from src.model.droid_package import DroidPackage
except:
    try:
        from .droid_package import DroidPackage
    except:
        DroidPackage = None

try:
    from src.model.gguf_backend import GgufBackend
except:
    try:
        from .gguf_backend import GgufBackend
    except:
        GgufBackend = None

# ponytail: Simple folder-backed Droid profile manager - extended with merge, versioning, GGUF

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

    def create_droid(self, name: str, plastic_adapter: str = "associative_rtu", adapter_kwargs: Optional[Dict[str, Any]] = None, gguf_model: Optional[str] = None, gguf_config: Optional[Dict[str, Any]] = None, **kwargs) -> DroidEngine:
        """Create a new Droid profile with optional adapter and GGUF config."""
        clean_name = name.strip().lower().replace(" ", "-")
        droid_dir = self.base_dir / clean_name
        droid_dir.mkdir(parents=True, exist_ok=True)

        adapter_kwargs = adapter_kwargs or {}
        gguf_config = gguf_config or {}

        # Handle gguf_model as string path or dict
        if gguf_model:
            if isinstance(gguf_model, str):
                gguf_config["gguf_model_path"] = gguf_model
            elif isinstance(gguf_model, dict):
                gguf_config.update(gguf_model)

        # Handle legacy kwargs: plastic_adapter might be in kwargs
        if "plastic_adapter" in kwargs:
            plastic_adapter = kwargs.pop("plastic_adapter")
        if "adapter_kwargs" in kwargs:
            adapter_kwargs = kwargs.pop("adapter_kwargs")
        if "gguf_model_path" in kwargs:
            gguf_config["gguf_model_path"] = kwargs.pop("gguf_model_path")
        if "gguf_repo_id" in kwargs:
            gguf_config["gguf_repo_id"] = kwargs.pop("gguf_repo_id")
        if "gguf_filename" in kwargs:
            gguf_config["gguf_filename"] = kwargs.pop("gguf_filename")

        engine = DroidEngine(name=clean_name, plastic_adapter=plastic_adapter, adapter_kwargs=adapter_kwargs, gguf_config=gguf_config, **kwargs)
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

    def merge_droids(self, name_a: str, name_b: str, new_name: str, policy: str = "require_manual") -> DroidEngine:
        """Merge two Droid profiles into a new profile."""
        clean_a = name_a.strip().lower().replace(" ", "-")
        clean_b = name_b.strip().lower().replace(" ", "-")
        clean_new = new_name.strip().lower().replace(" ", "-")

        droid_a = self.get_droid(clean_a)
        droid_b = self.get_droid(clean_b)

        # Create new droid
        new_droid = self.create_droid(clean_new)

        # Merge memory
        result = new_droid.merge_memory(droid_a, policy=policy)
        # Then merge second? Actually merge_memory merges other into self, but we need both A and B
        # For simplicity, first create from A, then merge B
        # We already created empty new_droid, so let's make it copy of A then merge B
        # Copy A's knowledge and memory
        # For simplicity, we will make new_droid start as copy of A, then merge B
        # Delete and recreate as copy of A
        self.delete_droid(clean_new)
        # Copy directory A to new
        src_a = self.base_dir / clean_a
        dst_new = self.base_dir / clean_new
        if src_a.exists():
            shutil.copytree(str(src_a), str(dst_new))
            # Load it
            new_droid = DroidEngine(name=clean_new)
            new_droid.load_profile(str(dst_new))
            self.cached_instances[clean_new] = new_droid
        else:
            new_droid = self.create_droid(clean_new)

        # Now merge B into new_droid
        merge_result = new_droid.merge_memory(droid_b, policy=policy)
        new_droid.save_profile(str(dst_new))

        return new_droid

    def export_version(self, name: str, version_id: str, out_path: str) -> Dict[str, Any]:
        """Export snapshot artifact."""
        clean_name = name.strip().lower().replace(" ", "-")
        droid = self.get_droid(clean_name)
        
        # Ensure snapshot exists
        if version_id not in droid.snapshots:
            # Create snapshot if not exists
            droid.create_snapshot(version_id)
        
        # For file-based export, we need to export the profile dir plus snapshot metadata
        droid_dir = self.base_dir / clean_name
        if DroidPackage is not None:
            # Use DroidPackage to export full droid
            result = DroidPackage.export_droid(str(droid_dir), out_path)
            return result
        else:
            # Fallback: copy knowledge.db and memory.safetensors and config
            import zipfile, hashlib, json as json_lib
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for fname in ["config.json", "memory.safetensors", "knowledge.db", "train_log.json", "snapshots.json"]:
                    fpath = droid_dir / fname
                    if fpath.exists():
                        zf.write(str(fpath), arcname=fname)
                # Add version metadata
                snap = droid.snapshots.get(version_id, {})
                meta = snap.get("metadata", {}) if snap else {}
                zf.writestr("version.json", json_lib.dumps({"version_id": version_id, "metadata": meta}, indent=2))
            return {"out_path": str(out_path), "version_id": version_id}

    def import_version(self, name: str, snapshot_path: str) -> DroidEngine:
        """Restore from exported snapshot."""
        clean_name = name.strip().lower().replace(" ", "-")
        target_dir = self.base_dir / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)

        if DroidPackage is not None:
            result = DroidPackage.import_droid(snapshot_path, str(target_dir))
        else:
            # Fallback: unzip
            import zipfile
            with zipfile.ZipFile(snapshot_path, 'r') as zf:
                # Security check
                for member in zf.namelist():
                    if member.startswith('/') or '..' in member:
                        raise ValueError(f"Security violation: {member}")
                zf.extractall(str(target_dir))

        # Load droid
        engine = DroidEngine(name=clean_name)
        engine.load_profile(str(target_dir))
        self.cached_instances[clean_name] = engine
        return engine

    def download_model(self, name: str, repo_id: str, filename: str, local_dir: Optional[str] = None) -> str:
        """Download GGUF model to profile dir."""
        clean_name = name.strip().lower().replace(" ", "-")
        droid_dir = self.base_dir / clean_name
        droid_dir.mkdir(parents=True, exist_ok=True)

        if local_dir is None:
            local_dir = str(droid_dir / "models")

        Path(local_dir).mkdir(parents=True, exist_ok=True)

        if GgufBackend is not None:
            backend = GgufBackend()
            try:
                path = backend.download_model(repo_id, filename, local_dir)
                # Update droid config
                droid = self.get_droid(clean_name)
                droid.gguf_config["gguf_model_path"] = path
                droid.gguf_config["gguf_repo_id"] = repo_id
                droid.gguf_config["gguf_filename"] = filename
                droid.save_profile(str(droid_dir))
                return path
            except Exception as e:
                print(f"Download via GgufBackend failed: {e}, trying huggingface_hub directly")
        
        # Fallback to huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
            # Update config
            droid = self.get_droid(clean_name)
            droid.gguf_config["gguf_model_path"] = path
            droid.gguf_config["gguf_repo_id"] = repo_id
            droid.gguf_config["gguf_filename"] = filename
            droid.save_profile(str(droid_dir))
            return path
        except Exception as e:
            raise RuntimeError(f"Failed to download model {repo_id}/{filename}: {e}")
