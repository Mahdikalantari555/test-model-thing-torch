
import json
import hashlib
import zipfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# ponytail: portable .droid brain archive format - ZIP with SHA-256 integrity verification

class DroidPackage:
    """Portable .droid archive format bundling Droid state with integrity verification."""

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def export_droid(profile_dir: str, out_path: str, 
                     extra_files: Optional[List[str]] = None) -> Dict:
        """
        Export Droid profile to .droid ZIP archive.
        Contains: manifest.json, config.json, memory.safetensors, knowledge.db, train_log.json
        """
        profile_path = Path(profile_dir)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Expected members
        members = ["config.json", "memory.safetensors", "knowledge.db", "train_log.json"]
        if extra_files:
            members.extend(extra_files)

        # Gather existing files
        files_to_include = []
        for m in members:
            p = profile_path / m
            if p.exists():
                files_to_include.append((m, p))

        if not files_to_include:
            raise FileNotFoundError(f"No Droid files found in {profile_dir}")

        # Build manifest with checksums
        manifest = {
            "version": "1.0",
            "created_at": time.time(),
            "base_anchor_version": "all-MiniLM-L6-v2",
            "dim": 384,
            "files": {},
            "checksums": {}
        }

        # Read config to get dim if available
        config_path = profile_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                    manifest["dim"] = cfg.get("dim", 384)
                    manifest["droid_name"] = cfg.get("name", "unknown")
            except:
                pass

        # Create ZIP
        with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # First, add files and compute checksums
            for arcname, filepath in files_to_include:
                # Compute checksum
                checksum = DroidPackage._sha256_file(filepath)
                manifest["checksums"][arcname] = checksum
                manifest["files"][arcname] = {
                    "size": filepath.stat().st_size,
                    "sha256": checksum
                }
                # Add to zip
                zf.write(filepath, arcname=arcname)

            # Write manifest.json last (with checksums of other files)
            manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')
            # Note: manifest's own checksum not included in itself to avoid circular dep
            # But we can add manifest checksum after?
            zf.writestr("manifest.json", manifest_bytes)

        return {
            "out_path": str(out_path),
            "files": list(manifest["files"].keys()),
            "manifest": manifest
        }

    @staticmethod
    def import_droid(archive_path: str, target_dir: str, verify: bool = True) -> Dict:
        """
        Import .droid archive to target directory with SHA-256 verification and zip-slip protection.
        """
        archive_path = Path(archive_path)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        with zipfile.ZipFile(archive_path, 'r') as zf:
            # Security: check for zip-slip
            for member in zf.namelist():
                # Reject absolute paths and .. traversal
                if member.startswith('/') or member.startswith('\\'):
                    raise ValueError(f"Security violation: absolute path in archive: {member}")
                # Normalize and check for .. 
                # Use Path to resolve
                p = Path(member)
                # Check if any part is ..
                if '..' in p.parts:
                    raise ValueError(f"Security violation: path traversal in archive: {member}")
                # Also check normalized path escapes target
                # The zipfile spec says members should be relative, but we double-check
                resolved = (target_path / member).resolve()
                try:
                    resolved.relative_to(target_path.resolve())
                except ValueError:
                    raise ValueError(f"Security violation: zip-slip detected: {member} would extract to {resolved}")

            # Read manifest
            if "manifest.json" not in zf.namelist():
                raise ValueError("Invalid .droid archive: missing manifest.json")
            
            manifest_data = zf.read("manifest.json")
            manifest = json.loads(manifest_data.decode('utf-8'))

            # Verify checksums if requested
            if verify:
                checksums = manifest.get("checksums", {})
                for arcname, expected_sha256 in checksums.items():
                    if arcname not in zf.namelist():
                        raise ValueError(f"Missing file in archive: {arcname} (expected per manifest)")
                    data = zf.read(arcname)
                    actual = DroidPackage._sha256_bytes(data)
                    if actual != expected_sha256:
                        raise ValueError(f"Checksum mismatch for {arcname}: expected {expected_sha256}, got {actual}")

            # Extract all
            for member in zf.namelist():
                if member == "manifest.json":
                    # Write manifest too
                    data = zf.read(member)
                    (target_path / member).write_bytes(data)
                else:
                    # Extract file
                    data = zf.read(member)
                    out_file = target_path / member
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_bytes(data)

        return {
            "target_dir": str(target_path),
            "manifest": manifest,
            "verified": verify
        }

    @staticmethod
    def list_contents(archive_path: str) -> Dict:
        """List contents of .droid archive without extracting."""
        archive_path = Path(archive_path)
        with zipfile.ZipFile(archive_path, 'r') as zf:
            namelist = zf.namelist()
            manifest = {}
            if "manifest.json" in namelist:
                manifest_data = zf.read("manifest.json")
                manifest = json.loads(manifest_data.decode('utf-8'))
            return {
                "files": namelist,
                "manifest": manifest
            }
