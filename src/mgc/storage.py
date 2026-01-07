from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class StoragePaths:
    data_dir: Path

    @property
    def tracks_dir(self) -> Path:
        return self.data_dir / "tracks"

    @property
    def previews_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def posts_dir(self) -> Path:
        return self.data_dir / "posts"

    def ensure(self) -> None:
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.posts_dir.mkdir(parents=True, exist_ok=True)
