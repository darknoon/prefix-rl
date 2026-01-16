"""
Virtual filesystem for the agentic evaluation harness.
Files are stored in memory (dict) to avoid sandboxing complexity.
"""

from dataclasses import dataclass, field
from typing import BinaryIO
from io import BytesIO
from PIL import Image


@dataclass
class VirtualFile:
    """A file in the virtual filesystem."""

    content: bytes
    is_image: bool = False

    @classmethod
    def from_text(cls, text: str) -> "VirtualFile":
        return cls(content=text.encode("utf-8"), is_image=False)

    @classmethod
    def from_image(cls, image: Image.Image, format: str = "PNG") -> "VirtualFile":
        buffer = BytesIO()
        image.save(buffer, format=format)
        return cls(content=buffer.getvalue(), is_image=True)

    @classmethod
    def from_bytes(cls, data: bytes, is_image: bool = False) -> "VirtualFile":
        return cls(content=data, is_image=is_image)

    def as_text(self) -> str:
        return self.content.decode("utf-8")

    def as_image(self) -> Image.Image:
        return Image.open(BytesIO(self.content))

    def as_bytes_io(self) -> BytesIO:
        return BytesIO(self.content)


@dataclass
class VirtualFS:
    """
    In-memory virtual filesystem.
    
    Files are stored as a dict mapping filename -> VirtualFile.
    """

    files: dict[str, VirtualFile] = field(default_factory=dict)
    _image_counter: int = field(default=1)

    def write(self, filename: str, content: str) -> None:
        """Write text content to a file."""
        self.files[filename] = VirtualFile.from_text(content)

    def write_bytes(self, filename: str, data: bytes, is_image: bool = False) -> None:
        """Write binary content to a file."""
        self.files[filename] = VirtualFile.from_bytes(data, is_image=is_image)

    def write_image(self, filename: str, image: Image.Image, format: str = "PNG") -> None:
        """Write a PIL Image to a file."""
        self.files[filename] = VirtualFile.from_image(image, format=format)

    def read(self, filename: str) -> VirtualFile | None:
        """Read a file, returns None if not found."""
        return self.files.get(filename)

    def read_text(self, filename: str) -> str | None:
        """Read a file as text, returns None if not found."""
        vf = self.files.get(filename)
        if vf is None:
            return None
        return vf.as_text()

    def read_image(self, filename: str) -> Image.Image | None:
        """Read a file as a PIL Image, returns None if not found."""
        vf = self.files.get(filename)
        if vf is None:
            return None
        return vf.as_image()

    def exists(self, filename: str) -> bool:
        """Check if a file exists."""
        return filename in self.files

    def list_files(self) -> list[str]:
        """List all files in the filesystem."""
        return list(self.files.keys())

    def delete(self, filename: str) -> bool:
        """Delete a file, returns True if it existed."""
        if filename in self.files:
            del self.files[filename]
            return True
        return False

    def next_image_filename(self, prefix: str = "", suffix: str = ".png") -> str:
        """Generate a unique image filename with incrementing counter."""
        filename = f"{prefix}{self._image_counter:05d}{suffix}"
        self._image_counter += 1
        return filename

    def save_to_disk(self, output_dir: "Path") -> dict[str, "Path"]:
        """Save all files to disk. Returns mapping of vfs filename -> disk path."""
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for filename, vf in self.files.items():
            path = output_dir / filename
            path.write_bytes(vf.content)
            paths[filename] = path
        
        return paths
