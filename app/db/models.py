"""SQLAlchemy models for dataset acquisition and lineage."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DatasetManifest(Base):
    __tablename__ = "dataset_manifests"

    manifest_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(255), nullable=False)
    root: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="verified")
    file_count: Mapped[int] = mapped_column(Integer, nullable=False)
    subject_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    manifest_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    files: Mapped[list[DatasetManifestFile]] = relationship(
        back_populates="manifest", cascade="all, delete-orphan"
    )


class DatasetManifestFile(Base):
    __tablename__ = "dataset_manifest_files"
    __table_args__ = (UniqueConstraint("manifest_id", "relative_path"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    manifest_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_manifests.manifest_id", ondelete="CASCADE"), nullable=False
    )
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    subject_id: Mapped[str | None] = mapped_column(String(255))
    modality: Mapped[str | None] = mapped_column(String(32))
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    manifest: Mapped[DatasetManifest] = relationship(back_populates="files")
