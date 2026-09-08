from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class PermitClass(str, Enum):
    PRIMARY_PROJECT = "primary_project"
    PRIMARY_UNIT = "primary_unit"
    SECONDARY = "secondary"


class AdChannel(str, Enum):
    ELECTRONIC = "electronic"
    PRINT = "print"
    OUTDOOR = "outdoor"
    SOCIAL = "social"
    PORTAL = "portal"


class ListingMetadata(BaseModel):
    listing_id: str
    permit_number: str
    permit_class: PermitClass = PermitClass.SECONDARY
    channel: AdChannel = AdChannel.PORTAL
    broker_name: str
    developer_name: str | None = None
    property_reference: str | None = None
    community: str | None = None
    permit_issued: date | None = None
    permit_expires: date | None = None
    asking_price_aed: int | None = None
    contact_phone: str | None = None

    @field_validator("permit_number")
    @classmethod
    def strip_permit(cls, value: str) -> str:
        return value.strip()


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "watermark_logo"


class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    detail: str
    evidence: dict = Field(default_factory=dict)


class ImageFinding(BaseModel):
    filename: str
    watermarks: list[BoundingBox]
    ocr_text: str
    permit_candidates: list[str]
    phash: str


class ComplianceReport(BaseModel):
    listing_id: str
    compliant: bool
    overall_score: float
    checks: list[CheckResult]
    images: list[ImageFinding]
    model: dict
