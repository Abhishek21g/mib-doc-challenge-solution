"""Typed seams between the batch runner and future processing stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .models import PredictionRow


class RendererStage(Protocol):
    def render(self, pdf_path: Path) -> Any:
        """Render one source PDF into visible page data."""


class EvidenceExtractor(Protocol):
    def extract(self, rendered_case: Any) -> Any:
        """Extract visible candidate evidence from a rendered case."""


class EvidenceResolver(Protocol):
    def resolve(self, candidate_evidence: Any) -> Any:
        """Link and resolve candidate evidence for the active case."""


class Adjudicator(Protocol):
    def adjudicate(self, resolved_case: Any) -> PredictionRow | Mapping[str, Any] | None:
        """Return one prediction, or None for a technical omission."""


class CaseProcessor(Protocol):
    def process_case(self, pdf_path: Path) -> PredictionRow | Mapping[str, Any] | None:
        """Process one case independently."""


@dataclass
class ProcessingPipeline:
    """Composition seam for the four downstream processing stages."""

    renderer: RendererStage
    extractor: EvidenceExtractor
    resolver: EvidenceResolver
    adjudicator: Adjudicator

    def process_case(self, pdf_path: Path) -> PredictionRow | Mapping[str, Any] | None:
        rendered_case = self.renderer.render(pdf_path)
        candidate_evidence = self.extractor.extract(rendered_case)
        resolved_case = self.resolver.resolve(candidate_evidence)
        return self.adjudicator.adjudicate(resolved_case)


class SafeFallbackProcessor:
    """Schema-valid placeholder until the processing stages are implemented.

    The filename supplies the only derived value. All substantive fields use
    conservative values and the case is routed to NEEDS_REVIEW rather than
    being omitted solely because downstream logic is not yet available.
    """

    def process_case(self, pdf_path: Path) -> Mapping[str, Any]:
        return {
            "case_id": pdf_path.stem,
            "applicant_name": "unknown",
            "species_code": "unknown",
            "home_world": "unknown",
            "visa_class": "unknown",
            "sponsor_id": "SPN-0000",
            "arrival_date": "1900-01-01",
            "declared_purpose": "unknown",
            "risk_flags": "none",
            "fee_status": "unknown",
            "adjudication": "NEEDS_REVIEW",
            "confidence": 0.0,
        }


@dataclass
class RenderFirstFallbackProcessor:
    """Exercise ingestion before conservative downstream stages exist."""

    renderer: RendererStage
    fallback: SafeFallbackProcessor

    def process_case(self, pdf_path: Path) -> Mapping[str, Any]:
        self.renderer.render(pdf_path)
        return self.fallback.process_case(pdf_path)


@dataclass
class ExtractThenFallbackProcessor:
    """Exercise visible extraction before resolution/adjudication exist."""

    renderer: RendererStage
    extractor: EvidenceExtractor
    fallback: SafeFallbackProcessor

    def process_case(self, pdf_path: Path) -> Mapping[str, Any]:
        rendered = self.renderer.render(pdf_path)
        self.extractor.extract(rendered)
        return self.fallback.process_case(pdf_path)


@dataclass
class ResolveThenFallbackProcessor:
    """Exercise linking/resolution before policy adjudication exists."""

    renderer: RendererStage
    extractor: EvidenceExtractor
    linker: Any
    resolver: EvidenceResolver
    fallback: SafeFallbackProcessor

    def process_case(self, pdf_path: Path) -> Mapping[str, Any]:
        rendered = self.renderer.render(pdf_path)
        candidates = self.extractor.extract(rendered)
        linked_case = self.linker.link(rendered.case_id, candidates)
        self.resolver.resolve(linked_case)
        return self.fallback.process_case(pdf_path)


@dataclass
class AdjudicatingCaseProcessor:
    """Run the complete visible-evidence path through deterministic policy."""

    renderer: RendererStage
    extractor: EvidenceExtractor
    linker: Any
    resolver: EvidenceResolver
    adjudicator: Adjudicator

    def process_case(self, pdf_path: Path) -> PredictionRow | Mapping[str, Any] | None:
        rendered = self.renderer.render(pdf_path)
        candidates = self.extractor.extract(rendered)
        linked_case = self.linker.link(rendered.case_id, candidates)
        resolved_case = self.resolver.resolve(linked_case)
        return self.adjudicator.adjudicate(resolved_case)
