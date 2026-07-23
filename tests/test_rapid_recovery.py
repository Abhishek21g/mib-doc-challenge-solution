import types
import unittest
from pathlib import Path

from mib_pipeline.adjudication import AdjudicationOutcome, DecisionTrace
from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.ingestion import Rect
from mib_pipeline.models import PredictionRow
from mib_pipeline.rapid_recovery import (
    RapidOcrEngine,
    RapidOutputRecoveryProcessor,
    REVIEW_APPROVAL_CONFIDENCE,
    SEMANTIC_DENIAL_CONFIDENCE,
    XW1_MULTISOURCE_REVIEW_APPROVAL_CONFIDENCE,
)
from mib_pipeline.resolution import (
    FieldState,
    ResolvedCase,
    ResolvedField,
)


CASE_ID = "MIB-000001"
APPLICANT = "Zed Zarnax"
BASE_VALUES = {
    "applicant_name": APPLICANT,
    "species_code": "ORION_GRAYS",
    "home_world": "Kepler-186f",
    "visa_class": "XW-2",
    "sponsor_id": "SPN-1042",
    "arrival_date": "2026-04-17",
    "declared_purpose": "research",
    "risk_flags": "none",
    "fee_status": "paid",
}


def evidence(
    field_name,
    value,
    *,
    evidence_type=EvidenceType.INTAKE_FORM,
    confidence=0.95,
    case_id=CASE_ID,
    applicant=APPLICANT,
    cues=(),
    superseded=False,
    legible=True,
    source="visible_ocr",
    page=0,
):
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page,
        box=Rect(1, 2, 3, 4),
        legible=legible,
        superseded=superseded,
        ocr_confidence=confidence,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint=case_id,
        applicant_hint=applicant,
    )


def field(name, value, *, state=FieldState.RESOLVED, considered=()):
    winner = next(
        (candidate for candidate in considered if candidate.value == value),
        None,
    )
    return ResolvedField(
        field_name=name,
        state=state,
        value=value,
        winning_evidence=winner,
        considered=tuple(considered),
        reason="test field",
    )


def resolved_case(
    *,
    values=None,
    unknown=(),
    considered=None,
    active=APPLICANT,
    unresolved_linkage=False,
    unresolved_reasons=(),
):
    values = {**BASE_VALUES, **(values or {})}
    considered = considered or {}
    fields = {
        name: field(
            name,
            None if name in unknown else value,
            state=FieldState.UNKNOWN if name in unknown else FieldState.RESOLVED,
            considered=considered.get(name, ()),
        )
        for name, value in values.items()
    }
    return ResolvedCase(
        case_id=CASE_ID,
        active_applicant=active,
        fields=fields,
        unresolved_linkage=unresolved_linkage,
        unresolved_reasons=tuple(unresolved_reasons),
    )


def row(**overrides):
    values = {
        "case_id": CASE_ID,
        **BASE_VALUES,
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.37,
        **overrides,
    }
    return PredictionRow.from_mapping(values)


def outcome(
    prediction=None,
    *,
    review_reasons=("test_review",),
    approval_facts=(),
    denial_reasons=(),
    authoritative_source=False,
    trace_decision=None,
):
    prediction = prediction or row()
    return AdjudicationOutcome(
        row=prediction,
        trace=DecisionTrace(
            decision=trace_decision or prediction.adjudication,
            authoritative_source=authoritative_source,
            denial_reasons=tuple(denial_reasons),
            review_reasons=tuple(review_reasons),
            approval_facts=tuple(approval_facts),
            exception_ids=(),
        ),
    )


def xw1_multisource_candidates(
    *,
    home_world=BASE_VALUES["home_world"],
    arrival_date=BASE_VALUES["arrival_date"],
):
    return (
        evidence(
            "applicant_name",
            APPLICANT,
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            page=10,
        ),
        evidence(
            "visa_class",
            "XW-1",
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            page=10,
        ),
        evidence(
            "applicant_name",
            APPLICANT,
            evidence_type=EvidenceType.REGISTRY_EXTRACT,
            page=11,
        ),
        evidence(
            "home_world",
            home_world,
            evidence_type=EvidenceType.REGISTRY_EXTRACT,
            page=11,
        ),
        evidence(
            "arrival_date",
            arrival_date,
            evidence_type=EvidenceType.REGISTRY_EXTRACT,
            page=11,
        ),
    )


class FakeRenderer:
    def __init__(self):
        self.calls = 0

    def render(self, path):
        self.calls += 1
        return types.SimpleNamespace(case_id=path.stem)


class FakeExtractor:
    def __init__(self, candidates=(), *, error=None):
        self.candidates = tuple(candidates)
        self.error = error
        self.calls = 0

    def extract(self, rendered):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.candidates


class FakeRapidFactory:
    def __init__(self, candidates=(), *, error=None):
        self.candidates = tuple(candidates)
        self.error = error
        self.calls = 0
        self.instances = []

    def __call__(self):
        self.calls += 1
        instance = FakeExtractor(self.candidates, error=self.error)
        self.instances.append(instance)
        return instance


class FakeLinker:
    def __init__(
        self,
        *,
        primary_active=APPLICANT,
        rapid_active=APPLICANT,
        primary_candidates=("primary",),
    ):
        self.primary = types.SimpleNamespace(
            kind="primary",
            active_applicant=primary_active,
        )
        self.rapid = types.SimpleNamespace(kind="rapid", active_applicant=rapid_active)
        self.primary_candidates = tuple(primary_candidates)
        self.calls = 0

    def link(self, case_id, candidates):
        self.calls += 1
        return (
            self.primary
            if tuple(candidates) == self.primary_candidates
            else self.rapid
        )


class FakeResolver:
    def __init__(self, primary, rapid):
        self.primary = primary
        self.rapid = rapid
        self.calls = 0

    def resolve(self, linked):
        self.calls += 1
        return self.primary if linked.kind == "primary" else self.rapid


class FakeAdjudicator:
    def __init__(self, primary_outcome):
        self.primary_outcome = primary_outcome
        self.calls = 0

    def adjudicate_case(self, resolved):
        self.calls += 1
        return self.primary_outcome


def processor(
    primary_resolved,
    rapid_resolved,
    *,
    primary_outcome=None,
    rapid_candidates=(),
    primary_active=APPLICANT,
    rapid_active=APPLICANT,
    rapid_error=None,
    primary_candidates=("primary",),
):
    renderer = FakeRenderer()
    primary_extractor = FakeExtractor(primary_candidates)
    linker = FakeLinker(
        primary_active=primary_active,
        rapid_active=rapid_active,
        primary_candidates=primary_candidates,
    )
    resolver = FakeResolver(primary_resolved, rapid_resolved)
    adjudicator = FakeAdjudicator(primary_outcome or outcome())
    factory = FakeRapidFactory(rapid_candidates, error=rapid_error)
    recovery = RapidOutputRecoveryProcessor(
        renderer=renderer,
        primary_extractor=primary_extractor,
        linker=linker,
        resolver=resolver,
        adjudicator=adjudicator,
        rapid_extractor_factory=factory,
    )
    return recovery, renderer, linker, resolver, adjudicator, factory


class RapidOcrEngineTests(unittest.TestCase):
    def test_uses_string_wheel_model_root_and_one_plus_one_threads(self):
        captured = {}

        class Engine:
            def __call__(self, image):
                return types.SimpleNamespace(
                    boxes=[[(1, 2), (5, 2), (5, 7), (1, 7)]],
                    txts=["  visible text  "],
                    scores=[0.97],
                )

        def factory(**kwargs):
            captured.update(kwargs)
            return Engine()

        adapter = RapidOcrEngine(
            engine_factory=factory,
            package_root=Path("/opt/rapidocr"),
        )
        params = captured["params"]

        self.assertIsInstance(params["Global.model_root_dir"], str)
        self.assertEqual(
            params["Global.model_root_dir"],
            "/opt/rapidocr/models",
        )
        self.assertEqual(
            params["EngineConfig.onnxruntime.intra_op_num_threads"], 1
        )
        self.assertEqual(
            params["EngineConfig.onnxruntime.inter_op_num_threads"], 1
        )

        tokens = adapter.read_page(
            types.SimpleNamespace(index=2, image_png=b"png")
        )
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].text, "visible text")
        self.assertEqual(tokens[0].box, Rect(1, 2, 5, 7))


class RapidPackagingContractTests(unittest.TestCase):
    def test_lock_and_docker_use_the_offline_headless_closure(self):
        root = Path(__file__).resolve().parents[1]
        lock = (root / "requirements.lock").read_text(encoding="utf-8")
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

        for requirement in (
            "rapidocr==3.9.2",
            "onnxruntime==1.27.0",
            "opencv-python-headless==5.0.0.93",
            "omegaconf==2.0.0",
        ):
            self.assertIn(requirement, lock)
        self.assertNotIn("\nopencv-python==", lock)
        self.assertIn("--no-deps", dockerfile)
        self.assertIn("OC_DISABLE_DOT_ACCESS_WARNING=1", dockerfile)
        self.assertIn(
            "COPY third_party_licenses /app/third_party_licenses",
            dockerfile,
        )

    def test_all_embedded_model_hashes_are_attributed(self):
        provenance = (
            Path(__file__).resolve().parents[1]
            / "third_party_licenses"
            / "MODEL_PROVENANCE.md"
        ).read_text(encoding="utf-8")

        for digest in (
            "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
            "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
        ):
            self.assertIn(digest, provenance)


class RapidOutputRecoveryTests(unittest.TestCase):
    def test_repairs_only_applicant_from_stronger_exact_case_biometric_value(self):
        intake_name = APPLICANT
        biometric_name = "Zed Zornax"
        candidates = (
            evidence(
                "applicant_name",
                intake_name,
                evidence_type=EvidenceType.INTAKE_FORM,
                confidence=0.84,
            ),
            evidence(
                "applicant_name",
                biometric_name,
                evidence_type=EvidenceType.BIOMETRIC_SLIP,
                confidence=0.91,
            ),
        )
        primary = resolved_case(unknown={"applicant_name"})
        rapid = resolved_case(values={"applicant_name": "Rapid Wrong"})
        primary_row = row(
            applicant_name=intake_name,
            adjudication="DENIED",
            confidence=0.61,
        )
        recovery, _renderer, _linker, resolver, adjudicator, factory = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
            primary_candidates=candidates,
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        expected = primary_row.to_dict()
        expected["applicant_name"] = biometric_name
        self.assertEqual(result.to_dict(), expected)
        self.assertEqual(result.adjudication, "DENIED")
        self.assertEqual(result.confidence, 0.61)
        self.assertEqual(resolver.calls, 1)
        self.assertEqual(adjudicator.calls, 1)
        self.assertEqual(factory.calls, 0)

    def test_biometric_applicant_repair_abstains_on_every_scope_ambiguity(self):
        intake = evidence(
            "applicant_name",
            APPLICANT,
            evidence_type=EvidenceType.INTAKE_FORM,
            confidence=0.84,
        )
        biometric = evidence(
            "applicant_name",
            "Zed Zornax",
            evidence_type=EvidenceType.BIOMETRIC_SLIP,
            confidence=0.91,
        )
        variants = {
            "biometric_below_minimum": (
                intake,
                evidence(
                    "applicant_name",
                    "Zed Zornax",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.799,
                ),
            ),
            "biometric_weaker_than_intake": (
                evidence(
                    "applicant_name",
                    APPLICANT,
                    evidence_type=EvidenceType.INTAKE_FORM,
                    confidence=0.92,
                ),
                biometric,
            ),
            "same_value": (
                intake,
                evidence(
                    "applicant_name",
                    APPLICANT,
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.91,
                ),
            ),
            "bad_visual_cue": (
                intake,
                biometric,
                evidence(
                    "applicant_name",
                    "Third Name",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.95,
                    cues=("strikethrough",),
                    superseded=True,
                ),
            ),
            "foreign_case_candidate": (
                intake,
                biometric,
                evidence(
                    "applicant_name",
                    "Foreign Name",
                    evidence_type=EvidenceType.INTAKE_FORM,
                    case_id="MIB-999999",
                    superseded=True,
                ),
            ),
            "multiple_biometric_values": (
                intake,
                biometric,
                evidence(
                    "applicant_name",
                    "Third Name",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.93,
                ),
            ),
            "multiple_intake_values": (
                intake,
                biometric,
                evidence(
                    "applicant_name",
                    "Third Name",
                    evidence_type=EvidenceType.INTAKE_FORM,
                    confidence=0.82,
                ),
            ),
            "superseded_biometric": (
                intake,
                evidence(
                    "applicant_name",
                    "Zed Zornax",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.91,
                    superseded=True,
                ),
            ),
            "non_visible_source": (
                intake,
                evidence(
                    "applicant_name",
                    "Zed Zornax",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.91,
                    source="embedded_text",
                ),
            ),
            "missing_exact_case_scope": (
                intake,
                evidence(
                    "applicant_name",
                    "Zed Zornax",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.91,
                    case_id=None,
                ),
            ),
            "illegible_biometric": (
                intake,
                evidence(
                    "applicant_name",
                    "Zed Zornax",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    confidence=0.91,
                    legible=False,
                ),
            ),
        }
        primary = resolved_case()
        rapid = resolved_case()
        primary_row = row()

        for label, candidates in variants.items():
            with self.subTest(label=label):
                recovery, *_rest = processor(
                    primary,
                    rapid,
                    primary_outcome=outcome(primary_row),
                    primary_candidates=candidates,
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result, primary_row)

    def test_repairs_only_three_frozen_source_priority_output_fields(self):
        intake_visa = evidence(
            "visa_class",
            "TRANSIT-7",
            evidence_type=EvidenceType.INTAKE_FORM,
            confidence=0.82,
            applicant=None,
        )
        sponsor_visa = evidence(
            "visa_class",
            "XW-1",
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            confidence=0.94,
            page=1,
            cues=("structured_sponsor_narrative",),
        )
        sponsor_name = evidence(
            "applicant_name",
            APPLICANT,
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            confidence=0.95,
            page=1,
            cues=("structured_sponsor_narrative",),
        )
        intake_sponsor = evidence(
            "sponsor_id",
            "SPN-1111",
            evidence_type=EvidenceType.INTAKE_FORM,
            confidence=0.78,
            applicant=None,
        )
        sponsor_sponsor = evidence(
            "sponsor_id",
            "SPN-2222",
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            confidence=0.93,
            applicant=None,
            page=1,
        )
        intake_arrival = evidence(
            "arrival_date",
            "2026-06-03",
            evidence_type=EvidenceType.INTAKE_FORM,
            confidence=0.71,
            applicant=None,
        )
        registry_arrival = evidence(
            "arrival_date",
            "2026-05-03",
            evidence_type=EvidenceType.REGISTRY_EXTRACT,
            confidence=0.96,
            page=2,
        )
        candidates = (
            intake_visa,
            sponsor_visa,
            sponsor_name,
            intake_sponsor,
            sponsor_sponsor,
            intake_arrival,
            registry_arrival,
        )
        primary = resolved_case(
            values={
                "visa_class": "TRANSIT-7",
                "sponsor_id": "SPN-1111",
                "arrival_date": "2026-06-03",
            },
            considered={
                "visa_class": (intake_visa, sponsor_visa),
                "sponsor_id": (intake_sponsor, sponsor_sponsor),
                "arrival_date": (intake_arrival, registry_arrival),
            },
        )
        rapid = resolved_case()
        primary_row = row(
            visa_class="TRANSIT-7",
            sponsor_id="SPN-1111",
            arrival_date="2026-06-03",
            adjudication="DENIED",
            confidence=0.61,
        )
        recovery, _renderer, _linker, resolver, adjudicator, factory = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
            primary_candidates=candidates,
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        expected = primary_row.to_dict()
        expected.update(
            {
                "visa_class": "XW-1",
                "sponsor_id": "SPN-2222",
                "arrival_date": "2026-05-03",
            }
        )
        self.assertEqual(result.to_dict(), expected)
        self.assertEqual(result.adjudication, "DENIED")
        self.assertEqual(result.confidence, 0.61)
        self.assertEqual(resolver.calls, 1)
        self.assertEqual(adjudicator.calls, 1)
        self.assertEqual(factory.calls, 0)

    def test_source_priority_repairs_abstain_on_scope_and_conflicts(self):
        intake = evidence(
            "visa_class",
            "TRANSIT-7",
            evidence_type=EvidenceType.INTAKE_FORM,
            confidence=0.82,
            applicant=None,
        )
        variants = {
            "below_frozen_confidence": (
                evidence(
                    "visa_class",
                    "XW-1",
                    evidence_type=EvidenceType.SPONSOR_ATTESTATION,
                    confidence=0.899,
                    page=1,
                ),
            ),
            "foreign_case": (
                evidence(
                    "visa_class",
                    "XW-1",
                    evidence_type=EvidenceType.SPONSOR_ATTESTATION,
                    confidence=0.95,
                    case_id="MIB-999999",
                    applicant=None,
                    page=1,
                ),
                evidence(
                    "applicant_name",
                    APPLICANT,
                    evidence_type=EvidenceType.SPONSOR_ATTESTATION,
                    confidence=0.95,
                    page=1,
                ),
            ),
            "conflicting_sponsor_values": (
                evidence(
                    "visa_class",
                    "XW-1",
                    evidence_type=EvidenceType.SPONSOR_ATTESTATION,
                    confidence=0.95,
                    page=1,
                ),
                evidence(
                    "visa_class",
                    "DIP-1",
                    evidence_type=EvidenceType.SPONSOR_ATTESTATION,
                    confidence=0.96,
                    page=1,
                ),
            ),
            "unsafe_visual_cue": (
                evidence(
                    "visa_class",
                    "XW-1",
                    evidence_type=EvidenceType.SPONSOR_ATTESTATION,
                    confidence=0.95,
                    cues=("sample_denial_watermark",),
                    page=1,
                ),
            ),
        }
        primary = resolved_case(
            values={"visa_class": "TRANSIT-7"},
            considered={"visa_class": (intake,)},
        )
        primary_row = row(visa_class="TRANSIT-7")

        for label, sponsor_candidates in variants.items():
            with self.subTest(label=label):
                recovery, *_rest = processor(
                    primary,
                    resolved_case(),
                    primary_outcome=outcome(primary_row),
                    primary_candidates=(intake, *sponsor_candidates),
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result, primary_row)

    def test_resolved_literal_unknown_does_not_route_rapid(self):
        primary = resolved_case(values={"fee_status": "unknown"})
        rapid = resolved_case(values={"fee_status": "paid"})
        primary_row = row(fee_status="unknown")
        recovery, renderer, _linker, resolver, adjudicator, factory = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result, primary_row)
        self.assertEqual(renderer.calls, 1)
        self.assertEqual(resolver.calls, 1)
        self.assertEqual(adjudicator.calls, 1)
        self.assertEqual(factory.calls, 0)

    def test_overlays_only_unknown_values_and_preserves_existing_priors(self):
        primary = resolved_case(
            values={"fee_status": "unknown"},
            unknown={"species_code", "home_world"},
        )
        rapid = resolved_case(
            values={
                "species_code": "ARCTURIAN",
                "fee_status": "paid",
            },
            unknown={"home_world"},
        )
        primary_row = row(
            species_code="TRIANGULAN",
            home_world="Wolf-1061c",
            fee_status="unknown",
            adjudication="NEEDS_REVIEW",
            confidence=0.37,
        )
        recovery, renderer, _linker, resolver, adjudicator, factory = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.species_code, "ARCTURIAN")
        self.assertEqual(result.home_world, "Wolf-1061c")
        self.assertEqual(result.fee_status, "unknown")
        self.assertEqual(result.adjudication, "NEEDS_REVIEW")
        self.assertEqual(result.confidence, 0.37)
        self.assertEqual(renderer.calls, 1)
        self.assertEqual(resolver.calls, 2)
        self.assertEqual(adjudicator.calls, 1)
        self.assertEqual(factory.calls, 1)

    def test_rapid_literal_unknown_preserves_primary_fee_prior(self):
        primary = resolved_case(unknown={"fee_status"})
        rapid = resolved_case(values={"fee_status": "unknown"})
        primary_row = row(
            fee_status="paid",
            adjudication="NEEDS_REVIEW",
            confidence=0.37,
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.fee_status, "paid")
        self.assertEqual(result.adjudication, "NEEDS_REVIEW")
        self.assertEqual(result.confidence, 0.37)

    def test_recovers_rapid_active_applicant_when_primary_is_absent(self):
        primary = resolved_case(unknown={"applicant_name"}, active=None)
        rapid = resolved_case(values={"applicant_name": "Miraul Miraquell"})
        primary_row = row(applicant_name="unknown")
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
            primary_active=None,
            rapid_active="Miraul Miraquell",
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.applicant_name, "Miraul Miraquell")

    def test_non_none_risk_recovers_only_when_primary_risk_is_unknown(self):
        anchor = evidence("risk_flags", None, applicant=None)
        primary = resolved_case(
            unknown={"risk_flags"},
            considered={"risk_flags": (anchor,)},
        )
        rapid = resolved_case(values={"risk_flags": "active_warrant"})
        recovery, *_rest = processor(primary, rapid)

        recovered = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(recovered.risk_flags, "active_warrant")

        primary_resolved = resolved_case(unknown={"species_code"})
        rapid_risk = resolved_case(
            values={
                "species_code": "ARCTURIAN",
                "risk_flags": "active_warrant",
            }
        )
        protected, *_rest = processor(primary_resolved, rapid_risk)

        protected_row = protected.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(protected_row.risk_flags, "none")

    def test_rapid_none_does_not_fill_an_unknown_risk(self):
        anchor = evidence("risk_flags", None, applicant=None)
        primary = resolved_case(
            unknown={"risk_flags"},
            considered={"risk_flags": (anchor,)},
        )
        rapid = resolved_case(values={"risk_flags": "none"})
        recovery, *_rest = processor(primary, rapid)

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.risk_flags, "none")

    def test_semantic_head_denies_each_visible_disqualifying_rapid_risk(self):
        for risk_flag in (
            "memory_tampering",
            "planetary_embargo",
            "active_warrant",
            "biohazard_red",
        ):
            with self.subTest(risk_flag=risk_flag):
                anchor = evidence("risk_flags", None, applicant=None)
                recovered = evidence(
                    "risk_flags",
                    risk_flag,
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    applicant=None,
                )
                primary = resolved_case(
                    unknown={"risk_flags"},
                    considered={"risk_flags": (anchor,)},
                )
                rapid = resolved_case(
                    values={"risk_flags": risk_flag},
                    considered={"risk_flags": (recovered,)},
                )
                recovery, *_rest = processor(
                    primary,
                    rapid,
                    rapid_candidates=(recovered,),
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result.risk_flags, risk_flag)
                self.assertEqual(result.adjudication, "DENIED")
                self.assertEqual(result.confidence, SEMANTIC_DENIAL_CONFIDENCE)

    def test_semantic_head_denies_visible_embargo_facts_only(self):
        for home_world in ("Eris Relay", "TRAPPIST-1e"):
            with self.subTest(home_world=home_world):
                recovered_home = evidence(
                    "home_world",
                    home_world,
                    evidence_type=EvidenceType.REGISTRY_EXTRACT,
                )
                primary = resolved_case(unknown={"home_world"})
                rapid = resolved_case(
                    values={"home_world": home_world},
                    considered={"home_world": (recovered_home,)},
                )
                recovery, *_rest = processor(
                    primary,
                    rapid,
                    rapid_candidates=(recovered_home,),
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result.home_world, home_world)
                self.assertEqual(result.adjudication, "DENIED")
                self.assertEqual(result.confidence, SEMANTIC_DENIAL_CONFIDENCE)

        visible_home = evidence(
            "home_world",
            "Wolf-1061c",
            evidence_type=EvidenceType.REGISTRY_EXTRACT,
        )
        recovered_visa = evidence("visa_class", "XW-1")
        primary = resolved_case(
            values={"home_world": "Wolf-1061c"},
            unknown={"visa_class"},
            considered={"home_world": (visible_home,)},
        )
        rapid = resolved_case(
            values={"visa_class": "XW-1"},
            considered={"visa_class": (recovered_visa,)},
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_outcome=outcome(
                row(home_world="Wolf-1061c", visa_class="MED-3")
            ),
            primary_candidates=(visible_home,),
            rapid_candidates=(recovered_visa,),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.home_world, "Wolf-1061c")
        self.assertEqual(result.visa_class, "XW-1")
        self.assertEqual(result.adjudication, "DENIED")
        self.assertEqual(result.confidence, SEMANTIC_DENIAL_CONFIDENCE)

    def test_semantic_barred_sponsor_requires_two_rapid_winners(self):
        recovered_sponsor = evidence("sponsor_id", "SPN-7331")
        recovered_visa = evidence("visa_class", "MED-3")
        primary = resolved_case(unknown={"sponsor_id", "visa_class"})
        rapid = resolved_case(
            values={"sponsor_id": "SPN-7331", "visa_class": "MED-3"},
            considered={
                "sponsor_id": (recovered_sponsor,),
                "visa_class": (recovered_visa,),
            },
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            rapid_candidates=(recovered_sponsor, recovered_visa),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.sponsor_id, "SPN-7331")
        self.assertEqual(result.visa_class, "MED-3")
        self.assertEqual(result.adjudication, "DENIED")
        self.assertEqual(result.confidence, SEMANTIC_DENIAL_CONFIDENCE)

        primary_visa = evidence("visa_class", "XW-1")
        primary = resolved_case(
            values={"visa_class": "XW-1"},
            unknown={"sponsor_id"},
            considered={"visa_class": (primary_visa,)},
        )
        rapid = resolved_case(
            values={"sponsor_id": "SPN-7331"},
            considered={"sponsor_id": (recovered_sponsor,)},
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_candidates=(primary_visa,),
            rapid_candidates=(recovered_sponsor,),
        )

        one_rapid_winner = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(one_rapid_winner.sponsor_id, "SPN-7331")
        self.assertEqual(one_rapid_winner.adjudication, "NEEDS_REVIEW")
        self.assertEqual(one_rapid_winner.confidence, 0.37)

    def test_semantic_head_rejects_priors_bad_cues_and_wrong_scope(self):
        anchor = evidence("risk_flags", None, applicant=None)
        variants = {
            "no_winning_evidence": (
                resolved_case(values={"risk_flags": "active_warrant"}),
                (),
            ),
            "wrong_case": (
                resolved_case(
                    values={"risk_flags": "active_warrant"},
                    considered={
                        "risk_flags": (
                            evidence(
                                "risk_flags",
                                "active_warrant",
                                case_id="MIB-999999",
                                applicant=None,
                            ),
                        )
                    },
                ),
                (
                    evidence(
                        "risk_flags",
                        "active_warrant",
                        case_id="MIB-999999",
                        applicant=None,
                    ),
                ),
            ),
            "sample_watermark": (
                resolved_case(
                    values={"risk_flags": "active_warrant"},
                    considered={
                        "risk_flags": (
                            evidence(
                                "risk_flags",
                                "active_warrant",
                                cues=("sample_denial_watermark",),
                                applicant=None,
                            ),
                        )
                    },
                ),
                (
                    evidence(
                        "risk_flags",
                        "active_warrant",
                        cues=("sample_denial_watermark",),
                        applicant=None,
                    ),
                ),
            ),
            "text_layer": (
                resolved_case(
                    values={"risk_flags": "active_warrant"},
                    considered={
                        "risk_flags": (
                            evidence(
                                "risk_flags",
                                "active_warrant",
                                evidence_type=EvidenceType.TEXT_LAYER,
                                applicant=None,
                            ),
                        )
                    },
                ),
                (
                    evidence(
                        "risk_flags",
                        "active_warrant",
                        evidence_type=EvidenceType.TEXT_LAYER,
                        applicant=None,
                    ),
                ),
            ),
        }
        primary = resolved_case(
            unknown={"risk_flags"},
            considered={"risk_flags": (anchor,)},
        )
        for label, (rapid, candidates) in variants.items():
            with self.subTest(label=label):
                recovery, *_rest = processor(
                    primary,
                    rapid,
                    rapid_candidates=candidates,
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result.adjudication, "NEEDS_REVIEW")
                self.assertEqual(result.confidence, 0.37)

        prior_home = resolved_case(
            values={"home_world": "Wolf-1061c", "visa_class": "XW-1"},
            unknown={"home_world"},
        )
        rapid_without_home = resolved_case(unknown={"home_world"})
        recovery, *_rest = processor(
            prior_home,
            rapid_without_home,
            primary_outcome=outcome(
                row(home_world="Wolf-1061c", visa_class="XW-1")
            ),
        )

        serialized_prior = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(serialized_prior.home_world, "Wolf-1061c")
        self.assertEqual(serialized_prior.adjudication, "NEEDS_REVIEW")

    def test_authority_vetoes_semantic_denial_and_transit_never_triggers_it(self):
        anchor = evidence("risk_flags", None, applicant=None)
        recovered_risk = evidence(
            "risk_flags",
            "active_warrant",
            evidence_type=EvidenceType.BIOMETRIC_SLIP,
            applicant=None,
        )
        primary = resolved_case(
            unknown={"risk_flags"},
            considered={"risk_flags": (anchor,)},
        )
        rapid = resolved_case(
            values={"risk_flags": "active_warrant"},
            considered={"risk_flags": (recovered_risk,)},
        )
        authoritative_review = evidence(
            "adjudication",
            "NEEDS_REVIEW",
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
            confidence=0.95,
            applicant=None,
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            rapid_candidates=(recovered_risk, authoritative_review),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.risk_flags, "active_warrant")
        self.assertEqual(result.adjudication, "NEEDS_REVIEW")
        self.assertEqual(result.confidence, 0.37)

        primary_authority = AdjudicationOutcome(
            row=row(),
            trace=DecisionTrace(
                decision="NEEDS_REVIEW",
                authoritative_source=True,
                denial_reasons=(),
                review_reasons=("authoritative_visible_decision",),
                approval_facts=(),
                exception_ids=(),
            ),
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_outcome=primary_authority,
            rapid_candidates=(recovered_risk,),
        )

        primary_veto = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(primary_veto.adjudication, "NEEDS_REVIEW")
        self.assertEqual(primary_veto.confidence, 0.37)

        recovered_transit = evidence("visa_class", "TRANSIT-7")
        transit_primary = resolved_case(unknown={"visa_class"})
        transit_rapid = resolved_case(
            values={"visa_class": "TRANSIT-7"},
            considered={"visa_class": (recovered_transit,)},
        )
        recovery, *_rest = processor(
            transit_primary,
            transit_rapid,
            rapid_candidates=(recovered_transit,),
        )

        transit = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(transit.visa_class, "TRANSIT-7")
        self.assertEqual(transit.adjudication, "NEEDS_REVIEW")
        self.assertEqual(transit.confidence, 0.37)

    def test_semantic_denial_preserves_biometric_applicant_and_fee_recovery(self):
        intake = evidence(
            "applicant_name",
            APPLICANT,
            evidence_type=EvidenceType.INTAKE_FORM,
            confidence=0.84,
        )
        biometric = evidence(
            "applicant_name",
            "Zed Zornax",
            evidence_type=EvidenceType.BIOMETRIC_SLIP,
            confidence=0.91,
        )
        risk_anchor = evidence("risk_flags", None, applicant=None)
        rapid_risk = evidence(
            "risk_flags",
            "memory_tampering",
            evidence_type=EvidenceType.BIOMETRIC_SLIP,
            applicant=None,
        )
        rapid_fee = evidence("fee_status", "waived", applicant=None)
        primary = resolved_case(
            unknown={"applicant_name", "fee_status", "risk_flags"},
            considered={"risk_flags": (risk_anchor,)},
        )
        rapid = resolved_case(
            values={"fee_status": "waived", "risk_flags": "memory_tampering"},
            considered={
                "fee_status": (rapid_fee,),
                "risk_flags": (rapid_risk,),
            },
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_candidates=(intake, biometric),
            rapid_candidates=(rapid_fee, rapid_risk),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.applicant_name, "Zed Zornax")
        self.assertEqual(result.fee_status, "waived")
        self.assertEqual(result.risk_flags, "memory_tampering")
        self.assertEqual(result.adjudication, "DENIED")
        self.assertEqual(result.confidence, SEMANTIC_DENIAL_CONFIDENCE)

    def test_exact_unanimous_authoritative_note_can_change_review_only(self):
        primary = resolved_case(unknown={"fee_status"})
        rapid = resolved_case(unknown={"fee_status"})
        candidate = evidence(
            "adjudication",
            "DENIED",
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
            confidence=0.90,
        )
        primary_row = row(adjudication="NEEDS_REVIEW", confidence=0.37)
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
            rapid_candidates=(candidate,),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.adjudication, "DENIED")
        self.assertEqual(result.confidence, 0.37)

    def test_xw1_multisource_recovery_accepts_only_the_two_audited_fee_shapes(self):
        required_facts = (
            "application_date_current_or_exempt",
            "sponsor_present_and_not_publicly_barred",
        )
        variants = {
            "paid_with_only_unknown_risk": {
                "fee_status": "paid",
                "unknown": {"risk_flags"},
                "review_reasons": (
                    "required_output_unknown:risk_flags",
                    "risk_flags_unknown",
                ),
                "approval_facts": (*required_facts, "fee_paid"),
            },
            "unsupported_waiver_only": {
                "fee_status": "waived",
                "unknown": set(),
                "review_reasons": ("unsupported_fee_waiver",),
                "approval_facts": required_facts,
            },
        }

        for label, values in variants.items():
            with self.subTest(label=label):
                primary_row = row(
                    visa_class="XW-1",
                    fee_status=values["fee_status"],
                    confidence=0.25,
                )
                primary = resolved_case(
                    values={
                        "visa_class": "XW-1",
                        "fee_status": values["fee_status"],
                    },
                    unknown=values["unknown"],
                )
                recovery, _renderer, _linker, _resolver, _adjudicator, factory = (
                    processor(
                        primary,
                        resolved_case(),
                        primary_outcome=outcome(
                            primary_row,
                            review_reasons=values["review_reasons"],
                            approval_facts=values["approval_facts"],
                        ),
                        primary_candidates=xw1_multisource_candidates(),
                    )
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                expected = primary_row.to_dict()
                expected.update(
                    {
                        "adjudication": "APPROVED",
                        "confidence": (
                            XW1_MULTISOURCE_REVIEW_APPROVAL_CONFIDENCE
                        ),
                    }
                )
                self.assertEqual(result.to_dict(), expected)
                self.assertEqual(factory.calls, 0)

    def test_xw1_multisource_recovery_vetoes_incomplete_or_unsafe_policy_state(self):
        candidates = xw1_multisource_candidates()
        review_reasons = (
            "required_output_unknown:risk_flags",
            "risk_flags_unknown",
        )
        approval_facts = (
            "application_date_current_or_exempt",
            "sponsor_present_and_not_publicly_barred",
            "fee_paid",
        )
        variants = {
            "confidence_above_ceiling": {
                "prediction": row(visa_class="XW-1", confidence=0.250001),
            },
            "incomplete_output": {
                "prediction": row(
                    visa_class="XW-1",
                    sponsor_id="SPN-0000",
                    confidence=0.25,
                ),
            },
            "non_none_final_risk": {
                "prediction": row(
                    visa_class="XW-1",
                    risk_flags="identity_conflict",
                    confidence=0.25,
                ),
            },
            "extra_review_reason": {
                "prediction": row(visa_class="XW-1", confidence=0.25),
                "review_reasons": (*review_reasons, "review_flag:sponsor_mismatch"),
            },
            "missing_required_fact": {
                "prediction": row(visa_class="XW-1", confidence=0.25),
                "approval_facts": ("application_date_current_or_exempt", "fee_paid"),
            },
            "policy_denial_present": {
                "prediction": row(visa_class="XW-1", confidence=0.25),
                "denial_reasons": ("barred_sponsor:SPN-1042",),
            },
            "unresolved_linkage": {
                "prediction": row(visa_class="XW-1", confidence=0.25),
                "unresolved_linkage": True,
            },
        }

        for label, values in variants.items():
            with self.subTest(label=label):
                primary_row = values["prediction"]
                primary = resolved_case(
                    values={"visa_class": "XW-1"},
                    unknown={"risk_flags"},
                    unresolved_linkage=values.get("unresolved_linkage", False),
                    unresolved_reasons=("ambiguous_packet",)
                    if values.get("unresolved_linkage")
                    else (),
                )
                recovery, *_rest = processor(
                    primary,
                    resolved_case(),
                    primary_outcome=outcome(
                        primary_row,
                        review_reasons=values.get(
                            "review_reasons",
                            review_reasons,
                        ),
                        approval_facts=values.get(
                            "approval_facts",
                            approval_facts,
                        ),
                        denial_reasons=values.get("denial_reasons", ()),
                    ),
                    primary_candidates=candidates,
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result, primary_row)

    def test_xw1_multisource_recovery_vetoes_provenance_and_evidence_conflicts(self):
        primary_row = row(visa_class="XW-1", confidence=0.25)
        primary = resolved_case(
            values={"visa_class": "XW-1"},
            unknown={"risk_flags"},
        )
        base_candidates = xw1_multisource_candidates()
        approval_facts = (
            "application_date_current_or_exempt",
            "sponsor_present_and_not_publicly_barred",
            "fee_paid",
        )
        variants = {
            "missing_same_page_registry_applicant": tuple(
                candidate
                for candidate in base_candidates
                if not (
                    candidate.field_name == "applicant_name"
                    and candidate.evidence_type is EvidenceType.REGISTRY_EXTRACT
                )
            ),
            "wrong_case_sponsor_fact": tuple(
                candidate
                for candidate in base_candidates
                if candidate.field_name != "visa_class"
            )
            + (
                evidence(
                    "visa_class",
                    "XW-1",
                    evidence_type=EvidenceType.SPONSOR_ATTESTATION,
                    case_id="MIB-999999",
                    page=10,
                ),
            ),
            "conflicting_registry_home": base_candidates
            + (
                evidence(
                    "home_world",
                    "Barnard's Star b",
                    evidence_type=EvidenceType.REGISTRY_EXTRACT,
                    page=11,
                ),
            ),
            "bad_cue_anywhere": base_candidates
            + (
                evidence(
                    "declared_purpose",
                    "sample",
                    cues=("sample_denial_watermark",),
                    page=12,
                ),
            ),
            "signed_page_anywhere": base_candidates
            + (
                evidence(
                    "adjudication",
                    "NEEDS_REVIEW",
                    evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
                    page=12,
                ),
            ),
            "visible_non_none_risk": base_candidates
            + (
                evidence(
                    "risk_flags",
                    "active_warrant",
                    evidence_type=EvidenceType.BIOMETRIC_SLIP,
                    page=12,
                ),
            ),
        }

        for label, candidates in variants.items():
            with self.subTest(label=label):
                recovery, *_rest = processor(
                    primary,
                    resolved_case(),
                    primary_outcome=outcome(
                        primary_row,
                        review_reasons=(
                            "required_output_unknown:risk_flags",
                            "risk_flags_unknown",
                        ),
                        approval_facts=approval_facts,
                    ),
                    primary_candidates=candidates,
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result, primary_row)

    def test_xw1_multisource_recovery_preserves_primary_and_rapid_authority(self):
        candidates = xw1_multisource_candidates()
        primary_row = row(visa_class="XW-1", confidence=0.25)
        review_reasons = (
            "required_output_unknown:risk_flags",
            "risk_flags_unknown",
        )
        approval_facts = (
            "application_date_current_or_exempt",
            "sponsor_present_and_not_publicly_barred",
            "fee_paid",
        )
        authoritative_primary = outcome(
            primary_row,
            review_reasons=review_reasons,
            approval_facts=approval_facts,
            authoritative_source=True,
        )
        recovery, *_rest = processor(
            resolved_case(
                values={"visa_class": "XW-1"},
                unknown={"risk_flags"},
            ),
            resolved_case(),
            primary_outcome=authoritative_primary,
            primary_candidates=candidates,
        )

        primary_result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(primary_result, primary_row)

        rapid_authority = evidence(
            "adjudication",
            "NEEDS_REVIEW",
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
            confidence=0.95,
        )
        primary = resolved_case(
            values={"visa_class": "XW-1"},
            unknown={"risk_flags", "species_code"},
        )
        rapid = resolved_case(
            values={"species_code": "ARCTURIAN", "risk_flags": "none"},
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_outcome=outcome(
                primary_row,
                review_reasons=review_reasons,
                approval_facts=approval_facts,
            ),
            primary_candidates=candidates,
            rapid_candidates=(rapid_authority,),
        )

        rapid_result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(rapid_result.adjudication, "NEEDS_REVIEW")
        self.assertEqual(rapid_result.confidence, 0.25)

    def test_frozen_review_approval_head_matches_each_early_return_branch(self):
        six_applicant_candidates = tuple(
            evidence("applicant_name", APPLICANT, page=page)
            for page in range(6)
        )
        branches = {
            "six_primary_applicant_candidates": {
                "prediction": row(arrival_date="2026-04-27"),
                "candidates": six_applicant_candidates,
                "review_reasons": ("test_review",),
                "approval_facts": (),
            },
            "clean_bio_and_age_over_71_days": {
                "prediction": row(arrival_date="2026-04-26"),
                "candidates": (),
                "review_reasons": ("test_review",),
                "approval_facts": ("no_visible_biohazard_risk",),
            },
            "required_sponsor_unknown_and_age_at_most_48_days": {
                "prediction": row(arrival_date="2026-05-20"),
                "candidates": (),
                "review_reasons": ("required_sponsor_unknown",),
                "approval_facts": (),
            },
        }

        for label, values in branches.items():
            with self.subTest(label=label):
                primary_row = values["prediction"]
                recovery, _renderer, _linker, _resolver, _adjudicator, factory = (
                    processor(
                        resolved_case(),
                        resolved_case(),
                        primary_outcome=outcome(
                            primary_row,
                            review_reasons=values["review_reasons"],
                            approval_facts=values["approval_facts"],
                        ),
                        primary_candidates=values["candidates"],
                    )
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                expected = primary_row.to_dict()
                expected.update(
                    {
                        "adjudication": "APPROVED",
                        "confidence": REVIEW_APPROVAL_CONFIDENCE,
                    }
                )
                self.assertEqual(result.to_dict(), expected)
                self.assertEqual(factory.calls, 0)

    def test_review_approval_head_enforces_strict_boundaries_and_common_guards(self):
        five_applicant_candidates = tuple(
            evidence("applicant_name", APPLICANT, page=page)
            for page in range(5)
        )
        six_applicant_candidates = five_applicant_candidates + (
            evidence("applicant_name", APPLICANT, page=5),
        )
        variants = {
            "candidate_count_must_exceed_five": {
                "prediction": row(),
                "candidates": five_applicant_candidates,
                "review_reasons": ("test_review",),
                "approval_facts": (),
            },
            "older_boundary_is_strict": {
                "prediction": row(arrival_date="2026-04-27"),
                "candidates": (),
                "review_reasons": ("test_review",),
                "approval_facts": ("no_visible_biohazard_risk",),
            },
            "recent_boundary_rejects_49_days": {
                "prediction": row(arrival_date="2026-05-19"),
                "candidates": (),
                "review_reasons": ("required_sponsor_unknown",),
                "approval_facts": (),
            },
            "invalid_date_abstains": {
                "prediction": row(arrival_date="not-a-date"),
                "candidates": (),
                "review_reasons": ("required_sponsor_unknown",),
                "approval_facts": ("no_visible_biohazard_risk",),
            },
            "risk_must_normalize_to_none": {
                "prediction": row(risk_flags="illegible_biometrics"),
                "candidates": six_applicant_candidates,
                "review_reasons": ("test_review",),
                "approval_facts": (),
            },
            "final_decision_must_remain_review": {
                "prediction": row(adjudication="DENIED", confidence=0.61),
                "candidates": six_applicant_candidates,
                "review_reasons": ("test_review",),
                "approval_facts": (),
            },
        }

        for label, values in variants.items():
            with self.subTest(label=label):
                primary_row = values["prediction"]
                recovery, *_rest = processor(
                    resolved_case(),
                    resolved_case(),
                    primary_outcome=outcome(
                        primary_row,
                        review_reasons=values["review_reasons"],
                        approval_facts=values["approval_facts"],
                    ),
                    primary_candidates=values["candidates"],
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result, primary_row)

        normalized_risk_row = row(risk_flags="NoNe")
        recovery, *_rest = processor(
            resolved_case(),
            resolved_case(),
            primary_outcome=outcome(normalized_risk_row),
            primary_candidates=six_applicant_candidates,
        )

        normalized_risk = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(normalized_risk.adjudication, "APPROVED")
        self.assertEqual(
            normalized_risk.confidence,
            REVIEW_APPROVAL_CONFIDENCE,
        )

    def test_review_approval_head_runs_after_rapid_output_recovery(self):
        six_applicant_candidates = tuple(
            evidence("applicant_name", APPLICANT, page=page)
            for page in range(6)
        )
        primary = resolved_case(unknown={"species_code"})
        rapid = resolved_case(values={"species_code": "ARCTURIAN"})
        recovery, _renderer, _linker, _resolver, _adjudicator, factory = (
            processor(
                primary,
                rapid,
                primary_candidates=six_applicant_candidates,
            )
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result.species_code, "ARCTURIAN")
        self.assertEqual(result.adjudication, "APPROVED")
        self.assertEqual(result.confidence, REVIEW_APPROVAL_CONFIDENCE)
        self.assertEqual(factory.calls, 1)

    def test_review_approval_head_preserves_authority_and_denial_precedence(self):
        six_applicant_candidates = tuple(
            evidence("applicant_name", APPLICANT, page=page)
            for page in range(6)
        )
        authoritative_primary = outcome(
            row(),
            review_reasons=("authoritative_visible_decision",),
            authoritative_source=True,
        )
        recovery, *_rest = processor(
            resolved_case(),
            resolved_case(),
            primary_outcome=authoritative_primary,
            primary_candidates=six_applicant_candidates,
        )

        primary_result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(primary_result.adjudication, "NEEDS_REVIEW")
        self.assertEqual(primary_result.confidence, 0.37)

        authoritative_rapid_review = evidence(
            "adjudication",
            "NEEDS_REVIEW",
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
            confidence=0.95,
        )
        primary = resolved_case(unknown={"species_code"})
        rapid = resolved_case(values={"species_code": "ARCTURIAN"})
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_candidates=six_applicant_candidates,
            rapid_candidates=(authoritative_rapid_review,),
        )

        rapid_result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(rapid_result.adjudication, "NEEDS_REVIEW")
        self.assertEqual(rapid_result.confidence, 0.37)

        risk_anchor = evidence("risk_flags", None, applicant=None)
        recovered_risk = evidence(
            "risk_flags",
            "active_warrant",
            evidence_type=EvidenceType.BIOMETRIC_SLIP,
            applicant=None,
        )
        primary = resolved_case(
            unknown={"risk_flags"},
            considered={"risk_flags": (risk_anchor,)},
        )
        rapid = resolved_case(
            values={"risk_flags": "active_warrant"},
            considered={"risk_flags": (recovered_risk,)},
        )
        recovery, *_rest = processor(
            primary,
            rapid,
            primary_candidates=six_applicant_candidates + (risk_anchor,),
            rapid_candidates=(recovered_risk,),
        )

        denied = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(denied.adjudication, "DENIED")
        self.assertEqual(denied.confidence, SEMANTIC_DENIAL_CONFIDENCE)

    def test_authoritative_conflict_or_bad_scope_abstains(self):
        primary = resolved_case(unknown={"fee_status"})
        rapid = resolved_case(unknown={"fee_status"})
        variants = (
            (
                evidence(
                    "adjudication",
                    "APPROVED",
                    evidence_type=EvidenceType.ADJUDICATOR_STAMP,
                ),
                evidence(
                    "adjudication",
                    "DENIED",
                    evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
                ),
            ),
            (
                evidence(
                    "adjudication",
                    "DENIED",
                    evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
                    case_id="MIB-999999",
                ),
            ),
            (
                evidence(
                    "adjudication",
                    "DENIED",
                    evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
                    applicant="Other Applicant",
                ),
            ),
            (
                evidence(
                    "adjudication",
                    "DENIED",
                    evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
                    confidence=0.899,
                ),
            ),
            (
                evidence(
                    "adjudication",
                    "DENIED",
                    evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
                    cues=("strikethrough",),
                ),
            ),
        )
        for candidates in variants:
            with self.subTest(candidates=candidates):
                primary_row = row(adjudication="NEEDS_REVIEW", confidence=0.37)
                recovery, *_rest = processor(
                    primary,
                    rapid,
                    primary_outcome=outcome(primary_row),
                    rapid_candidates=candidates,
                )

                result = recovery.process_case(Path(CASE_ID + ".pdf"))

                self.assertEqual(result, primary_row)

    def test_any_rapid_exception_fails_closed_to_primary_row(self):
        primary = resolved_case(unknown={"species_code"})
        rapid = resolved_case(values={"species_code": "ARCTURIAN"})
        primary_row = row(species_code="TRIANGULAN")
        recovery, renderer, _linker, _resolver, adjudicator, factory = processor(
            primary,
            rapid,
            primary_outcome=outcome(primary_row),
            rapid_error=RuntimeError("onnx failure"),
        )

        result = recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(result, primary_row)
        self.assertEqual(renderer.calls, 1)
        self.assertEqual(adjudicator.calls, 1)
        self.assertEqual(factory.calls, 1)

    def test_rapid_extractor_is_reused_within_one_worker_thread(self):
        primary = resolved_case(unknown={"species_code"})
        rapid = resolved_case(values={"species_code": "ARCTURIAN"})
        recovery, *_components, factory = processor(primary, rapid)

        recovery.process_case(Path(CASE_ID + ".pdf"))
        recovery.process_case(Path(CASE_ID + ".pdf"))

        self.assertEqual(factory.calls, 1)
        self.assertEqual(factory.instances[0].calls, 2)


if __name__ == "__main__":
    unittest.main()
