import unittest
from dataclasses import replace

from mib_pipeline import (
    AdjudicationOutcome,
    CandidateEvidence,
    CaseLinker,
    DecisionTrace,
    EvidencePrecedenceResolver,
    EvidenceType,
    FieldState,
    OcrLine,
    PredictionRow,
    Rect,
    REVIEW_APPROVAL_CONFIDENCE,
    REVIEW_DENIAL_CONFIDENCE,
    REVIEW_DIPLOMATIC_APPROVAL_CONFIDENCE,
    RenderedPage,
    ResolvedCase,
    ResolvedField,
    ReviewDenialRecoveryAdjudicator,
    VisibleEvidenceExtractor,
)


CASE_ID = "MIB-000001"


def row(**overrides):
    values = {
        "case_id": CASE_ID,
        "applicant_name": "Veenax Qortari",
        "species_code": "ANDROMEDAN",
        "home_world": "Mars Dome-7",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-1042",
        "arrival_date": "2026-04-17",
        "declared_purpose": "technical work",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.25,
    }
    values.update(overrides)
    return PredictionRow.from_mapping(values)


def outcome(
    *,
    review_reasons=(),
    denial_reasons=(),
    approval_facts=("fee_paid",),
    prediction=None,
    trace_decision=None,
):
    prediction = prediction or row()
    decision = trace_decision or prediction.adjudication
    return AdjudicationOutcome(
        row=prediction,
        trace=DecisionTrace(
            decision=decision,
            authoritative_source=False,
            denial_reasons=tuple(denial_reasons),
            review_reasons=tuple(review_reasons),
            approval_facts=tuple(approval_facts),
            exception_ids=(),
        ),
    )


def marker(field_name, page_type, *, source="visible_ocr", cues=None):
    evidence = CandidateEvidence(
        field_name=field_name,
        value="present",
        evidence_type=(
            EvidenceType.SPONSOR_ATTESTATION
            if page_type == "sponsor_attestation"
            else EvidenceType.INTAKE_FORM
        ),
        page_index=0,
        box=Rect(0, 0, 100, 30),
        legible=True,
        superseded=False,
        ocr_confidence=0.95,
        visual_cues=(f"packet_page_type:{page_type}",) if cues is None else cues,
        source=source,
        case_id_hint=CASE_ID,
        applicant_hint=None,
    )
    return ResolvedField(
        field_name=field_name,
        state=FieldState.RESOLVED,
        value="present",
        winning_evidence=evidence,
        considered=(evidence,),
        reason="test marker",
    )


def resolved_case(*markers):
    return ResolvedCase(
        case_id=CASE_ID,
        active_applicant="Veenax Qortari",
        fields={item.field_name: item for item in markers},
        unresolved_linkage=False,
        unresolved_reasons=(),
    )


def visible_field(
    field_name,
    value,
    *,
    source="visible_ocr",
    evidence_type=EvidenceType.INTAKE_FORM,
    superseded=False,
    cues=(),
):
    evidence = CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=1,
        box=Rect(0, 40, 100, 70),
        legible=True,
        superseded=superseded,
        ocr_confidence=0.95,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint=CASE_ID,
        applicant_hint="Veenax Qortari",
    )
    return ResolvedField(
        field_name=field_name,
        state=FieldState.RESOLVED,
        value=value,
        winning_evidence=evidence,
        considered=(evidence,),
        reason="test visible field",
    )


class FakeAdjudicator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def adjudicate_case(self, resolved):
        self.calls += 1
        return self.result


def recover(result, resolved):
    baseline = FakeAdjudicator(result)
    recovered = ReviewDenialRecoveryAdjudicator(baseline).adjudicate_case(resolved)
    return recovered, baseline


def line(text, index=0):
    return OcrLine(
        page_index=0,
        text=text,
        confidence=0.91,
        box=Rect(0, index * 20, 200, index * 20 + 15),
        tokens=(),
    )


class PacketPageTypeMarkerTests(unittest.TestCase):
    def test_classification_exactly_uses_first_four_lines_and_mining_priority(self):
        self.assertEqual(
            VisibleEvidenceExtractor.packet_page_type(
                (line("Sponsor Attestation Letter"),)
            ),
            "sponsor_attestation",
        )
        self.assertEqual(
            VisibleEvidenceExtractor.packet_page_type(
                (line("Biometric Sponsor Attestation Letter"),)
            ),
            "biometric_slip",
        )
        self.assertEqual(
            VisibleEvidenceExtractor.packet_page_type(
                tuple(line("ordinary text", index) for index in range(4))
                + (line("Sponsor Attestation Letter", 4),)
            ),
            "other",
        )

    def test_only_frozen_types_create_visible_policy_markers(self):
        page = RenderedPage(
            index=0,
            image_png=b"",
            width_px=100,
            height_px=100,
            dpi=200,
            rotation_deg=0,
            skew_correction_deg=0.0,
            crop_box=Rect(0, 0, 100, 100),
            text_spans=(),
        )
        sponsor = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(line("Sponsor Letter"),),
            case_id=CASE_ID,
        )
        fee = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(line("MIB Fee Receipt"),),
            case_id=CASE_ID,
        )
        intake = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(line("FORM I-8090"),),
            case_id=CASE_ID,
        )

        self.assertIsNotNone(sponsor)
        self.assertEqual(
            sponsor.field_name,
            "page_type_present_sponsor_attestation",
        )
        self.assertEqual(sponsor.value, "present")
        self.assertEqual(sponsor.source, "visible_ocr")
        self.assertIsNotNone(fee)
        self.assertEqual(fee.field_name, "page_type_present_fee_receipt")
        self.assertEqual(fee.visual_cues, ("packet_page_type:fee_receipt",))
        self.assertIsNone(intake)

        linked = CaseLinker().link(CASE_ID, (sponsor, fee))
        resolved = EvidencePrecedenceResolver().resolve(linked)
        self.assertEqual(
            resolved.value("page_type_present_sponsor_attestation"),
            "present",
        )
        self.assertEqual(
            resolved.value("page_type_present_fee_receipt"),
            "present",
        )


class ReviewDiplomaticApprovalRecoveryTests(unittest.TestCase):
    FEE = marker("page_type_present_fee_receipt", "fee_receipt")
    DIPLOMATIC_VISA = visible_field("visa_class", "DIP-1")

    def assert_recovered(self, original, recovered):
        self.assertEqual(recovered.row.adjudication, "APPROVED")
        self.assertEqual(
            recovered.row.confidence,
            REVIEW_DIPLOMATIC_APPROVAL_CONFIDENCE,
        )
        self.assertEqual(recovered.trace.decision, "APPROVED")
        self.assertFalse(recovered.trace.authoritative_source)
        self.assertEqual(recovered.trace.review_reasons, ())
        self.assertEqual(recovered.trace.denial_reasons, ())
        self.assertIn(
            "review_diplomatic_fee_receipt_recovery",
            recovered.trace.approval_facts,
        )
        original_values = original.row.to_dict()
        recovered_values = recovered.row.to_dict()
        for field_name in original_values:
            if field_name not in {"adjudication", "confidence"}:
                self.assertEqual(
                    recovered_values[field_name],
                    original_values[field_name],
                    field_name,
                )

    def test_policy_fact_and_visible_dip_each_recover_at_frozen_boundary(self):
        cases = (
            (
                outcome(
                    approval_facts=(
                        "application_date_current_or_exempt",
                        "diplomatic_sponsor_exemption",
                    ),
                    prediction=row(visa_class="DIP-1", confidence=0.25),
                    review_reasons=("fee_status_unknown",),
                ),
                resolved_case(self.FEE),
            ),
            (
                outcome(
                    approval_facts=("application_date_current_or_exempt",),
                    prediction=row(visa_class="DIP-1", confidence=0.25),
                    review_reasons=("required_output_unknown:risk_flags",),
                ),
                resolved_case(self.FEE, self.DIPLOMATIC_VISA),
            ),
        )
        for original, resolved in cases:
            with self.subTest(approval_facts=original.trace.approval_facts):
                recovered, baseline = recover(original, resolved)
                self.assertEqual(baseline.calls, 1)
                self.assert_recovered(original, recovered)

    def test_approval_vetoes_missing_or_untrusted_features(self):
        diplomatic = outcome(
            approval_facts=("diplomatic_sponsor_exemption",),
            prediction=row(visa_class="DIP-1"),
        )
        second_fee = replace(self.FEE.winning_evidence, page_index=2)
        duplicate_fee_pages = replace(
            self.FEE,
            considered=(self.FEE.winning_evidence, second_fee),
        )
        variants = (
            (diplomatic, resolved_case()),
            (diplomatic, resolved_case(duplicate_fee_pages)),
            (
                diplomatic,
                resolved_case(
                    marker(
                        "page_type_present_fee_receipt",
                        "fee_receipt",
                        source="text_layer",
                    )
                ),
            ),
            (
                outcome(prediction=row(visa_class="DIP-1")),
                resolved_case(self.FEE),
            ),
            (
                outcome(
                    prediction=row(visa_class="DIP-1"),
                    approval_facts=(),
                ),
                resolved_case(
                    self.FEE,
                    visible_field(
                        "visa_class",
                        "DIP-1",
                        evidence_type=EvidenceType.TEXT_LAYER,
                    ),
                ),
            ),
            (
                outcome(
                    approval_facts=("diplomatic_sponsor_exemption",),
                    prediction=row(visa_class="DIP-1", confidence=0.250001),
                ),
                resolved_case(self.FEE),
            ),
            (
                outcome(
                    approval_facts=("diplomatic_sponsor_exemption",),
                    denial_reasons=("policy_denial",),
                    prediction=row(visa_class="DIP-1"),
                ),
                resolved_case(self.FEE),
            ),
        )
        for original, resolved in variants:
            with self.subTest(
                confidence=original.row.confidence,
                fields=tuple(resolved.fields),
                denial_reasons=original.trace.denial_reasons,
            ):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_denial_recovery_has_priority_over_approval_recovery(self):
        sponsor = marker(
            "page_type_present_sponsor_attestation",
            "sponsor_attestation",
        )
        original = outcome(
            approval_facts=("diplomatic_sponsor_exemption",),
            prediction=row(
                arrival_date="2025-01-01",
                visa_class="DIP-1",
                confidence=0.25,
            ),
        )

        recovered, _baseline = recover(
            original,
            resolved_case(self.FEE, sponsor),
        )

        self.assertEqual(recovered.row.adjudication, "DENIED")
        self.assertIn(
            "review_denial_sponsor_stale_gt365",
            recovered.trace.denial_reasons,
        )


class ReviewApprovalRecoveryTests(unittest.TestCase):
    SPONSOR = marker(
        "page_type_present_sponsor_attestation",
        "sponsor_attestation",
    )
    OTHER = marker("page_type_present_other", "other")
    CURRENT_APPLICATION = "application_date_current_or_exempt"

    def assert_recovered(self, original, recovered, expected_fact):
        self.assertEqual(recovered.row.adjudication, "APPROVED")
        self.assertEqual(recovered.row.confidence, REVIEW_APPROVAL_CONFIDENCE)
        self.assertEqual(recovered.trace.decision, "APPROVED")
        self.assertFalse(recovered.trace.authoritative_source)
        self.assertEqual(recovered.trace.review_reasons, ())
        self.assertEqual(recovered.trace.denial_reasons, ())
        self.assertIn(expected_fact, recovered.trace.approval_facts)
        original_values = original.row.to_dict()
        recovered_values = recovered.row.to_dict()
        for field_name in original_values:
            if field_name not in {"adjudication", "confidence"}:
                self.assertEqual(
                    recovered_values[field_name],
                    original_values[field_name],
                    field_name,
                )

    def test_all_three_frozen_approval_rules_recover_at_boundaries(self):
        cases = (
            (
                outcome(prediction=row(visa_class="XW-1", confidence=0.20)),
                resolved_case(self.SPONSOR),
                "review_approval_sponsor_attestation_xw1",
            ),
            (
                outcome(
                    review_reasons=("visa_class_unknown",),
                    approval_facts=(self.CURRENT_APPLICATION,),
                    prediction=row(confidence=0.25),
                ),
                resolved_case(),
                "review_approval_current_application_visa_unknown",
            ),
            (
                outcome(
                    review_reasons=("required_output_unknown:home_world",),
                    approval_facts=(self.CURRENT_APPLICATION,),
                    prediction=row(confidence=0.35),
                ),
                resolved_case(),
                "review_approval_current_application_home_world_unknown",
            ),
        )
        for original, resolved, expected_fact in cases:
            with self.subTest(expected_fact=expected_fact):
                recovered, baseline = recover(original, resolved)
                self.assertEqual(baseline.calls, 1)
                self.assert_recovered(original, recovered, expected_fact)

    def test_current_home_world_unknown_rule_vetoes_unsupported_fee_waiver(self):
        blocked = outcome(
            review_reasons=(
                "required_output_unknown:home_world",
                "unsupported_fee_waiver",
            ),
            approval_facts=(self.CURRENT_APPLICATION,),
            prediction=row(confidence=0.35),
        )
        recovered, _baseline = recover(blocked, resolved_case())
        self.assertIs(recovered, blocked)

        boundary = outcome(
            review_reasons=("required_output_unknown:home_world",),
            approval_facts=(self.CURRENT_APPLICATION,),
            prediction=row(confidence=0.35),
        )
        recovered, _baseline = recover(boundary, resolved_case())
        self.assert_recovered(
            boundary,
            recovered,
            "review_approval_current_application_home_world_unknown",
        )

    def test_sponsor_xw1_rule_vetoes_each_missing_or_untrusted_feature(self):
        valid = outcome(prediction=row(visa_class="XW-1", confidence=0.20))
        variants = (
            (valid, resolved_case()),
            (
                outcome(prediction=row(visa_class="XW-2", confidence=0.20)),
                resolved_case(self.SPONSOR),
            ),
            (
                outcome(prediction=row(visa_class="XW-1", confidence=0.200001)),
                resolved_case(self.SPONSOR),
            ),
            (
                valid,
                resolved_case(
                    marker(
                        "page_type_present_sponsor_attestation",
                        "sponsor_attestation",
                        source="text_layer",
                    )
                ),
            ),
        )
        for original, resolved in variants:
            with self.subTest(
                visa_class=original.row.visa_class,
                confidence=original.row.confidence,
                fields=tuple(resolved.fields),
            ):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_current_visa_unknown_rule_enforces_fact_reason_and_open_lower_bound(self):
        variants = (
            outcome(
                review_reasons=("visa_class_unknown",),
                approval_facts=(),
                prediction=row(confidence=0.25),
            ),
            outcome(
                review_reasons=(),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.25),
            ),
            outcome(
                review_reasons=("visa_class_unknown",),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.20),
            ),
            outcome(
                review_reasons=("visa_class_unknown",),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.250001),
            ),
        )
        for original in variants:
            with self.subTest(
                confidence=original.row.confidence,
                review_reasons=original.trace.review_reasons,
                approval_facts=original.trace.approval_facts,
            ):
                recovered, _baseline = recover(original, resolved_case())
                self.assertIs(recovered, original)

    def test_current_home_world_rule_enforces_fact_reason_and_upper_bound(self):
        variants = (
            outcome(
                review_reasons=("required_output_unknown:home_world",),
                approval_facts=(),
                prediction=row(confidence=0.35),
            ),
            outcome(
                review_reasons=(),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.35),
            ),
            outcome(
                review_reasons=("required_output_unknown:home_world",),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.350001),
            ),
        )
        for original in variants:
            with self.subTest(
                confidence=original.row.confidence,
                review_reasons=original.trace.review_reasons,
                approval_facts=original.trace.approval_facts,
            ):
                recovered, _baseline = recover(original, resolved_case())
                self.assertIs(recovered, original)

    def test_denial_recovery_has_priority_over_every_new_approval_rule(self):
        cases = (
            outcome(
                review_reasons=("clean_biohazard_check_missing",),
                prediction=row(visa_class="XW-1", confidence=0.20),
            ),
            outcome(
                review_reasons=(
                    "clean_biohazard_check_missing",
                    "visa_class_unknown",
                ),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.25),
            ),
            outcome(
                review_reasons=(
                    "clean_biohazard_check_missing",
                    "required_output_unknown:home_world",
                ),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.35),
            ),
        )
        for original in cases:
            with self.subTest(
                confidence=original.row.confidence,
                review_reasons=original.trace.review_reasons,
            ):
                recovered, _baseline = recover(
                    original,
                    resolved_case(self.OTHER, self.SPONSOR),
                )
                self.assertEqual(recovered.row.adjudication, "DENIED")
                self.assertIn(
                    "review_denial_other_missing_biohazard",
                    recovered.trace.denial_reasons,
                )


class ReviewDenialRecoveryTests(unittest.TestCase):
    OTHER = marker("page_type_present_other", "other")
    SPONSOR = marker(
        "page_type_present_sponsor_attestation",
        "sponsor_attestation",
    )

    def assert_recovered(self, original, recovered, expected_reason):
        self.assertEqual(recovered.row.adjudication, "DENIED")
        self.assertEqual(recovered.row.confidence, REVIEW_DENIAL_CONFIDENCE)
        self.assertEqual(recovered.trace.decision, "DENIED")
        self.assertFalse(recovered.trace.authoritative_source)
        self.assertIn(expected_reason, recovered.trace.denial_reasons)
        original_values = original.row.to_dict()
        recovered_values = recovered.row.to_dict()
        for field_name in original_values:
            if field_name not in {"adjudication", "confidence"}:
                self.assertEqual(
                    recovered_values[field_name],
                    original_values[field_name],
                    field_name,
                )

    def test_all_three_frozen_rules_recover_review(self):
        cases = (
            (
                outcome(review_reasons=("clean_biohazard_check_missing",)),
                resolved_case(self.OTHER),
                "review_denial_other_missing_biohazard",
            ),
            (
                outcome(prediction=row(arrival_date="2025-01-01")),
                resolved_case(self.SPONSOR),
                "review_denial_sponsor_stale_gt365",
            ),
            (
                outcome(
                    review_reasons=(
                        "required_output_unknown:home_world",
                        "required_output_unknown:risk_flags",
                        "required_output_unknown:sponsor_id",
                    )
                ),
                resolved_case(),
                "review_denial_three_required_outputs_unknown",
            ),
        )
        for original, resolved, reason in cases:
            with self.subTest(reason=reason):
                recovered, baseline = recover(original, resolved)
                self.assertEqual(baseline.calls, 1)
                self.assert_recovered(original, recovered, reason)

    def test_rule_one_vetoes_missing_reason_marker_high_confidence_and_spoof(self):
        baseline = outcome(review_reasons=("clean_biohazard_check_missing",))
        variants = (
            (outcome(), resolved_case(self.OTHER)),
            (baseline, resolved_case()),
            (
                outcome(
                    review_reasons=("clean_biohazard_check_missing",),
                    prediction=row(confidence=0.350001),
                ),
                resolved_case(self.OTHER),
            ),
            (
                baseline,
                resolved_case(
                    marker(
                        "page_type_present_other",
                        "other",
                        source="text_layer",
                    )
                ),
            ),
        )
        for original, resolved in variants:
            with self.subTest(original=original, fields=tuple(resolved.fields)):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_rule_two_vetoes_missing_marker_current_arrival_and_high_confidence(self):
        stale = outcome(prediction=row(arrival_date="2025-01-01"))
        variants = (
            (stale, resolved_case()),
            (
                outcome(prediction=row(arrival_date="1900-01-01")),
                resolved_case(self.SPONSOR),
            ),
            (outcome(prediction=row(arrival_date="2026-01-01")), resolved_case(self.SPONSOR)),
            (
                outcome(
                    prediction=row(
                        arrival_date="2025-01-01",
                        confidence=0.350001,
                    )
                ),
                resolved_case(self.SPONSOR),
            ),
        )
        for original, resolved in variants:
            with self.subTest(arrival=original.row.arrival_date):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_rule_three_requires_each_of_its_three_reasons(self):
        required = (
            "required_output_unknown:home_world",
            "required_output_unknown:risk_flags",
            "required_output_unknown:sponsor_id",
        )
        for omitted in required:
            with self.subTest(omitted=omitted):
                original = outcome(
                    review_reasons=tuple(
                        reason for reason in required if reason != omitted
                    )
                )
                recovered, _baseline = recover(original, resolved_case())
                self.assertIs(recovered, original)

    def test_non_review_or_inconsistent_baseline_is_unchanged(self):
        triggers = resolved_case(self.OTHER)
        for decision in ("APPROVED", "DENIED"):
            original = outcome(
                review_reasons=("clean_biohazard_check_missing",),
                prediction=row(adjudication=decision),
            )
            recovered, _baseline = recover(original, triggers)
            self.assertIs(recovered, original)

        inconsistent = outcome(
            review_reasons=("clean_biohazard_check_missing",),
            trace_decision="APPROVED",
        )
        recovered, _baseline = recover(inconsistent, triggers)
        self.assertIs(recovered, inconsistent)

    def test_adjudicate_returns_schema_typed_row(self):
        original = outcome(review_reasons=("clean_biohazard_check_missing",))
        wrapper = ReviewDenialRecoveryAdjudicator(FakeAdjudicator(original))

        recovered = wrapper.adjudicate(resolved_case(self.OTHER))

        self.assertIsInstance(recovered, PredictionRow)
        self.assertEqual(tuple(recovered.to_dict()), tuple(original.row.to_dict()))
        self.assertEqual(recovered.confidence, 0.551819438046983)


if __name__ == "__main__":
    unittest.main()
