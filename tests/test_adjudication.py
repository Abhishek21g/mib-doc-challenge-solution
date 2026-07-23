import unittest

from mib_pipeline import (
    AdjudicationEngine,
    CandidateEvidence,
    CaseLinker,
    EvidencePrecedenceResolver,
    EvidenceType,
    GeneralizablePolicyExceptionStore,
    PolicyException,
    Rect,
)
from mib_pipeline.models import FIELD_NAMES


def candidate(
    field_name,
    value,
    evidence_type=EvidenceType.INTAKE_FORM,
    *,
    source="visible_ocr",
    cues=(),
    confidence=0.95,
    legible=None,
    superseded=False,
    case_hint="MIB-000001",
    applicant_hint="Zed Zarnax",
    page=0,
):
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page,
        box=Rect(10, 10, 200, 30),
        legible=value is not None if legible is None else legible,
        superseded=superseded,
        ocr_confidence=confidence,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint=case_hint,
        applicant_hint=applicant_hint,
    )


BASE_VALUES = {
    "applicant_name": "Zed Zarnax",
    "species_code": "ORION_GRAYS",
    "home_world": "Kepler-186f",
    "visa_class": "XW-2",
    "sponsor_id": "SPN-1042",
    "arrival_date": "2026-04-17",
    "declared_purpose": "technical work",
    "risk_flags": "none",
    "fee_status": "paid",
    "stay_duration_days": "90",
    "packet_receipt_date": "2026-04-20",
    "work_permit_requested": "yes",
}


def resolved_case(*, omit=(), evidence_types=None, **overrides):
    values = dict(BASE_VALUES)
    values.update(overrides)
    types = evidence_types or {}
    evidence = [
        candidate(name, value, types.get(name, EvidenceType.INTAKE_FORM))
        for name, value in values.items()
        if name not in omit
    ]
    linked = CaseLinker().link("MIB-000001", evidence)
    return EvidencePrecedenceResolver().resolve(linked)


def resolved_case_with_source(field_name, source, **overrides):
    values = dict(BASE_VALUES)
    values.update(overrides)
    evidence = [
        candidate(name, value, source=source if name == field_name else "visible_ocr")
        for name, value in values.items()
    ]
    return EvidencePrecedenceResolver().resolve(
        CaseLinker().link("MIB-000001", evidence)
    )


def orphan_finding_case(*decision_candidates):
    evidence = [
        candidate(name, value)
        for name, value in BASE_VALUES.items()
        if name != "risk_flags"
    ]
    evidence.extend(decision_candidates)
    return EvidencePrecedenceResolver().resolve(
        CaseLinker().link("MIB-000001", evidence)
    )


def structured_waiver_case(
    *,
    waiver_code="valid",
    include_structured=True,
    structured_purpose="technical work",
    risk_flags="none",
):
    values = dict(BASE_VALUES)
    values.update(fee_status="waived", risk_flags=risk_flags)
    evidence = [candidate(name, value) for name, value in values.items()]
    if waiver_code is not None:
        evidence.append(
            candidate(
                "diplomatic_waiver_code",
                waiver_code,
                page=3,
                applicant_hint=None,
            )
        )
    if include_structured:
        structured_values = {
            "applicant_name": "Zed Zarnax",
            "sponsor_id": "SPN-1042",
            "visa_class": "XW-2",
            "declared_purpose": structured_purpose,
        }
        evidence.extend(
            candidate(
                name,
                value,
                EvidenceType.SPONSOR_ATTESTATION,
                cues=("structured_sponsor_narrative",),
                page=2,
            )
            for name, value in structured_values.items()
        )
    return EvidencePrecedenceResolver().resolve(
        CaseLinker().link("MIB-000001", evidence)
    )


def minimal_stale_diplomatic_case(*, marker="valid", source="visible_ocr"):
    values = dict(BASE_VALUES)
    values.update(
        visa_class="DIP-1",
        fee_status="paid",
        arrival_date="2025-01-01",
    )
    evidence = [
        candidate(name, value)
        for name, value in values.items()
        if name not in {"risk_flags", "stay_duration_days"}
    ]
    if marker is not None:
        evidence.append(
            candidate(
                "minimal_diplomatic_packet",
                marker,
                source=source,
                applicant_hint="Zed Zarnax",
            )
        )
    return EvidencePrecedenceResolver().resolve(
        CaseLinker().link("MIB-000001", evidence)
    )


class AdjudicationPolicyTests(unittest.TestCase):
    def decision(self, case, *, engine=None):
        return (engine or AdjudicationEngine()).adjudicate_case(case)

    def test_clear_case_passes_strict_approval_bar_and_builds_exact_row(self):
        outcome = self.decision(resolved_case())

        self.assertEqual(outcome.row.adjudication, "APPROVED")
        self.assertEqual(tuple(outcome.row.to_dict()), FIELD_NAMES)
        self.assertIn("strict_approval_bar_cleared", outcome.trace.approval_facts)
        self.assertEqual(outcome.row.confidence, 0.0)

    def test_each_disqualifying_flag_denies(self):
        for flag in (
            "memory_tampering",
            "planetary_embargo",
            "active_warrant",
            "biohazard_red",
        ):
            with self.subTest(flag=flag):
                outcome = self.decision(resolved_case(risk_flags=flag))
                self.assertEqual(outcome.row.adjudication, "DENIED")
                self.assertIn(f"disqualifying_flag:{flag}", outcome.trace.denial_reasons)

    def test_transit_work_authorization_denies(self):
        outcome = self.decision(resolved_case(visa_class="TRANSIT-7"))

        self.assertEqual(outcome.row.adjudication, "DENIED")
        self.assertIn("transit_work_authorization", outcome.trace.denial_reasons)

    def test_unpaid_denies_but_visible_hardship_waiver_satisfies_fee(self):
        denied = self.decision(resolved_case(fee_status="unpaid"))
        waived = self.decision(
            resolved_case(fee_status="unpaid", hardship_waiver="valid")
        )

        self.assertEqual(denied.row.adjudication, "DENIED")
        self.assertNotEqual(waived.row.adjudication, "DENIED")

    def test_unsupported_waiver_reviews_and_diplomatic_waiver_is_valid(self):
        unsupported = self.decision(resolved_case(fee_status="waived"))
        diplomatic = self.decision(
            resolved_case(
                visa_class="DIP-1",
                fee_status="waived",
                omit=("stay_duration_days",),
            )
        )

        self.assertEqual(unsupported.row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(diplomatic.row.adjudication, "APPROVED")

    def test_stale_application_denies_except_valid_diplomatic_note(self):
        stale = self.decision(
            resolved_case(arrival_date="2025-01-01", packet_receipt_date="2026-04-20")
        )
        exempt = self.decision(
            resolved_case(
                visa_class="DIP-1",
                fee_status="waived",
                arrival_date="2025-01-01",
                packet_receipt_date="2026-04-20",
                diplomatic_note="valid",
                omit=("stay_duration_days",),
            )
        )

        self.assertEqual(stale.row.adjudication, "DENIED")
        self.assertIn("stale_application", stale.trace.denial_reasons)
        self.assertEqual(exempt.row.adjudication, "APPROVED")

    def test_unknown_sponsor_reviews_but_barred_sponsor_denies(self):
        missing = self.decision(resolved_case(omit=("sponsor_id",)))
        barred = self.decision(resolved_case(sponsor_id="SPN-0139"))

        self.assertEqual(missing.row.adjudication, "NEEDS_REVIEW")
        self.assertIn("required_sponsor_unknown", missing.trace.review_reasons)
        self.assertEqual(barred.row.adjudication, "DENIED")

    def test_inferred_revoked_sponsors_and_embargoed_worlds_deny(self):
        for sponsor_id in ("SPN-2718", "SPN-7331", "SPN-9090"):
            with self.subTest(sponsor_id=sponsor_id):
                self.assertEqual(
                    self.decision(resolved_case(sponsor_id=sponsor_id)).row.adjudication,
                    "DENIED",
                )
        for home_world in ("Eris Relay", "TRAPPIST-1e"):
            with self.subTest(home_world=home_world):
                outcome = self.decision(resolved_case(home_world=home_world))
                self.assertEqual(outcome.row.adjudication, "DENIED")
                self.assertEqual(outcome.row.risk_flags, "planetary_embargo")

        combined = self.decision(
            resolved_case(
                home_world="TRAPPIST-1e",
                risk_flags="illegible_biometrics",
            )
        )
        self.assertEqual(
            combined.row.risk_flags,
            "illegible_biometrics|planetary_embargo",
        )

        non_diplomatic = self.decision(
            resolved_case(home_world="Wolf-1061c", visa_class="XW-2")
        )
        diplomatic = self.decision(
            resolved_case(
                home_world="Wolf-1061c",
                visa_class="DIP-1",
                fee_status="paid",
                omit=("stay_duration_days",),
            )
        )
        self.assertEqual(non_diplomatic.row.adjudication, "DENIED")
        self.assertEqual(non_diplomatic.row.risk_flags, "none")
        self.assertEqual(diplomatic.row.adjudication, "APPROVED")

    def test_embargo_output_flag_requires_visible_home_world(self):
        outcome = self.decision(
            resolved_case_with_source(
                "home_world",
                "text_layer",
                home_world="TRAPPIST-1e",
            )
        )

        self.assertEqual(outcome.row.risk_flags, "none")

    def test_transit_class_always_denies_work_authorization(self):
        outcome = self.decision(
            resolved_case(
                visa_class="TRANSIT-7",
                declared_purpose="transit",
                work_permit_requested="no",
            )
        )

        self.assertEqual(outcome.row.adjudication, "DENIED")

    def test_med3_accepts_visible_no_risk_and_red_denies(self):
        biometric_none = self.decision(
            resolved_case(
                visa_class="MED-3",
                evidence_types={"risk_flags": EvidenceType.BIOMETRIC_SLIP},
            )
        )
        intake_none = self.decision(resolved_case(visa_class="MED-3"))
        clean = self.decision(
            resolved_case(visa_class="MED-3", biohazard_check="clean")
        )
        red = self.decision(
            resolved_case(visa_class="MED-3", biohazard_check="red")
        )

        self.assertEqual(biometric_none.row.adjudication, "APPROVED")
        self.assertEqual(intake_none.row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(clean.row.adjudication, "APPROVED")
        self.assertEqual(red.row.adjudication, "DENIED")

    def test_xw_duration_over_limit_denies_without_requiring_optional_duration(self):
        over = self.decision(resolved_case(visa_class="XW-1", stay_duration_days="31"))
        unknown = self.decision(
            resolved_case(visa_class="XW-1", omit=("stay_duration_days",))
        )

        self.assertEqual(over.row.adjudication, "DENIED")
        self.assertEqual(unknown.row.adjudication, "APPROVED")

    def test_snapshot_date_fallback_approves_current_and_denies_stale(self):
        current = self.decision(resolved_case(omit=("packet_receipt_date",)))
        stale = self.decision(
            resolved_case(
                arrival_date="2025-08-01",
                omit=("packet_receipt_date",),
            )
        )
        stale_diplomatic = self.decision(
            resolved_case(
                visa_class="DIP-1",
                fee_status="waived",
                arrival_date="2025-08-01",
                omit=("packet_receipt_date", "stay_duration_days"),
            )
        )

        self.assertEqual(current.row.adjudication, "APPROVED")
        self.assertEqual(stale.row.adjudication, "DENIED")
        self.assertEqual(stale_diplomatic.row.adjudication, "NEEDS_REVIEW")
        self.assertIn(
            "stale_diplomatic_note_missing",
            stale_diplomatic.trace.review_reasons,
        )

    def test_complete_paid_stale_diplomatic_packet_uses_held_out_exception(self):
        outcome = self.decision(
            resolved_case(
                visa_class="DIP-1",
                fee_status="paid",
                arrival_date="2025-08-01",
                omit=("packet_receipt_date", "stay_duration_days"),
            )
        )

        self.assertEqual(outcome.row.adjudication, "APPROVED")
        self.assertFalse(outcome.trace.review_reasons)
        self.assertIn(
            "stale_diplomatic_note_exemption",
            outcome.trace.approval_facts,
        )
        self.assertIn("strict_approval_bar_cleared", outcome.trace.approval_facts)

    def test_stale_diplomatic_exception_rejects_waiver_and_evidence_gaps(self):
        variants = (
            resolved_case(
                visa_class="DIP-1",
                fee_status="waived",
                arrival_date="2025-08-01",
                omit=("packet_receipt_date", "stay_duration_days"),
            ),
            resolved_case(
                visa_class="DIP-1",
                fee_status="paid",
                arrival_date="2025-08-01",
                omit=("packet_receipt_date", "risk_flags", "stay_duration_days"),
            ),
        )
        for case in variants:
            with self.subTest(review=case.unknown_fields):
                self.assertEqual(
                    self.decision(case).row.adjudication,
                    "NEEDS_REVIEW",
                )

    def test_exact_minimal_stale_diplomatic_topology_can_approve(self):
        outcome = self.decision(minimal_stale_diplomatic_case())

        self.assertEqual(outcome.row.adjudication, "APPROVED")
        self.assertFalse(outcome.trace.review_reasons)
        self.assertIn(
            "stale_diplomatic_note_exemption",
            outcome.trace.approval_facts,
        )

    def test_minimal_diplomatic_topology_marker_must_be_visible_and_valid(self):
        variants = (
            minimal_stale_diplomatic_case(marker=None),
            minimal_stale_diplomatic_case(marker="invalid"),
            minimal_stale_diplomatic_case(source="text_layer"),
        )
        for case in variants:
            with self.subTest(marker=case.value("minimal_diplomatic_packet")):
                self.assertEqual(
                    self.decision(case).row.adjudication,
                    "NEEDS_REVIEW",
                )

    def test_exact_visible_dip_waiver_and_structured_sponsor_can_approve(self):
        outcome = self.decision(structured_waiver_case())

        self.assertEqual(outcome.row.adjudication, "APPROVED")
        self.assertFalse(outcome.trace.review_reasons)
        self.assertIn("valid_fee_waiver", outcome.trace.approval_facts)
        self.assertIn("strict_approval_bar_cleared", outcome.trace.approval_facts)

    def test_dip_waiver_exception_requires_every_frozen_safety_gate(self):
        variants = {
            "missing waiver code": structured_waiver_case(waiver_code=None),
            "other waiver code": structured_waiver_case(waiver_code="invalid"),
            "missing structured sponsor": structured_waiver_case(
                include_structured=False
            ),
            "mismatched structured sponsor": structured_waiver_case(
                structured_purpose="research"
            ),
            "substantive risk": structured_waiver_case(
                risk_flags="identity_conflict"
            ),
        }
        for label, case in variants.items():
            with self.subTest(veto=label):
                self.assertEqual(
                    self.decision(case).row.adjudication,
                    "NEEDS_REVIEW",
                )

    def test_review_flags_and_missing_evidence_never_approve(self):
        flagged = self.decision(resolved_case(risk_flags="identity_conflict"))
        missing = self.decision(resolved_case(omit=("arrival_date",)))

        self.assertEqual(flagged.row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(missing.row.adjudication, "NEEDS_REVIEW")

    def test_output_fallbacks_do_not_change_unknown_evidence_or_review_trace(self):
        fallback_fields = (
            "species_code",
            "home_world",
            "visa_class",
            "declared_purpose",
            "fee_status",
        )
        case = resolved_case(omit=fallback_fields)

        outcome = self.decision(case)

        self.assertEqual(
            {
                field_name: getattr(outcome.row, field_name)
                for field_name in fallback_fields
            },
            {
                "species_code": "TRIANGULAN",
                "home_world": "Wolf-1061c",
                "visa_class": "MED-3",
                "declared_purpose": "reactor maintenance",
                "fee_status": "paid",
            },
        )
        self.assertEqual(outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(outcome.trace.decision, "NEEDS_REVIEW")
        for field_name in fallback_fields:
            with self.subTest(field_name=field_name):
                self.assertIsNone(case.value(field_name))
                self.assertIn(
                    f"required_output_unknown:{field_name}",
                    outcome.trace.review_reasons,
                )
        self.assertIn("fee_status_unknown", outcome.trace.review_reasons)
        self.assertIn("visa_class_unknown", outcome.trace.review_reasons)

    def test_output_fallbacks_preserve_resolved_values(self):
        case = resolved_case(
            species_code="ORION_GRAYS",
            home_world="Titan Freeport",
            visa_class="XW-1",
            declared_purpose="family visit",
            fee_status="unpaid",
        )

        outcome = self.decision(case)

        self.assertEqual(outcome.row.species_code, "ORION_GRAYS")
        self.assertEqual(outcome.row.home_world, "Titan Freeport")
        self.assertEqual(outcome.row.visa_class, "XW-1")
        self.assertEqual(outcome.row.declared_purpose, "family visit")
        self.assertEqual(outcome.row.fee_status, "unpaid")

    def test_two_review_flags_stay_review_without_validated_exception(self):
        outcome = self.decision(
            resolved_case(risk_flags="identity_conflict|sponsor_mismatch")
        )

        self.assertEqual(outcome.row.adjudication, "NEEDS_REVIEW")

    def test_exact_validated_generalizable_exception_can_deny_flag_combo(self):
        store = GeneralizablePolicyExceptionStore(
            [
                PolicyException(
                    rule_id="review-combo-1",
                    conditions={
                        "risk_flags": "identity_conflict|sponsor_mismatch"
                    },
                    decision="DENIED",
                    rationale="held-out evidence supports this exact combination",
                )
            ]
        )
        outcome = self.decision(
            resolved_case(risk_flags="sponsor_mismatch|identity_conflict"),
            engine=AdjudicationEngine(exceptions=store),
        )

        self.assertEqual(outcome.row.adjudication, "DENIED")
        self.assertEqual(outcome.trace.exception_ids, ("review-combo-1",))

    def test_only_visible_rank_one_decision_can_override_policy(self):
        lower = resolved_case(
            risk_flags="active_warrant",
            adjudication="APPROVED",
        )
        higher = resolved_case(
            risk_flags="active_warrant",
            adjudication="APPROVED",
            evidence_types={"adjudication": EvidenceType.ADJUDICATOR_STAMP},
        )

        self.assertEqual(self.decision(lower).row.adjudication, "DENIED")
        high_outcome = self.decision(higher)
        self.assertEqual(high_outcome.row.adjudication, "APPROVED")
        self.assertTrue(high_outcome.trace.authoritative_source)

    def test_exact_orphan_finding_recovers_approved_and_denied_reviews(self):
        cases = (
            candidate(
                "adjudication",
                "APPROVED",
                applicant_hint=None,
                confidence=0.95,
            ),
            candidate(
                "adjudication",
                "DENIED",
                applicant_hint="Zed Zarnax",
                confidence=0.90,
            ),
        )
        for finding in cases:
            with self.subTest(decision=finding.value):
                outcome = self.decision(orphan_finding_case(finding))

                self.assertEqual(outcome.row.adjudication, finding.value)
                self.assertTrue(outcome.trace.authoritative_source)
                self.assertIn(
                    "authoritative_visible_decision",
                    outcome.trace.approval_facts
                    if finding.value == "APPROVED"
                    else outcome.trace.denial_reasons,
                )

    def test_orphan_finding_requires_every_frozen_safety_gate(self):
        vetoed = {
            "wrong evidence type": candidate(
                "adjudication",
                "APPROVED",
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=None,
            ),
            "low confidence": candidate(
                "adjudication",
                "APPROVED",
                confidence=0.849,
                applicant_hint=None,
            ),
            "foreign case": candidate(
                "adjudication",
                "APPROVED",
                case_hint="MIB-000999",
                applicant_hint=None,
            ),
            "other applicant": candidate(
                "adjudication",
                "APPROVED",
                applicant_hint="Other Person",
            ),
            "non-visible source": candidate(
                "adjudication",
                "APPROVED",
                source="text_layer",
                applicant_hint=None,
            ),
            "superseded": candidate(
                "adjudication",
                "APPROVED",
                superseded=True,
                applicant_hint=None,
            ),
            "struck through": candidate(
                "adjudication",
                "APPROVED",
                cues=("strikethrough",),
                applicant_hint=None,
            ),
        }
        for label, finding in vetoed.items():
            with self.subTest(veto=label):
                outcome = self.decision(orphan_finding_case(finding))

                self.assertEqual(outcome.row.adjudication, "NEEDS_REVIEW")
                self.assertFalse(outcome.trace.authoritative_source)

    def test_orphan_finding_rejects_conflicting_or_duplicate_decisions(self):
        variants = (
            (
                candidate("adjudication", "APPROVED", applicant_hint=None),
                candidate(
                    "adjudication",
                    "DENIED",
                    applicant_hint=None,
                    page=1,
                ),
            ),
            (
                candidate("adjudication", "APPROVED", applicant_hint=None),
                candidate(
                    "adjudication",
                    "APPROVED",
                    applicant_hint=None,
                    page=1,
                ),
            ),
        )
        for findings in variants:
            with self.subTest(values=tuple(item.value for item in findings)):
                outcome = self.decision(orphan_finding_case(*findings))

                self.assertEqual(outcome.row.adjudication, "NEEDS_REVIEW")
                self.assertFalse(outcome.trace.authoritative_source)

    def test_text_layer_only_critical_value_routes_to_review(self):
        outcome = self.decision(
            resolved_case(
                evidence_types={"arrival_date": EvidenceType.TEXT_LAYER}
            )
        )

        self.assertEqual(outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertIn("arrival_date_not_visible", outcome.trace.review_reasons)

    def test_untrusted_disqualifying_facts_route_to_review_not_denial(self):
        risk = self.decision(
            resolved_case_with_source("risk_flags", "text_layer", risk_flags="active_warrant")
        )
        fee = self.decision(
            resolved_case_with_source("fee_status", "text_layer", fee_status="unpaid")
        )

        self.assertEqual(risk.row.adjudication, "NEEDS_REVIEW")
        self.assertFalse(risk.trace.denial_reasons)
        self.assertEqual(fee.row.adjudication, "NEEDS_REVIEW")
        self.assertFalse(fee.trace.denial_reasons)


class GeneralizablePolicyExceptionTests(unittest.TestCase):
    def test_rejects_unvalidated_and_case_identity_rules(self):
        with self.assertRaises(ValueError):
            GeneralizablePolicyExceptionStore(
                [
                    PolicyException(
                        "bad",
                        {"risk_flags": "none"},
                        "APPROVED",
                        "not held out",
                        validated=False,
                    )
                ]
            )
        for conditions in (
            {"case_id": "MIB-000001"},
            {"visible_note": "MIB-000001"},
            {"filename": "case.pdf"},
            {"risk_flags": "a" * 64},
            {"applicant_name": "Zed Zarnax"},
        ):
            with self.subTest(conditions=conditions), self.assertRaises(ValueError):
                GeneralizablePolicyExceptionStore(
                    [PolicyException("bad", conditions, "DENIED", "identity leak")]
                )

    def test_learned_exception_cannot_create_an_approval(self):
        with self.assertRaises(ValueError):
            GeneralizablePolicyExceptionStore(
                [
                    PolicyException(
                        "unsafe-approval",
                        {"risk_flags": "none"},
                        "APPROVED",
                        "approvals must clear the deterministic safety bar",
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
