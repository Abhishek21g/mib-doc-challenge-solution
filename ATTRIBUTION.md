# Attribution

This solution repository is derived from Chris Strobl's public MIT-licensed
[`strobl/mib-doc-solution`](https://github.com/strobl/mib-doc-solution)
(commit corresponding to the ~130.37 public-train release), including its
render-first OCR stack, RapidOCR recovery, and identity-free review heads.

We retain upstream notices under `third_party_licenses/` and the MIT license.

## Our policy constraints (stricter than some public forks)

- No hardcoded case IDs or ground-truth answer tables
- No reading `SYSTEM:` / white answer-key decoy text as features
- Offline Docker-legal only (no network / LLM / VLM at score time)

Competing forks that enable answer-key field transcription (e.g. arjun v27
claimed ~132.5) are out of scope for this submission. Layout-consensus
approval heads from those forks were measured net-negative on a held slice
when answer keys were disabled, so they are not shipped.

## Additional legal ports

- `apply_visible_field_repairs` from arjunkshah12345-hash’s public MIT fork of
  strobl (`arjun_heads.py`): Amount/$809 + DIP-WAIVER fee cues, registry name
  precedence, sponsor visa/arrival/purpose repairs from **visible** layout text
  only. Layout-consensus approval and answer-key transcription are **not** used
  (measured net-negative / policy-forbidden respectively).
