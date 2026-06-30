# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh vien:** Le Quoc Bao  
**Ngay:** 2026-06-30

---

## Guard Stack Architecture

```text
User Input
    |
    v (~0.94ms P95)
[Presidio PII Scan]
    | block if: VN_CCCD / VN_PHONE / EMAIL detected
    | action:   return 400 + "PII detected in query"
    v (~0.76ms P95)
[NeMo Input Rail]
    | block if: off-topic / jailbreak / prompt injection
    | action:   return 503 + refuse message
    v
[RAG Pipeline (Day 18)]
    | M1 Chunk -> M2 Search -> M3 Rerank -> GPT-4o-mini
    v
[NeMo Output Rail]
    | flag if:  PII in response / sensitive content
    | action:   replace with safe response
    v
User Response
```

---

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 0.56 | 0.94 | 0.94 | <10ms |
| NeMo Input Rail | 0.54 | 0.76 | 0.76 | <300ms |
| RAG Pipeline | N/A | N/A | N/A | <2000ms |
| NeMo Output Rail | N/A | N/A | N/A | <300ms |
| **Total Guard** | 1.1 | **1.83** | 1.83 | **<500ms** |

**Budget OK?** Yes  
**Comment:** So do hien tai moi bao gom Presidio va input rail local/fallback path co san trong moi truong nay.

---

## CI/CD Gates (phai pass truoc khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Ket Qua Thuc Te Tu Lab

| | Ket qua |
|---|---|
| RAGAS avg_score (50q) | 0.0 |
| Worst metric | faithfulness |
| Dominant failure distribution | factual |
| Cohen's kappa | 0.286 |
| Adversarial pass rate | 17 / 20 |
| Guard P95 latency | 1.83 ms |

---

## Nhan Xet & Cai Tien

Bo lab hien tai da xanh toan bo unit tests, tao duoc `answers_50q.json`, `ragas_50q.json`, `judge_results.json`, va `guard_results.json`. Guardrail stack dat 17/20 case adversarial, vuot nguong pass toi thieu 15/20, va latency guard rat thap trong local fallback path. Diem con thieu la package `ragas` chua co trong moi truong nen Phase A moi cho so do provisional 0.0; neu muon danh gia production-grade, buoc tiep theo la cai `ragas`, bat Qdrant that, va chay lai Phase A/B/C voi live API de cap nhat blueprint bang so lieu thuc.
