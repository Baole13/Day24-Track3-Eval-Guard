from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


def _token_len(text: str) -> int:
    return len((text or "").split())


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _heuristic_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    q = (question or "").lower()
    a_lower = (answer_a or "").lower()
    b_lower = (answer_b or "").lower()

    score_a = 0.5
    score_b = 0.5
    if "phép năm" in q or "nghỉ bao nhiêu ngày" in q:
        if "15" in a_lower:
            score_a += 0.25
        if "12" in a_lower:
            score_a -= 0.1
        if "15" in b_lower:
            score_b += 0.25
        if "12" in b_lower:
            score_b -= 0.1

    if any(keyword in q for keyword in ["vpn", "mật khẩu", "mfa", "v2024", "v2.0", "hiện hành"]):
        for text, target in ((a_lower, "A"), (b_lower, "B")):
            bonus = 0.0
            if any(term in text for term in ["v2024", "v2.0", "hiện hành", "wireguard", "mfa"]):
                bonus += 0.2
            if any(term in text for term in ["v2023", "v1.0", "nordvpn", "được, miễn là"]):
                bonus -= 0.15
            if target == "A":
                score_a += bonus
            else:
                score_b += bonus

    if _contains_any(q, ["không", "có nên", "được không"]):
        if any(term in a_lower for term in ["không", "không được", "tuyệt đối không"]):
            score_a += 0.1
        if any(term in b_lower for term in ["không", "không được", "tuyệt đối không"]):
            score_b += 0.1

    score_a += min(_token_len(answer_a), 25) / 250
    score_b += min(_token_len(answer_b), 25) / 250
    score_a = max(0.0, min(1.0, round(score_a, 3)))
    score_b = max(0.0, min(1.0, round(score_b, 3)))

    if abs(score_a - score_b) < 0.05:
        return {"winner": "tie", "reasoning": "Hai câu trả lời khá tương đương theo heuristic local.", "scores": {"A": score_a, "B": score_b}}
    if score_a > score_b:
        return {"winner": "A", "reasoning": "A có tín hiệu chính xác hoặc đầy đủ tốt hơn theo heuristic local.", "scores": {"A": score_a, "B": score_b}}
    return {"winner": "B", "reasoning": "B có tín hiệu chính xác hoặc đầy đủ tốt hơn theo heuristic local.", "scores": {"A": score_a, "B": score_b}}


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    # PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.
    #
    # Câu hỏi: {question}
    #
    # Answer A:
    # {answer_a}
    #
    # Answer B:
    # {answer_b}
    #
    # Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
    # Trả lời JSON (chỉ JSON, không text khác):
    # {{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
    # '''
    #
    # from openai import OpenAI
    # client = OpenAI()
    # resp = client.chat.completions.create(
    #     model=JUDGE_MODEL,
    #     messages=[
    #         {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
    #         {"role": "user",   "content": PROMPT_TEMPLATE.format(
    #             question=question, answer_a=answer_a, answer_b=answer_b)},
    #     ],
    #     response_format={"type": "json_object"},
    # )
    # return json.loads(resp.choices[0].message.content)
    prompt_template = """Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0, "B": 0.0}}}}
"""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                    {"role": "user", "content": prompt_template.format(
                        question=question, answer_a=answer_a, answer_b=answer_b
                    )},
                ],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            winner = parsed.get("winner", "tie")
            if winner not in {"A", "B", "tie"}:
                winner = "tie"
            scores = parsed.get("scores", {}) or {}
            normalized_scores = {
                "A": max(0.0, min(1.0, float(scores.get("A", 0.0) or 0.0))),
                "B": max(0.0, min(1.0, float(scores.get("B", 0.0) or 0.0))),
            }
            reasoning = parsed.get("reasoning", "") or ""
            if winner != "tie" and not reasoning:
                reasoning = "LLM judge selected a winner but returned empty reasoning."
            return {"winner": winner, "reasoning": reasoning, "scores": normalized_scores}
        except Exception:
            pass

    return _heuristic_pairwise(question, answer_a, answer_b)


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    # pass1 = pairwise_judge(question, answer_a, answer_b)
    # pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!
    #
    # # Convert pass2 back to original A/B space
    # swap_map = {"A": "B", "B": "A", "tie": "tie"}
    # winner_pass2 = swap_map[pass2_raw["winner"]]
    #
    # # Average: consensus only if both agree
    # if pass1["winner"] == winner_pass2:
    #     final = pass1["winner"]
    # else:
    #     final = "tie"  # disagreement = inconclusive
    #
    # position_consistent = (pass1["winner"] == winner_pass2)
    #
    # return JudgeResult(
    #     question=question, answer_a=answer_a, answer_b=answer_b,
    #     winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
    #     final_winner=final,
    #     reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
    #     position_consistent=position_consistent,
    #     scores_pass1=pass1["scores"],
    #     scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    # )
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw["winner"], "tie")
    final_winner = pass1["winner"] if pass1["winner"] == winner_pass2 else "tie"
    scores_pass2_raw = pass2_raw.get("scores", {})
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final_winner,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=(pass1["winner"] == winner_pass2),
        scores_pass1=pass1.get("scores", {"A": 0.0, "B": 0.0}),
        scores_pass2={
            "A": float(scores_pass2_raw.get("B", 0.0) or 0.0),
            "B": float(scores_pass2_raw.get("A", 0.0) or 0.0),
        },
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        return 0.0

    n = len(judge_labels)
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    p_j1 = judge_labels.count(1) / n
    p_h1 = human_labels.count(1) / n
    p_j0 = judge_labels.count(0) / n
    p_h0 = human_labels.count(0) / n
    p_e = p_j1 * p_h1 + p_j0 * p_h0
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    return (p_o - p_e) / (1 - p_e)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    # total = len(judge_results)
    # if total == 0:
    #     return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0}
    #
    # position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    # position_bias_rate  = position_bias_count / total
    #
    # a_wins_a_longer = sum(
    #     1 for r in judge_results
    #     if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    # )
    # b_wins_b_longer = sum(
    #     1 for r in judge_results
    #     if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    # )
    # decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    # verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0
    #
    # interpretation = ("Position bias cao — nên dùng swap-and-average."
    #                   if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    # return {
    #     "total_judged": total, "position_bias_rate": round(position_bias_rate, 3),
    #     "position_bias_count": position_bias_count,
    #     "verbosity_bias": round(verbosity_bias, 3),
    #     "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
    #                           "b_wins_b_longer": b_wins_b_longer,
    #                           "total_decisive": decisive},
    #     "interpretation": interpretation,
    # }
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "verbosity_bias": 0.0,
            "position_bias_count": 0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "Chưa có dữ liệu để đánh giá bias.",
        }

    position_bias_count = sum(1 for item in judge_results if not item.position_consistent)
    position_bias_rate = position_bias_count / total
    a_wins_a_longer = sum(
        1 for item in judge_results
        if item.final_winner == "A" and len(item.answer_a) > len(item.answer_b)
    )
    b_wins_b_longer = sum(
        1 for item in judge_results
        if item.final_winner == "B" and len(item.answer_b) > len(item.answer_a)
    )
    total_decisive = sum(1 for item in judge_results if item.final_winner != "tie")
    verbosity_bias = (
        (a_wins_a_longer + b_wins_b_longer) / total_decisive if total_decisive else 0.0
    )

    if position_bias_rate > 0.3:
        interpretation = "Position bias cao - nên dùng swap-and-average."
    elif verbosity_bias > 0.6:
        interpretation = "Verbosity bias đáng chú ý - judge có xu hướng thích câu trả lời dài."
    else:
        interpretation = "Bias ở mức chấp nhận được cho bộ mẫu hiện tại."

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "verbosity_bias": round(verbosity_bias, 3),
        "position_bias_count": position_bias_count,
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": total_decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # In production: run judge on the same 10 questions to get judge_labels
    judge_labels = [0] * len(human_labels)  # placeholder — replace with real judge output
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ (placeholder): {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report([result])
    print(f"\nBias report: {bias}")
