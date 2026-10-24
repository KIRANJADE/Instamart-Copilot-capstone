import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm


TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 6000

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.

Allowed categories: billing, delivery, technical, fraud.

Return JSON only, using this format:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not include Markdown or any text outside the JSON.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.

Allowed categories: billing, delivery, technical, fraud.

Return JSON only, using this format:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not include Markdown or any text outside the JSON.

EXAMPLES:
Example 1:
Ticket: My payment was deducted twice.
Answer: {"category": "billing", "reason": "Duplicate payment"}

Example 2:
Ticket: My account was hacked and I see an unauthorized transaction.
Answer: {"category": "fraud", "reason": "Unauthorized account activity"}

Example 3:
Ticket: The application crashes when I log in.
Answer: {"category": "technical", "reason": "Application crash"}

Example 4:
Ticket: My delivery arrived late.
Answer: {"category": "delivery", "reason": "Late delivery"}
"""


# --------------------------------------------------
# 2. Redact personal information
# --------------------------------------------------

def redact_text(text):
    """Apply redactions in the exact required order."""
    text = str(text)

    text, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phone_count = PHONE_PATTERN.subn("[PHONE]", text)
    text, order_count = ORDER_PATTERN.subn("[ORDER_ID]", text)

    counts = {
        "emails": email_count,
        "phones": phone_count,
        "order_ids": order_count,
    }

    return text, counts


def contains_pii(text):
    """Check for any remaining sensitive patterns."""
    return bool(
        EMAIL_PATTERN.search(text)
        or PHONE_PATTERN.search(text)
        or ORDER_PATTERN.search(text)
    )


# --------------------------------------------------
# 3. Validate model responses
# --------------------------------------------------

def validate_response(content):
    """
    Return (category, status).

    Valid JSON with an allowed category:
        classified_json

    Non-JSON response with exactly one allowed category word:
        classified_keyword

    Anything else:
        flagged_invalid_output
    """
    try:
        data = json.loads(content)

        # Valid JSON must contain an allowed category.
        # Invalid JSON categories must not be guessed.
        if (
            isinstance(data, dict)
            and data.get("category") in CATEGORIES
        ):
            return data["category"], "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        matches = [word for word in words if word in CATEGORIES]

        if len(matches) == 1:
            return matches[0], "classified_keyword"

        return "", "flagged_invalid_output"


# --------------------------------------------------
# 4. Retry temporary model failures
# --------------------------------------------------

def call_with_retry(messages, ticket_id):
    """Retry transient failures up to three times."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"[RETRY] ticket={ticket_id} "
                f"retry={retry_number} wait={wait:.4f}s"
            )

            time.sleep(wait)

    raise RuntimeError(f"Retries exhausted for ticket {ticket_id}")


# --------------------------------------------------
# 5. Classify tickets
# --------------------------------------------------

def run_pipeline(tickets, system_prompt, token_budget=None):
    """
    Process tickets in file order.

    A token budget is checked before each model call.
    Only successful responses add their usage to the
    running token total.
    """
    # Required immediately before each independent full run.
    mock_llm.reset()

    results = []
    total_tokens = 0
    prompt_calls = 0

    for _, ticket in tickets.iterrows():
        ticket_id = str(ticket["ticket_id"])
        customer_id = str(ticket["customer_id"])
        original_text = str(ticket["text"])

        result = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": "",
            "category": "",
            "true_category": str(ticket["true_category"]).strip().lower(),
            "tokens_used": 0,
        }

        # Guardrail 1: empty ticket.
        if not original_text.strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue

        # Guardrail 2: excessively long ticket.
        if mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue

        # Guardrail 3: stop before a call if budget is exhausted.
        if (
            token_budget is not None
            and total_tokens >= token_budget
        ):
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        redacted_text, _ = redact_text(original_text)

        # Ensure no PII remains in the user message.
        assert not contains_pii(redacted_text), (
            f"PII detected in ticket {ticket_id}"
        )

        # The user message must contain only the redacted ticket text.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text},
        ]

        assert messages[-1]["content"] == redacted_text
        assert not contains_pii(messages[-1]["content"])

        response = call_with_retry(messages, ticket_id)
        prompt_calls += 1

        usage_tokens = response.usage.total_tokens
        total_tokens += usage_tokens
        result["tokens_used"] = usage_tokens

        content = response.choices[0].message.content or ""
        category, status = validate_response(content)

        result["category"] = category
        result["status"] = status

        results.append(result)

    results_df = pd.DataFrame(results)

    classified_statuses = {
        "classified_json",
        "classified_keyword",
    }

    classified = results_df[
        results_df["status"].isin(classified_statuses)
    ]

    if len(classified):
        accuracy = (
            classified["category"] == classified["true_category"]
        ).mean()
    else:
        accuracy = 0.0

    report = {
        "tickets_total": len(results_df),
        "model_calls": prompt_calls,
        "tickets_classified": len(classified),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": (
            total_tokens / len(classified) if len(classified) else 0
        ),
        "status_counts": results_df["status"].value_counts().to_dict(),
    }

    return results_df, report


# --------------------------------------------------
# 6. Routing using Part B customer segments
# --------------------------------------------------

def load_customer_segments():
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SEGMENTS_FILE}. Run Part B first."
        )

    segments = pd.read_csv(
        SEGMENTS_FILE,
        dtype={"customer_id": str},
        keep_default_na=False,
    )

    required = {"customer_id", "segment_name", "is_premium"}
    missing = required - set(segments.columns)

    if missing:
        raise ValueError(
            f"customer_segments.csv is missing columns: {sorted(missing)}"
        )

    segments = segments.drop_duplicates(
        subset=["customer_id"], keep="last"
    )

    def to_bool(value):
        return str(value).strip().lower() in {
            "true", "1", "yes"
        }

    segments["is_premium"] = segments["is_premium"].map(to_bool)

    return segments[["customer_id", "segment_name", "is_premium"]]


def apply_routing(results_df, segments):
    merged = results_df.merge(
        segments,
        on="customer_id",
        how="left",
    )

    merged["segment_name"] = merged["segment_name"].fillna("")
    merged["is_premium"] = merged["is_premium"].fillna(False)

    human_review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    routes = []
    priorities = []

    for _, row in merged.iterrows():
        if row["status"] in human_review_statuses:
            route = "human_review"
            priority = "NORMAL"

        elif row["category"] == "fraud":
            route = "security_team"
            priority = "HIGH"

        elif bool(row["is_premium"]):
            route = "premium_desk"
            priority = "HIGH"

        else:
            route = "support_queue"
            priority = "NORMAL"

        routes.append(route)
        priorities.append(priority)

    merged["route"] = routes
    merged["priority"] = priorities

    # Required output columns, in the specified order.
    return merged[
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ]
    ]


# --------------------------------------------------
# 7. Reports and confusion matrix
# --------------------------------------------------

def print_report(name, report):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)
    print(f"Tickets: {report['tickets_total']}")
    print(f"Model calls: {report['model_calls']}")
    print(f"Tickets classified: {report['tickets_classified']}")
    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Total tokens: {report['total_tokens']}")
    print(
        "Tokens per classified ticket: "
        f"{report['tokens_per_classified_ticket']:.2f}"
    )
    print("Status counts:", report["status_counts"])


def print_zero_shot_confusion(results_df):
    classified_statuses = {
        "classified_json",
        "classified_keyword",
    }

    classified = results_df[
        results_df["status"].isin(classified_statuses)
    ]

    print("\nZero-shot confusion table")
    print(
        pd.crosstab(
            classified["true_category"],
            classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
            dropna=False,
        ).to_string()
    )


# --------------------------------------------------
# 8. Main execution
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TICKETS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TICKETS_FILE}. Run generate_data.py first."
        )

    # keep_default_na=False preserves empty ticket text.
    tickets = pd.read_csv(
        TICKETS_FILE,
        dtype={"ticket_id": str, "customer_id": str},
        keep_default_na=False,
    )

    print(f"Loaded tickets: {len(tickets)}")

    # Redaction audit across all tickets, including empty and long ones.
    redaction_totals = {
        "emails": 0,
        "phones": 0,
        "order_ids": 0,
    }

    redacted_texts = []

    for text in tickets["text"]:
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)

        for key in redaction_totals:
            redaction_totals[key] += counts[key]

        # Reapplying the patterns must find no remaining PII.
        assert not contains_pii(redacted)

    print("\nPII redaction audit")
    for key, count in redaction_totals.items():
        print(f"{key}: {count}")

    print(
        "Total substitutions:",
        sum(redaction_totals.values()),
    )
    print("Remaining PII matches: 0")

    # Full zero-shot run, with no token budget.
    zero_results, zero_report = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        token_budget=None,
    )
    print_report("ZERO-SHOT RESULTS", zero_report)
    print_zero_shot_confusion(zero_results)

    # Full few-shot run, with no token budget.
    few_results, few_report = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        token_budget=None,
    )
    print_report("FEW-SHOT RESULTS", few_report)

    # Zero-shot budget demonstration.
    demo_budget = TOKEN_BUDGET
    budget_results, budget_report = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        token_budget=demo_budget,
    )

    # Lower the budget if the first run did not exhaust it.
    if "budget_exhausted" not in budget_report["status_counts"]:
        demo_budget = 5000
        print(
            "\nNo tickets exhausted the 6000-token budget. "
            "Retrying the demonstration with 5000 tokens."
        )
        budget_results, budget_report = run_pipeline(
            tickets,
            ZERO_SHOT_PROMPT,
            token_budget=demo_budget,
        )

    # If an unusually short prompt still avoids exhaustion,
    # lower the demo budget further until it demonstrates stopping.
    while "budget_exhausted" not in budget_report["status_counts"]:
        demo_budget -= 500
        if demo_budget <= 0:
            raise RuntimeError(
                "Could not demonstrate token-budget exhaustion."
            )

        budget_results, budget_report = run_pipeline(
            tickets,
            ZERO_SHOT_PROMPT,
            token_budget=demo_budget,
        )

    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_results.to_csv(budget_path, index=False)

    print_report(
        f"BUDGET DEMO RESULTS (budget={demo_budget})",
        budget_report,
    )
    print(
        "Budget-exhausted tickets:",
        budget_report["status_counts"]["budget_exhausted"],
    )
    print(f"Saved: {budget_path}")

    # Final few-shot run uses the required 100000-token budget.
    final_results, final_report = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        token_budget=100000,
    )

    segments = load_customer_segments()
    final_output = apply_routing(final_results, segments)

    final_path = OUTPUT_DIR / "triage_results.csv"
    final_output.to_csv(final_path, index=False)

    print_report("FINAL FEW-SHOT RUN (budget=100000)", final_report)
    print("\nFinal routing counts:")
    print(final_output["route"].value_counts().to_string())
    print(f"\nSaved: {final_path}")

    # Acceptance checks.
    assert len(final_output) == 164, (
        f"Expected 164 final rows, got {len(final_output)}"
    )
    assert list(final_output.columns) == [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    print("\nPart C pipeline completed successfully.")


if __name__ == "__main__":
    main()
