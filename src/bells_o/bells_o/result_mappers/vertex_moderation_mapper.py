"""Implement the result mapping function for Vertex AI Moderation."""

from typing import Any

from bells_o.common import Result, Usage


def mapper(output: dict[str, Any], usage: Usage = Usage()) -> Result:
    """Map the output format of Vertex Gemini moderation to `Result`.

    Only maps to `content_moderation`.

    Args:
        output: A dict that is the raw response of a supervisor.
        usage: Conformity with ResultMapper type. Argument is ignored as usage type is static.

    Returns:
        Result: A Result object with content_moderation flag set based on safety ratings.
    """
    candidates = output.get("candidates", [])
    safety_ratings = []
    blocked = False

    for cand in candidates:
        for rating in cand.get("safetyRatings", []):
            rating_blocked = rating.get("blocked", False)
            safety_ratings.append(
                {
                    "category": rating.get("category"),
                    "probability": rating.get("probability"),  # e.g. "MEDIUM"
                    "blocked": rating_blocked,
                }
            )
            if rating_blocked:
                blocked = True

    # If no candidates or no safety ratings, default to not blocked
    # This handles cases where the API returns an error or empty response
    if not safety_ratings:
        blocked = False

    result = Result(content_moderation=blocked)
    return result

