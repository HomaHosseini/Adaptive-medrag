import json
import re


class EvidenceCoverageAnalyzer:
    """
    Evaluates how well accumulated retrieved evidence covers
    the fixed requirements produced by QuestionPlanner.

    The analyzer:
        - sees the question
        - sees core requirements
        - sees supporting constraints
        - sees accumulated retrieved evidence

    It does NOT:
        - see answer choices
        - answer the medical question
        - decide STOP / CONTINUE
        - generate queries
        - calculate its own aggregate scores via the LLM

    The LLM only evaluates each requirement as:
        COVERED
        PARTIAL
        MISSING
        CONTRADICTED

    Python calculates the aggregate coverage scores.
    """

    VALID_STATUSES = {
        "COVERED",
        "PARTIAL",
        "MISSING",
        "CONTRADICTED",
    }

    STATUS_SCORES = {
        "COVERED": 1.0,
        "PARTIAL": 0.5,
        "MISSING": 0.0,
        "CONTRADICTED": 0.0,
    }

    def __init__(self, medrag_instance):
        self.medrag = medrag_instance

    # ==========================================================
    # PUBLIC METHOD
    # ==========================================================

    def analyze(
        self,
        question,
        core_requirements,
        supporting_constraints,
        context,
    ):
        """
        Analyze evidence coverage.

        Returns:
            {
                "coverage": [...],
                "constraint_coverage": [...],
                "core_coverage": float,
                "weighted_coverage": float,
                "reasoning": str
            }
        """

        # ------------------------------------------------------
        # No evidence
        # ------------------------------------------------------

        if not context or not context.strip():
            return {
                "coverage": [],
                "constraint_coverage": [],
                "core_coverage": 0.0,
                "weighted_coverage": 0.0,
                "reasoning": "No retrieved evidence available.",
            }

        # ------------------------------------------------------
        # No requirements
        # ------------------------------------------------------

        if not core_requirements:
            return {
                "coverage": [],
                "constraint_coverage": [],
                "core_coverage": 0.0,
                "weighted_coverage": 0.0,
                "reasoning": "No core requirements were provided.",
            }

        core_text = self._format_items(core_requirements)
        constraint_text = self._format_items(
            supporting_constraints
        )

        prompt = f"""
You are a strict medical evidence coverage analyzer.

Your task is NOT to answer the medical question.

Your task is to evaluate whether the accumulated retrieved evidence
covers the information requirements identified before retrieval.

QUESTION:
{question}

==================================================
CORE REQUIREMENTS
==================================================

{core_text}

==================================================
SUPPORTING CONSTRAINTS
==================================================

{constraint_text}

==================================================
ACCUMULATED RETRIEVED EVIDENCE
==================================================

{context}

==================================================
TASK
==================================================

Evaluate EVERY CORE REQUIREMENT individually.

Use exactly one status:

COVERED:
The retrieved evidence directly and sufficiently supports
the requirement.

PARTIAL:
The evidence provides useful support, but an important part
of the requirement remains uncertain or unsupported.

MISSING:
The evidence does not provide sufficient information
to support the requirement.

CONTRADICTED:
The retrieved evidence contains information that directly
conflicts with the requirement.

IMPORTANT RULES:

- Judge ONLY the retrieved evidence.
- Do NOT use outside medical knowledge to fill gaps.
- Do NOT use answer choices.
- Do NOT infer evidence that is not present.
- Be conservative.
- Evaluate every requirement independently.
- Do not combine multiple requirements into one judgment.
- Do not omit any requirement.

For every core requirement return:

- id
- status
- confidence from 0 to 1
- evidence_summary

Also evaluate every supporting constraint using the same statuses.

Do NOT calculate overall coverage scores.
Python will calculate those values.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "coverage": [
        {{
            "id": "R1",
            "status": "COVERED",
            "confidence": 0.95,
            "evidence_summary": "The evidence directly supports this requirement."
        }}
    ],

    "constraint_coverage": [
        {{
            "id": "C1",
            "status": "PARTIAL",
            "confidence": 0.65,
            "evidence_summary": "The evidence partially supports this constraint."
        }}
    ],

    "reasoning": "Brief explanation of the overall evidence coverage."
}}
"""

        try:
            response = self.medrag.generate(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )

            if response is None:
                raise ValueError(
                    "Coverage model returned None."
                )

            response = str(response).strip()

            print("\n[Coverage Raw Response]")
            print(repr(response))

            if not response:
                raise ValueError(
                    "Coverage model returned an empty response."
                )

            result = self._parse_json_response(
                response
            )

            # --------------------------------------------------
            # Parse core coverage
            # --------------------------------------------------

            raw_coverage = result.get(
                "coverage",
                []
            )

            cleaned_coverage = self._clean_results(
                raw_coverage
            )

            # --------------------------------------------------
            # Parse supporting constraints
            # --------------------------------------------------

            raw_constraints = result.get(
                "constraint_coverage",
                []
            )

            cleaned_constraints = self._clean_results(
                raw_constraints
            )

            # --------------------------------------------------
            # Calculate aggregate scores in Python
            # --------------------------------------------------

            core_coverage = self._calculate_core_coverage(
                core_requirements,
                cleaned_coverage
            )

            weighted_coverage = (
                self._calculate_weighted_coverage(
                    core_requirements,
                    cleaned_coverage
                )
            )

            reasoning = str(
                result.get(
                    "reasoning",
                    ""
                )
            ).strip()

            return {
                "coverage": cleaned_coverage,
                "constraint_coverage": cleaned_constraints,
                "core_coverage": core_coverage,
                "weighted_coverage": weighted_coverage,
                "reasoning": reasoning,
            }

        except Exception as e:

            print(
                f"\n[Evidence Coverage Error]: {e}"
            )

            return {
                "coverage": [],
                "constraint_coverage": [],
                "core_coverage": 0.0,
                "weighted_coverage": 0.0,
                "reasoning": (
                    f"Coverage analyzer error: {e}"
                ),
            }

    # ==========================================================
    # JSON PARSER
    # ==========================================================

# ==========================================================
    # JSON PARSER (FIXED & BULLETPROOF)
    # ==========================================================

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """
        Robustly extract and repair JSON objects from LLM responses.
        
        Handles:
            - Markdown code blocks (```json ... ```)
            - Leading/trailing conversational text
            - Trailing commas before closing braces (e.g., {"a": 1,})
            - Malformed quotes/escapes via fallback repairs
        """
        if not response or not response.strip():
            raise ValueError("Response string is empty.")

        text = response.strip()

        # ------------------------------------------------------
        # Tier 1: Extract block inside markdown fences if present
        # ------------------------------------------------------
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if markdown_match:
            text = markdown_match.group(1).strip()

        # ------------------------------------------------------
        # Tier 2: Extract outermost JSON object bounds ({ ... })
        # ------------------------------------------------------
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        # Attempt 1: Direct standard parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # ------------------------------------------------------
        # Tier 3: Regex fix for trailing commas before } or ]
        # ------------------------------------------------------
        sanitized_text = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            parsed = json.loads(sanitized_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # ------------------------------------------------------
        # Tier 4: Fallback to `json_repair` library (if installed)
        # ------------------------------------------------------
        try:
            import json_repair  # Install via: pip install json-repair
            repaired = json_repair.loads(text)
            if isinstance(repaired, dict):
                return repaired
        except ImportError:
            pass
        except Exception:
            pass

        # ------------------------------------------------------
        # Fail Safe: Return empty structure instead of crashing pipeline
        # ------------------------------------------------------
        print(f"[JSON Parse Warning]: Failed to parse structure. Payload raw head:\n{response[:200]}...")
        return {
            "coverage": [],
            "constraint_coverage": [],
            "reasoning": "Failed to parse JSON response from LLM."
        }

    # ==========================================================
    # CLEAN LLM RESULT
    # ==========================================================

    def _clean_results(self, items):
        """
        Normalize LLM-produced coverage records.
        """

        if not isinstance(items, list):
            return []

        cleaned = []

        for item in items:

            if not isinstance(item, dict):
                continue

            item_id = str(
                item.get(
                    "id",
                    ""
                )
            ).strip()

            if not item_id:
                continue

            status = str(
                item.get(
                    "status",
                    "MISSING"
                )
            ).upper().strip()

            if status not in self.VALID_STATUSES:
                status = "MISSING"

            try:
                confidence = float(
                    item.get(
                        "confidence",
                        0.0
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                confidence = 0.0

            confidence = max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            )

            evidence_summary = str(
                item.get(
                    "evidence_summary",
                    ""
                )
            ).strip()

            cleaned.append(
                {
                    "id": item_id,
                    "status": status,
                    "confidence": confidence,
                    "evidence_summary": evidence_summary,
                }
            )

        return cleaned

    # ==========================================================
    # CORE COVERAGE
    # ==========================================================

    def _calculate_core_coverage(
        self,
        core_requirements,
        coverage,
    ):
        """
        Unweighted fraction of core requirements covered.

        COVERED = 1
        PARTIAL = 0.5
        MISSING = 0
        CONTRADICTED = 0
        """

        if not core_requirements:
            return 0.0

        coverage_map = {
            item["id"]: item
            for item in coverage
        }

        total_score = 0.0

        for requirement in core_requirements:

            req_id = str(
                requirement.get(
                    "id",
                    ""
                )
            )

            result = coverage_map.get(
                req_id
            )

            if result is None:
                score = 0.0
            else:
                score = self.STATUS_SCORES.get(
                    result["status"],
                    0.0
                )

            total_score += score

        return total_score / len(
            core_requirements
        )

    # ==========================================================
    # WEIGHTED COVERAGE
    # ==========================================================

    def _calculate_weighted_coverage(
        self,
        core_requirements,
        coverage,
    ):
        """
        Importance-weighted coverage.

        This is useful because all requirements do not necessarily
        have equal importance.
        """

        if not core_requirements:
            return 0.0

        coverage_map = {
            item["id"]: item
            for item in coverage
        }

        weighted_score = 0.0
        total_weight = 0.0

        for requirement in core_requirements:

            req_id = str(
                requirement.get(
                    "id",
                    ""
                )
            )

            try:
                importance = float(
                    requirement.get(
                        "importance",
                        1.0
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                importance = 1.0

            importance = max(
                0.0,
                min(
                    1.0,
                    importance
                )
            )

            total_weight += importance

            result = coverage_map.get(
                req_id
            )

            if result is None:
                score = 0.0
            else:
                score = self.STATUS_SCORES.get(
                    result["status"],
                    0.0
                )

            weighted_score += (
                score * importance
            )

        if total_weight == 0:
            return 0.0

        return (
            weighted_score /
            total_weight
        )

    # ==========================================================
    # FORMAT REQUIREMENTS
    # ==========================================================

    @staticmethod
    def _format_items(items):
        """
        Convert requirement/constraint dictionaries
        into readable prompt text.
        """

        if not items:
            return "None."

        lines = []

        for item in items:

            item_id = str(
                item.get(
                    "id",
                    ""
                )
            )

            description = str(
                item.get(
                    "description",
                    ""
                )
            )

            item_type = str(
                item.get(
                    "type",
                    "other"
                )
            )

            try:
                importance = float(
                    item.get(
                        "importance",
                        0.0
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                importance = 0.0

            lines.append(
                f"{item_id}: "
                f"{description} "
                f"[type={item_type}, "
                f"importance={importance:.2f}]"
            )

        return "\n".join(lines)