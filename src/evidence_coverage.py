import json
import re


class EvidenceCoverageAnalyzer:
    """
    Evaluates whether accumulated retrieved evidence supports the
    information needs identified by QuestionPlanner.

    The analyzer sees:
        - the original question
        - the question target
        - information needs
        - decision-critical constraints
        - relationships to establish
        - accumulated retrieved evidence

    The analyzer does NOT:
        - see answer choices
        - produce the final answer
        - decide STOP / CONTINUE
        - generate queries
        - use planner hypotheses as evidence

    The LLM evaluates each information need independently as:

        COVERED
        PARTIAL
        MISSING
        CONTRADICTED

    Python calculates aggregate coverage scores.
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


    def analyze(
        self,
        question,
        question_target,
        information_needs,
        constraints,
        relationships,
        context,
    ):
        """
        Analyze whether accumulated evidence covers the information
        required to resolve the question.

        Returns:

        {
            "coverage": [...],
            "constraint_coverage": [...],
            "core_coverage": float,
            "weighted_coverage": float,
            "reasoning": str
        }

        Note:
            "core_coverage" is retained as a compatibility name.
            It now means information-need coverage.
        """

        

        if not context or not context.strip():

            return {
                "coverage": [],
                "constraint_coverage": [],
                "core_coverage": 0.0,
                "weighted_coverage": 0.0,
                "reasoning":
                    "No retrieved evidence available.",
            }

       

        if not information_needs:

            return {
                "coverage": [],
                "constraint_coverage": [],
                "core_coverage": 0.0,
                "weighted_coverage": 0.0,
                "reasoning":
                    "No information needs were provided.",
            }

        

        information_need_text = (
            self._format_information_needs(
                information_needs
            )
        )

        constraint_text = (
            self._format_strings(
                constraints
            )
        )

        relationship_text = (
            self._format_strings(
                relationships
            )
        )

        

        prompt = f"""
You are a strict medical evidence coverage analyzer.

Your job is to evaluate whether the retrieved evidence contains
the information needed to resolve the medical question.

You are NOT the final answerer.

Do NOT answer the question.

==================================================
QUESTION
==================================================

{question}

==================================================
QUESTION TARGET
==================================================

{question_target}

The question target describes what the downstream system still
needs to determine.

Do not treat the question target itself as evidence.

==================================================
INFORMATION NEEDS
==================================================

{information_need_text}

==================================================
DECISION-CRITICAL CONSTRAINTS
==================================================

{constraint_text}

==================================================
RELATIONSHIPS TO ESTABLISH
==================================================

{relationship_text}

==================================================
ACCUMULATED RETRIEVED EVIDENCE
==================================================

{context}

==================================================
MAIN TASK
==================================================

Evaluate EVERY INFORMATION NEED independently.

For each information need determine whether the retrieved evidence
is sufficient to support it.

Use exactly one status:

COVERED:
The evidence directly and sufficiently supports the complete
information need.

PARTIAL:
The evidence provides useful support, but an important part of
the information need remains uncertain, indirect, or unsupported.

MISSING:
The evidence does not provide sufficient information to support
the information need.

CONTRADICTED:
The retrieved evidence contains information that directly conflicts
with the information need.

==================================================
IMPORTANT RULES
==================================================

1. Judge ONLY the retrieved evidence.

2. Do NOT use outside medical knowledge to fill gaps.

3. Do NOT use answer choices.

4. Do NOT answer the medical question.

5. Do NOT assume that a statement is true merely because it appears
   in the question.

6. Do NOT treat the planner's hypotheses as evidence.

7. Do NOT treat the question target as evidence.

8. Evaluate EVERY information need independently.

9. Do NOT combine multiple information needs into one judgment.

10. Do NOT mark a requirement COVERED simply because a related
    medical topic is mentioned.

11. Evidence must support the actual decision or relationship
    represented by the information need.

12. A generic fact about a disease is insufficient if the information
    need requires a patient-specific relationship or discrimination.

13. Distinguish between:
       "the evidence mentions X"
    and
       "the evidence establishes the relationship required by X."

==================================================
EXAMPLE
==================================================

Information need:

"Determine which stated historical exposure is associated with
the established condition."

Evidence:

"Radiation therapy has been associated with increased risk of
some malignancies."

This may provide support, but it does NOT automatically prove
that radiation is the relevant factor for this question unless
the evidence sufficiently supports the required relationship.

Similarly:

Evidence saying:

"Non-Hodgkin lymphoma can involve lymph nodes."

does NOT establish which historical exposure is responsible.

==================================================
PATIENT-SPECIFIC INFORMATION
==================================================

Use the decision-critical constraints and relationships only to
understand what kind of evidence is needed.

Do NOT mark a constraint COVERED merely because the same fact
appears in the original question.

A constraint is COVERED only when the retrieved evidence provides
meaningful support for that constraint when it is relevant to
the unresolved decision.

==================================================
CONFIDENCE
==================================================

For each information need provide confidence from 0 to 1.

Interpretation:

0.0 - 0.2 = essentially unsupported
0.2 - 0.4 = weak support
0.4 - 0.6 = partial support
0.6 - 0.8 = useful but incomplete support
0.8 - 1.0 = strong/direct support

Confidence describes confidence in the coverage judgment,
NOT confidence in the final medical answer.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "coverage": [
        {{
            "id": "N1",
            "status": "COVERED",
            "confidence": 0.95,
            "evidence_summary":
                "The retrieved evidence directly supports this information need."
        }}
    ],

    "constraint_coverage": [
        {{
            "id": "C1",
            "status": "PARTIAL",
            "confidence": 0.65,
            "evidence_summary":
                "The retrieved evidence provides partial support for this constraint."
        }}
    ],

    "reasoning":
        "Brief explanation of the overall evidence coverage."
}}

IMPORTANT:

- Return one record for EVERY information need.
- Return one record for every supplied constraint.
- Use the exact IDs supplied in the input.
- Do NOT calculate aggregate coverage.
- Python will calculate aggregate scores.
"""

        try:

            response = self.medrag.generate([
                {
                    "role": "user",
                    "content": prompt
                }
            ])

            if response is None:
                raise ValueError(
                    "Coverage model returned None."
                )

            response = str(
                response
            ).strip()

            print(
                "\n[Coverage Raw Response]"
            )

            print(
                repr(response)
            )

            if not response:
                raise ValueError(
                    "Coverage model returned an empty response."
                )

            result = self._parse_json_response(
                response
            )

            

            raw_coverage = result.get(
                "coverage",
                []
            )

            cleaned_coverage = (
                self._clean_results(
                    raw_coverage
                )
            )

            

            raw_constraints = (
                result.get(
                    "constraint_coverage",
                    []
                )
            )

            cleaned_constraints = (
                self._clean_results(
                    raw_constraints
                )
            )

            core_coverage = (
                self._calculate_core_coverage(
                    information_needs,
                    cleaned_coverage
                )
            )

            weighted_coverage = (
                self._calculate_weighted_coverage(
                    information_needs,
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
                "coverage":
                    cleaned_coverage,

                "constraint_coverage":
                    cleaned_constraints,

                "core_coverage":
                    core_coverage,

                "weighted_coverage":
                    weighted_coverage,

                "reasoning":
                    reasoning,
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
                "reasoning":
                    f"Coverage analyzer error: {e}",
            }

    
    @staticmethod
    def _parse_json_response(
        response: str
    ) -> dict:
        """
        Robustly parse JSON from an LLM response.

        Handles:
            - plain JSON
            - markdown code fences
            - surrounding text
            - trailing commas
            - json_repair if installed
        """

        if not response or not response.strip():
            raise ValueError(
                "Response string is empty."
            )

        text = response.strip()

        
        markdown_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            text,
            flags=re.IGNORECASE
        )

        if markdown_match:

            text = (
                markdown_match
                .group(1)
                .strip()
            )

        
        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):

            text = text[
                start:end + 1
            ]

        
        try:

            parsed = json.loads(
                text
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        

        sanitized = re.sub(
            r",\s*([}\]])",
            r"\1",
            text
        )

        try:

            parsed = json.loads(
                sanitized
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

       
        try:

            import json_repair

            parsed = json_repair.loads(
                text
            )

            if isinstance(parsed, dict):
                return parsed

        except ImportError:

            pass

        except Exception:

            pass

        
        print(
            "[Coverage JSON Parse Warning]"
        )

        print(
            f"Raw response:\n"
            f"{response[:1000]}"
        )

        raise ValueError(
            "Unable to parse coverage JSON."
        )

    

    def _clean_results(
        self,
        items
    ):
        """
        Normalize LLM-produced coverage records.
        """

        if not isinstance(
            items,
            list
        ):
            return []

        cleaned = []

        for item in items:

            if not isinstance(
                item,
                dict
            ):
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

            if (
                status
                not in self.VALID_STATUSES
            ):

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

            cleaned.append({
                "id":
                    item_id,

                "status":
                    status,

                "confidence":
                    confidence,

                "evidence_summary":
                    evidence_summary,
            })

        return cleaned



    def _calculate_core_coverage(
        self,
        information_needs,
        coverage,
    ):
        """
        Unweighted coverage across information needs.

        COVERED      = 1.0
        PARTIAL      = 0.5
        MISSING      = 0.0
        CONTRADICTED = 0.0
        """

        if not information_needs:
            return 0.0

        coverage_map = {
            item["id"]: item
            for item in coverage
        }

        total_score = 0.0

        for need in information_needs:

            need_id = str(
                need.get(
                    "id",
                    ""
                )
            )

            result = coverage_map.get(
                need_id
            )

            if result is None:

                score = 0.0

            else:

                score = (
                    self.STATUS_SCORES.get(
                        result["status"],
                        0.0
                    )
                )

            total_score += score

        return (
            total_score
            / len(information_needs)
        )

    
    def _calculate_weighted_coverage(
        self,
        information_needs,
        coverage,
    ):
        """
        Importance-weighted coverage across information needs.
        """

        if not information_needs:
            return 0.0

        coverage_map = {
            item["id"]: item
            for item in coverage
        }

        weighted_score = 0.0
        total_weight = 0.0

        for need in information_needs:

            need_id = str(
                need.get(
                    "id",
                    ""
                )
            )

            try:

                importance = float(
                    need.get(
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
                need_id
            )

            if result is None:

                score = 0.0

            else:

                score = (
                    self.STATUS_SCORES.get(
                        result["status"],
                        0.0
                    )
                )

            weighted_score += (
                score * importance
            )

        if total_weight <= 0:
            return 0.0

        return (
            weighted_score
            / total_weight
        )

    

    @staticmethod
    def _format_information_needs(
        items
    ):
        """
        Format information needs for the LLM.
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
                f"[importance="
                f"{importance:.2f}]"
            )

        return "\n".join(
            lines
        )


    @staticmethod
    def _format_strings(
        items
    ):
        """
        Format planner string lists.
        """

        if not items:
            return "None."

        lines = []

        for item in items:

            value = str(
                item
            ).strip()

            if not value:
                continue

            lines.append(
                f"- {value}"
            )

        if not lines:
            return "None."

        return "\n".join(
            lines
        )