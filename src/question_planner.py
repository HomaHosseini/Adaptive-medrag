import json
import re


class QuestionPlanner:
    """
    Analyzes a medical question before retrieval.

    The planner runs ONCE per question.

    It:
        - estimates question complexity
        - determines an initial query budget
        - identifies CORE requirements
        - identifies SUPPORTING constraints

    It does NOT:
        - answer the question
        - see answer choices
        - see retrieved evidence
        - decide STOP / CONTINUE
    """

    def __init__(self, medrag_instance):
        self.medrag = medrag_instance

    def plan(self, question):

        prompt = f"""
You are a medical retrieval planning agent.

Your task is to analyze the structure of the QUESTION before retrieval.

QUESTION:
{question}

IMPORTANT RULES:

1. Do NOT answer the question.
2. Do NOT use answer choices.
3. Do NOT use external medical knowledge to determine the answer.
4. Analyze only what information the question requires in order to be
   answered reliably.
5. Requirements must be atomic and specific.
6. Do not combine multiple distinct information needs into one requirement.
7. Do not create unnecessary requirements.

--------------------------------------------------
PART 1 — QUESTION COMPLEXITY
--------------------------------------------------

Estimate the structural complexity of the question from 0 to 1.

Consider:

- number of distinct facts that must be connected
- number of reasoning steps
- whether multiple clinical findings must be integrated
- whether a mechanism is required
- whether a diagnosis must be established
- whether treatment or management must be determined
- whether timing matters
- whether patient-specific constraints matter
- whether multiple relationships must be established

Complexity:

0.0-0.2 = very simple
0.2-0.4 = simple
0.4-0.6 = moderate
0.6-0.8 = complex
0.8-1.0 = very complex

--------------------------------------------------
PART 2 — INITIAL QUERY BUDGET
--------------------------------------------------

Choose an INITIAL query budget from 1 to 4.

This is only the starting budget.
It is NOT the maximum number of queries for the question.

Use:

1 = very simple
2 = normal
3 = complex
4 = very complex

The later controller will dynamically change the budget.

--------------------------------------------------
PART 3 — CORE REQUIREMENTS
--------------------------------------------------

Identify the CORE information requirements.

A core requirement is an atomic piece of information that must be
supported by evidence for the medical QA system to reliably answer
the question.

IMPORTANT:

A CORE requirement must be directly necessary to determine the answer
to the question.

Do NOT classify supporting evidence as a core requirement merely because
it helps explain the answer.

For example, if a question asks "What is the most likely diagnosis?",
the diagnosis is a core requirement. Findings, risk factors, mechanisms,
and causal explanations are usually supporting constraints unless the
question explicitly requires them.

Only make a mechanism a CORE requirement when the question asks for,
depends directly on, or requires identifying that mechanism.

Each core requirement must contain:

- id
- description
- type
- importance

Allowed types:

"diagnosis"
"cause"
"mechanism"
"treatment"
"management"
"therapeutic_effect"
"adverse_effect"
"clinical_relationship"
"comparison"
"other"

Importance:

1.0 = essential
0.8-0.9 = highly important
0.6-0.7 = important
0.4-0.5 = useful

IMPORTANT:
Keep requirements atomic.

BAD:
"Understand the relationship between the drug, disease,
symptoms, and treatment."

--------------------------------------------------
PART 4 — SUPPORTING CONSTRAINTS
--------------------------------------------------

Identify important constraints and contextual facts from the question
that help determine the answer but are not necessarily separate
answer requirements.

Examples:

- patient age
- sex
- pregnancy
- recent procedure
- timing
- specific symptoms
- laboratory findings
- imaging findings
- disease stage
- previous treatment
- important clinical context

Each supporting constraint must contain:

- id
- description
- type
- importance

Allowed types:

"patient_constraint"
"clinical_finding"
"temporal_relationship"
"diagnostic_context"
"treatment_context"
"other"

Importance:

1.0 = essential constraint
0.8-0.9 = highly important
0.6-0.7 = important
0.4-0.5 = useful

Do NOT duplicate a core requirement as a supporting constraint.

--------------------------------------------------
PART 5 — REASONING
--------------------------------------------------

Briefly explain why the question has this complexity and why the
identified requirements are necessary.

--------------------------------------------------
OUTPUT
--------------------------------------------------

Return ONLY valid JSON.

Use exactly this structure:

{{
    "complexity": ,

    "initial_query_budget":,

    "core_requirements": [
        {{
            "id": "R1",
            "description": "",
            "type": "",
            "importance":
        }}
    ],

    "supporting_constraints": [
        {{
            "id": "C1",
            "description": "",
            "type": "",
            "importance":
        }}
    ],

    "reasoning": "Brief explanation."
}}
"""

        try:

            response = self.medrag.generate([
                {
                    "role": "user",
                    "content": prompt
                }
            ]).strip()

            # --------------------------------------------------
            # Remove markdown fences
            # --------------------------------------------------

            response = re.sub(
                r"^```json\s*",
                "",
                response,
                flags=re.IGNORECASE
            )

            response = re.sub(
                r"^```\s*",
                "",
                response
            )

            response = re.sub(
                r"\s*```$",
                "",
                response
            )

            response = response.strip()

            result = json.loads(response)

            # --------------------------------------------------
            # Complexity
            # --------------------------------------------------

            complexity = float(
                result.get("complexity", 0.5)
            )

            complexity = max(
                0.0,
                min(1.0, complexity)
            )

            # --------------------------------------------------
            # Initial query budget
            # --------------------------------------------------

            initial_budget = int(
                result.get(
                    "initial_query_budget",
                    2
                )
            )

            initial_budget = max(
                1,
                min(4, initial_budget)
            )

            # --------------------------------------------------
            # Parse core requirements
            # --------------------------------------------------

            core_requirements = self._clean_requirements(
                result.get("core_requirements", []),
                prefix="R"
            )

            # --------------------------------------------------
            # Parse supporting constraints
            # --------------------------------------------------

            supporting_constraints = self._clean_constraints(
                result.get("supporting_constraints", []),
                prefix="C"
            )

            # --------------------------------------------------
            # Reasoning
            # --------------------------------------------------

            reasoning = str(
                result.get("reasoning", "")
            )

            return {
                "complexity": complexity,
                "initial_query_budget": initial_budget,
                "core_requirements": core_requirements,
                "supporting_constraints": supporting_constraints,
                "reasoning": reasoning
            }

        except Exception as e:

            print(
                f"[Question Planner Error]: {e}"
            )

            # Safe fallback
            return {
                "complexity": 0.5,
                "initial_query_budget": 2,
                "core_requirements": [],
                "supporting_constraints": [],
                "reasoning": f"Planner error: {e}"
            }

    # ==========================================================
    # CORE REQUIREMENTS
    # ==========================================================

    @staticmethod
    def _clean_requirements(
        requirements,
        prefix="R"
    ):
        cleaned = []

        if not isinstance(requirements, list):
            return cleaned

        for i, req in enumerate(requirements):

            if not isinstance(req, dict):
                continue

            description = str(
                req.get("description", "")
            ).strip()

            if not description:
                continue

            req_id = str(
                req.get(
                    "id",
                    f"{prefix}{i + 1}"
                )
            ).strip()

            req_type = str(
                req.get(
                    "type",
                    "other"
                )
            ).strip()

            try:
                importance = float(
                    req.get(
                        "importance",
                        0.5
                    )
                )
            except (TypeError, ValueError):
                importance = 0.5

            importance = max(
                0.0,
                min(1.0, importance)
            )

            cleaned.append({
                "id": req_id,
                "description": description,
                "type": req_type,
                "importance": importance
            })

        return cleaned

    # ==========================================================
    # SUPPORTING CONSTRAINTS
    # ==========================================================

    @staticmethod
    def _clean_constraints(
        constraints,
        prefix="C"
    ):
        cleaned = []

        if not isinstance(constraints, list):
            return cleaned

        for i, item in enumerate(constraints):

            if not isinstance(item, dict):
                continue

            description = str(
                item.get("description", "")
            ).strip()

            if not description:
                continue

            item_id = str(
                item.get(
                    "id",
                    f"{prefix}{i + 1}"
                )
            ).strip()

            item_type = str(
                item.get(
                    "type",
                    "other"
                )
            ).strip()

            try:
                importance = float(
                    item.get(
                        "importance",
                        0.5
                    )
                )
            except (TypeError, ValueError):
                importance = 0.5

            importance = max(
                0.0,
                min(1.0, importance)
            )

            cleaned.append({
                "id": item_id,
                "description": description,
                "type": item_type,
                "importance": importance
            })

        return cleaned