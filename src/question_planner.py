import json
import re


class QuestionPlanner:
   

    VALID_INTENTS = {
        "diagnosis",
        "cause",
        "mechanism",
        "treatment",
        "management",
        "therapeutic_effect",
        "adverse_effect",
        "prognosis",
        "comparison",
        "interpretation",
        "association",
        "other",
    }

    VALID_RETRIEVAL_NEEDS = {
        "NO_RAG",
        "RAG_RECOMMENDED",
        "RAG_REQUIRED",
    }

    def __init__(self, medrag_instance):
        self.medrag = medrag_instance

    

    def plan(self, question):

        if not question or not str(question).strip():
            raise ValueError(
                "QuestionPlanner received an empty question."
            )

        question = str(question).strip()

        prompt = self._build_prompt(question)

        try:

            response = self.medrag.generate([
                {
                    "role": "user",
                    "content": prompt
                }
            ])

            if response is None:
                raise ValueError(
                    "Planner model returned None."
                )

            response = str(response).strip()

            print("\n[Planner Raw Response]")
            print(repr(response))

            result = self._parse_json_response(
                response
            )

            plan = self._clean_plan(
                result,
                question
            )

            self._print_plan(plan)

            return plan

        except Exception as e:

            print(
                f"[Question Planner Error]: {e}"
            )

            return self._fallback_plan()

    

    def _build_prompt(self, question):

        return f"""
You are a clinical retrieval planning agent.

Your job is to analyze a medical question BEFORE retrieval and
construct a precise retrieval schema.

You are NOT the final answerer.

You may generate plausible hypotheses ONLY to improve retrieval.

A hypothesis is NOT evidence.

A hypothesis must NEVER be presented as an established fact.

QUESTION:
{question}

==================================================
CORE PRINCIPLE
==================================================

First determine:

1. WHAT IS ALREADY EXPLICITLY ESTABLISHED BY THE QUESTION?
2. WHAT DOES THE QUESTION ACTUALLY ASK THE SOLVER TO DETERMINE?
3. WHAT INFORMATION IS STILL MISSING AND WOULD NEED EXTERNAL
   EVIDENCE?

Do NOT confuse these three categories.

For example, suppose the question explicitly says that pathology
is suspicious for non-Hodgkin lymphoma and asks which historical
risk factor is responsible.

Then:

KNOWN:
    non-Hodgkin lymphoma

QUESTION TARGET:
    determine which historical risk factor is associated with it

RETRIEVAL NEED:
    evidence about the relationship between the listed risk factors
    and non-Hodgkin lymphoma

Do NOT create a diagnostic hypothesis such as:
    "lymphoma"

because the condition is already explicitly identified.

However, if the question only provides symptoms and does NOT identify
the diagnosis, plausible diagnostic hypotheses may be generated to
improve retrieval.

==================================================
PART 1 — CLINICAL INTENT
==================================================

Identify the PRIMARY task.

Choose exactly ONE:

"diagnosis"
"cause"
"mechanism"
"treatment"
"management"
"therapeutic_effect"
"adverse_effect"
"prognosis"
"comparison"
"interpretation"
"association"
"other"

The intent describes what the question asks the solver to determine.

==================================================
PART 2 — QUESTION TARGET
==================================================

Describe precisely what remains to be determined.

This is NOT the answer.

It should state the decision or relationship the downstream system
must resolve.

Examples:

"identify the diagnosis explaining the clinical presentation"

"identify which historical exposure is associated with the established diagnosis"

"identify the mechanism producing the stated therapeutic effect"

"identify the appropriate management for the established condition"

GOOD:
"identify which stated risk factor is associated with the patient's lymphoma"

BAD:
"previous radiation therapy"

because that would already be the answer.

==================================================
PART 3 — OBSERVED ENTITIES
==================================================

Extract clinically important concepts that are explicitly present
in the question.

These may include:

- explicitly stated diagnoses
- symptoms
- signs
- drugs
- treatments
- procedures
- laboratory findings
- pathology findings
- imaging findings
- organisms explicitly named
- anatomical structures
- exposures
- clinical context

IMPORTANT:

Use only concepts grounded in the question.

Do not add an inferred diagnosis, cause, mechanism, organism,
drug, or treatment if it is not explicitly stated.

If a condition is explicitly stated, it may be included.

==================================================
PART 4 — DECISION-CRITICAL CONSTRAINTS
==================================================

Extract only explicitly stated facts that can materially affect
interpretation or retrieval.

Examples:

- age
- sex
- timing
- recent procedure
- recent treatment
- exposure
- disease stage
- severity
- anatomical site
- comorbidity
- risk factor
- clinical setting

Do NOT copy every detail from the question.

==================================================
PART 5 — RETRIEVAL HYPOTHESES
==================================================

Generate 0 to 3 hypotheses ONLY when hypotheses are useful for
retrieval.

A hypothesis is appropriate when an important part of the question
remains uncertain.

If the question already explicitly establishes the relevant target
condition, DO NOT generate diagnostic hypotheses merely to repeat it.

Hypotheses must:

- be plausible
- be relevant to the unresolved question
- be grounded in the clinical clues
- help retrieve or discriminate evidence
- remain explicitly hypothetical

Example when diagnosis is unknown:

Observed:
fever + lymphadenopathy + night sweats

Possible hypotheses:
- lymphoproliferative disorder
- infectious process
- malignancy-related condition

Example when diagnosis is already explicit:

Observed:
non-Hodgkin lymphoma

Question:
which historical factor is responsible?

Hypotheses:
[]

==================================================
PART 6 — RELATIONSHIPS TO ESTABLISH
==================================================

Identify the specific relationships external evidence may need to
establish.

Examples:

"historical exposure -> disease risk"

"drug -> therapeutic effect"

"drug -> adverse effect"

"clinical findings -> diagnosis"

"procedure -> complication"

"laboratory finding -> pathology"

"treatment -> intended outcome"

"exposure -> clinical syndrome"

Keep the relationship NEUTRAL.

BAD:
"radiation -> lymphoma"

if that is not already established.

GOOD:
"historical exposure -> lymphoma risk"

==================================================
PART 7 — INFORMATION NEEDS
==================================================

Convert the question target into the smallest possible set of
concrete information needs.

Each information need must describe something that external evidence
could establish.

Information needs MUST be directly aligned with:

    question target
    intent
    relationships

Do NOT turn every patient detail into an information need.

For a question asking which risk factor is responsible:

GOOD:
"determine which stated historical exposure is associated with the established condition"

"distinguish the relevant exposure from the other historical factors"

For a diagnostic question:

GOOD:
"identify conditions that explain the combined clinical findings"

"identify findings that distinguish the relevant diagnostic possibilities"

Each information need requires:

- id
- description
- importance

Importance:

1.0 = essential
0.8-0.9 = highly important
0.6-0.7 = useful

==================================================
PART 8 — RETRIEVAL NECESSITY
==================================================

Choose exactly ONE:

"NO_RAG"
"RAG_RECOMMENDED"
"RAG_REQUIRED"

NO_RAG:
The question can reasonably be answered from information already
contained in the question.

RAG_RECOMMENDED:
The model may know the answer, but retrieval would substantially
improve reliability.

RAG_REQUIRED:
The question depends on external medical knowledge that is not
contained in the question.

This is a retrieval-routing decision.

It is NOT the final answer.

==================================================
PART 9 — COMPLEXITY
==================================================

Estimate structural complexity from 0 to 1.

Consider:

- number of unresolved information needs
- number of relationships
- number of reasoning steps
- amount of clinical integration required
- specificity of external knowledge
- temporal reasoning
- patient-specific constraints
- ambiguity among plausible hypotheses

Do NOT increase complexity merely because the question is long.

0.0-0.2 = very simple
0.2-0.4 = simple
0.4-0.6 = moderate
0.6-0.8 = complex
0.8-1.0 = very complex

==================================================
PART 10 — INITIAL QUERY BUDGET
==================================================

Choose:

0
1
2
3
4

Use:

0 = NO_RAG
1 = one narrow information need
2 = normal retrieval
3 = multiple information needs or competing hypotheses
4 = highly complex retrieval

The budget is ONLY the starting budget.

A downstream adaptive controller can change it.

==================================================
PART 11 — INITIAL RETRIEVAL QUERY
==================================================

Generate ONE concise retrieval query.

The query should target the unresolved question.

It may use:

- observed entities
- important constraints
- retrieval hypotheses
- intent
- question target

The query MUST NOT simply repeat the whole question.

The query must not assume the final answer.

For questions where the target is already known, retrieve the
UNRESOLVED RELATIONSHIP.

Example:

Question:
patient has non-Hodgkin lymphoma and asks which historical risk
factor is responsible.

BAD:
"non-Hodgkin lymphoma risk factors including previous radiation"
because this begins to assume the desired answer.

BETTER:
"non-Hodgkin lymphoma risk factors prior radiation breast cancer"

For diagnostic questions where several explanations remain possible,
the query should preserve discrimination among them.

If retrieval_need = "NO_RAG":

q_init = ""

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "intent": "",

    "question_target": "",

    "observed_entities": [],

    "constraints": [],

    "hypotheses": [],

    "relationships_to_establish": [],

    "information_needs": [
        {{
            "id": "N1",
            "description": "",
            "importance": 1.0
        }}
    ],

    "retrieval_need": "RAG_REQUIRED",

    "complexity": 0.0,

    "initial_query_budget": 2,

    "q_init": "",

    "reasoning": ""
}}

Do not add any other keys.
"""

    

    @staticmethod
    def _parse_json_response(response):

        if not response or not response.strip():
            raise ValueError(
                "Empty planner response."
            )

        text = response.strip()

        # Markdown fences
        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^```\s*",
            "",
            text
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

        # Direct JSON
        try:

            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # Extract JSON object
        start = text.find("{")
        end = text.rfind("}")

        if (
            start == -1
            or end == -1
            or end <= start
        ):
            raise ValueError(
                "No JSON object found in planner response."
            )

        json_text = text[
            start:end + 1
        ]

        try:

            parsed = json.loads(
                json_text
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # Optional repair library
        try:

            import json_repair

            parsed = json_repair.loads(
                json_text
            )

            if isinstance(parsed, dict):
                return parsed

        except ImportError:
            pass

        except Exception:
            pass

        raise ValueError(
            "Unable to parse planner JSON."
        )

    
    def _clean_plan(self, result, question):

        

        intent = str(
            result.get(
                "intent",
                "other"
            )
        ).strip().lower()

        if intent not in self.VALID_INTENTS:
            intent = "other"

        

        question_target = str(
            result.get(
                "question_target",
                ""
            )
        ).strip()

        if not self._is_reasonable_text(
            question_target
        ):
            question_target = ""


        observed_entities = (
            self._clean_string_list(
                result.get(
                    "observed_entities",
                    []
                )
            )
        )

        

        constraints = (
            self._clean_string_list(
                result.get(
                    "constraints",
                    []
                )
            )
        )

        

        hypotheses = (
            self._clean_string_list(
                result.get(
                    "hypotheses",
                    []
                )
            )
        )

        

        relationships = (
            self._clean_string_list(
                result.get(
                    "relationships_to_establish",
                    []
                )
            )
        )

        

        information_needs = (
            self._clean_information_needs(
                result.get(
                    "information_needs",
                    []
                )
            )
        )

        

        retrieval_need = str(
            result.get(
                "retrieval_need",
                "RAG_RECOMMENDED"
            )
        ).strip().upper()

        if (
            retrieval_need
            not in self.VALID_RETRIEVAL_NEEDS
        ):
            retrieval_need = (
                "RAG_RECOMMENDED"
            )

        

        try:

            complexity = float(
                result.get(
                    "complexity",
                    0.5
                )
            )

        except (
            TypeError,
            ValueError
        ):

            complexity = 0.5

        complexity = max(
            0.0,
            min(
                1.0,
                complexity
            )
        )

        

        try:

            initial_budget = int(
                result.get(
                    "initial_query_budget",
                    2
                )
            )

        except (
            TypeError,
            ValueError
        ):

            initial_budget = 2

        initial_budget = max(
            0,
            min(
                4,
                initial_budget
            )
        )

        if retrieval_need == "NO_RAG":

            initial_budget = 0

        elif initial_budget == 0:

            initial_budget = 1

        
        q_init = str(
            result.get(
                "q_init",
                ""
            )
        ).strip()

        if not self._is_reasonable_text(
            q_init
        ):
            q_init = ""

        if retrieval_need == "NO_RAG":

            q_init = ""

        

        reasoning = str(
            result.get(
                "reasoning",
                ""
            )
        ).strip()

        if not self._is_reasonable_text(
            reasoning
        ):
            reasoning = ""

        plan = {
            "intent": intent,

            "question_target":
                question_target,

            "observed_entities":
                observed_entities,

            "constraints":
                constraints,

            "hypotheses":
                hypotheses,

            "relationships_to_establish":
                relationships,

            "information_needs":
                information_needs,

            "retrieval_need":
                retrieval_need,

            "complexity":
                complexity,

            "initial_query_budget":
                initial_budget,

            "q_init":
                q_init,

            "reasoning":
                reasoning,
        }

        return self._validate_plan(
            plan
        )

    

    @staticmethod
    def _validate_plan(plan):

        

        if (
            plan["retrieval_need"]
            == "NO_RAG"
        ):

            plan[
                "initial_query_budget"
            ] = 0

            plan["q_init"] = ""

        

        else:

            if (
                plan[
                    "initial_query_budget"
                ] <= 0
            ):

                plan[
                    "initial_query_budget"
                ] = 1

        
        if (
            plan["retrieval_need"]
            != "NO_RAG"
            and not plan["question_target"]
        ):

            plan["retrieval_need"] = (
                "RAG_RECOMMENDED"
            )

        

        if (
            plan["retrieval_need"]
            != "NO_RAG"
            and not plan["information_needs"]
        ):

            plan["retrieval_need"] = (
                "RAG_RECOMMENDED"
            )

        

        if plan["intent"] in {
            "association",
            "cause",
            "mechanism",
            "treatment",
            "management",
            "therapeutic_effect",
            "adverse_effect",
        }:

            pass

        return plan

   

    @staticmethod
    def _clean_string_list(items):

        if not isinstance(
            items,
            list
        ):

            return []

        cleaned = []

        for item in items:

            value = str(
                item
            ).strip()

            if not value:
                continue

            value = re.sub(
                r"^\s*[-*•]\s*",
                "",
                value
            ).strip()

            if not value:
                continue

            if not QuestionPlanner._is_reasonable_text(
                value
            ):
                continue

            cleaned.append(
                value
            )

        return QuestionPlanner._deduplicate(
            cleaned
        )

    @staticmethod
    def _deduplicate(items):

        output = []
        seen = set()

        for item in items:

            key = item.lower().strip()

            if key in seen:
                continue

            seen.add(key)

            output.append(item)

        return output

    

    @staticmethod
    def _clean_information_needs(items):

        if not isinstance(
            items,
            list
        ):

            return []

        cleaned = []

        for i, item in enumerate(items):

            if not isinstance(
                item,
                dict
            ):
                continue

            description = str(
                item.get(
                    "description",
                    ""
                )
            ).strip()

            if not description:
                continue

            if not QuestionPlanner._is_reasonable_text(
                description
            ):
                continue

            item_id = str(
                item.get(
                    "id",
                    f"N{i + 1}"
                )
            ).strip()

            if not item_id:
                item_id = f"N{i + 1}"

            try:

                importance = float(
                    item.get(
                        "importance",
                        0.5
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                importance = 0.5

            importance = max(
                0.0,
                min(
                1.0,
                importance
                )
            )

            cleaned.append({
                "id": item_id,
                "description": description,
                "importance": importance
            })

        return cleaned


    @staticmethod
    def _is_reasonable_text(value):

        if value is None:
            return False

        value = str(
            value
        ).strip()

        if not value:
            return False

        
        if len(value) > 500:
            return False

        
        if re.search(
            r"(.)\1{8,}",
            value
        ):
            return False

        return True

    

    @staticmethod
    def _print_plan(plan):

        print("\n[Clinical Schema]")

        print(
            f"   Intent: "
            f"{plan['intent']}"
        )

        print(
            f"   Question Target: "
            f"{plan['question_target']}"
        )

        print(
            f"   Retrieval Need: "
            f"{plan['retrieval_need']}"
        )

        print(
            f"   Complexity: "
            f"{plan['complexity']:.2f}"
        )

        print(
            f"   Initial Query Budget: "
            f"{plan['initial_query_budget']}"
        )

        print("\n   Observed Entities:")

        for item in plan[
            "observed_entities"
        ]:
            print(
                f"      - {item}"
            )

        print("\n   Constraints:")

        for item in plan[
            "constraints"
        ]:
            print(
                f"      - {item}"
            )

        print("\n   Retrieval Hypotheses:")

        for item in plan[
            "hypotheses"
        ]:
            print(
                f"      - {item}"
            )

        print("\n   Relationships:")

        for item in plan[
            "relationships_to_establish"
        ]:
            print(
                f"      - {item}"
            )

        print("\n   Information Needs:")

        for item in plan[
            "information_needs"
        ]:

            print(
                f"      {item['id']}: "
                f"{item['description']} "
                f"(importance="
                f"{item['importance']:.2f})"
            )

        print(
            f"\n   Initial Query: "
            f"{plan['q_init'] or '[NONE]'}"
        )

        print(
            f"\n   Planner Reasoning: "
            f"{plan['reasoning']}"
        )

    

    @staticmethod
    def _fallback_plan():

        return {
            "intent": "other",

            "question_target": "",

            "observed_entities": [],

            "constraints": [],

            "hypotheses": [],

            "relationships_to_establish": [],

            "information_needs": [],

            "retrieval_need":
                "RAG_RECOMMENDED",

            "complexity": 0.5,

            "initial_query_budget": 2,

            "q_init": "",

            "reasoning":
                "Planner failed; using "
                "a conservative fallback."
        }