from .medrag import MedRAG

import json
import re
import ast

from .evidence_tracker import EvidenceTracker
from .evidence_coverage import EvidenceCoverageAnalyzer
from .question_planner import QuestionPlanner


# ============================================================
# JSON HELPER
# ============================================================

def parse_json_response(response):
    """
    Robustly extract a JSON object from an LLM response.
    """

    if response is None:
        raise ValueError("LLM returned None.")

    text = str(response).strip()

    if not text:
        raise ValueError("LLM returned empty response.")

    # Remove markdown fences
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

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "No JSON object found in LLM response."
        )

    json_text = text[start:end + 1]

    try:
        parsed = json.loads(json_text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Optional repair
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
        "Unable to parse JSON response."
    )


def check_stopping_gate(
    medrag_instance,
    question,
    context,
    threshold=85,
    gate_features=None,
    coverage_stats=None,
    evidence_stats=None,
    options=None,
    use_options_in_gate=False,
):
    """
    Adaptive stopping controller.

    The controller decides whether additional retrieval is
    likely to materially improve the reliability of downstream
    answering.

    Gate features:
        confidence
        answerable
        coverage
        novelty
        redundancy
        discrimination

    IMPORTANT:
        Answer choices can optionally be supplied.

        When use_options_in_gate=True, the controller uses
        the options ONLY to evaluate whether the retrieved
        evidence can discriminate among candidate answers.

        The controller must NOT produce the final answer.
    """

    if gate_features is None:
        gate_features = [
            "confidence",
            "answerable",
        ]

    # --------------------------------------------------------
    # Safety: options automatically enable discrimination
    # --------------------------------------------------------

    if use_options_in_gate:

        if "discrimination" not in gate_features:
            gate_features = list(gate_features)
            gate_features.append(
                "discrimination"
            )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not context or not context.strip():

        return (
            False,
            0.0,
            False,
            "No retrieved evidence available."
        )

    # ========================================================
    # OPTIONS
    # ========================================================

    option_text = ""

    if use_options_in_gate and options:

        option_lines = []

        for key in sorted(options.keys()):

            option_lines.append(
                f"{key}. {options[key]}"
            )

        option_text = "\n".join(
            option_lines
        )

    # ========================================================
    # BUILD SIGNAL TEXT
    # ========================================================

    signal_text = []

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    if "coverage" in gate_features:

        if coverage_stats is None:

            signal_text.append(
                "COVERAGE: unavailable"
            )

        else:

            core_coverage = coverage_stats.get(
                "core_coverage",
                0.0
            )

            weighted_coverage = coverage_stats.get(
                "weighted_coverage",
                0.0
            )

            requirements = []

            for item in coverage_stats.get(
                "coverage",
                []
            ):

                requirements.append(
                    f"- {item.get('id', '')}: "
                    f"{item.get('status', 'MISSING')} "
                    f"(confidence="
                    f"{float(item.get('confidence', 0.0)):.2f})"
                )

            signal_text.append(
                f"""
COVERAGE:

Core coverage:
{core_coverage:.2f}

Weighted coverage:
{weighted_coverage:.2f}

Requirement status:
{chr(10).join(requirements)}
"""
            )

    # --------------------------------------------------------
    # NOVELTY
    # --------------------------------------------------------

    if "novelty" in gate_features:

        if evidence_stats is None:

            signal_text.append(
                "SEMANTIC NOVELTY: unavailable"
            )

        else:

            signal_text.append(
                f"""
SEMANTIC NOVELTY:
{evidence_stats.get(
    "semantic_novelty",
    None
)}
"""
            )

    # --------------------------------------------------------
    # REDUNDANCY
    # --------------------------------------------------------

    if "redundancy" in gate_features:

        if evidence_stats is None:

            signal_text.append(
                "SEMANTIC REDUNDANCY: unavailable"
            )

        else:

            signal_text.append(
                f"""
SEMANTIC REDUNDANCY:
{evidence_stats.get(
    "semantic_redundancy",
    None
)}
"""
            )

    # --------------------------------------------------------
    # DISCRIMINATION
    # --------------------------------------------------------

    if "discrimination" in gate_features:

        if options:

            signal_text.append(
                f"""
OPTION DISCRIMINATION TARGET:

The downstream QA problem is multiple-choice.

Candidate options:

{option_text}

Evaluate whether the RETRIEVED EVIDENCE itself contains
enough information to distinguish the relevant candidate
from the alternatives.

Do NOT simply use your own medical knowledge.
"""
            )

        else:

            signal_text.append(
                "OPTION DISCRIMINATION: unavailable because options were not supplied."
            )

    signal_text = "\n".join(
        signal_text
    )

    # ========================================================
    # FEATURE INSTRUCTIONS
    # ========================================================

    feature_instructions = []

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if "confidence" in gate_features:

        feature_instructions.append(
            """
CONFIDENCE:

Estimate how strongly the RETRIEVED EVIDENCE supports a
reliable downstream answer.

Do not increase confidence merely because you personally
know the medical answer.
"""
        )

    # --------------------------------------------------------
    # ANSWERABILITY
    # --------------------------------------------------------

    if "answerable" in gate_features:

        feature_instructions.append(
            """
ANSWERABILITY:

Determine whether the current retrieved evidence is sufficient
for a downstream QA system to answer reliably without another
retrieval iteration.
"""
        )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    if "coverage" in gate_features:

        feature_instructions.append(
            """
REQUIREMENT COVERAGE:

Use the coverage information as evidence about whether the
required information has actually been retrieved.

A high confidence value must NOT override a missing or
partial high-importance requirement.

If an essential requirement is still missing, prefer CONTINUE.
"""
        )

    # --------------------------------------------------------
    # NOVELTY
    # --------------------------------------------------------

    if "novelty" in gate_features:

        feature_instructions.append(
            """
SEMANTIC NOVELTY:

Low novelty means the latest retrieval adds little information
that is meaningfully different from what was already retrieved.

Very low novelty can support STOP when the existing evidence
is otherwise sufficient.

High novelty means another retrieval direction may still be
bringing useful new information.
"""
        )

    # --------------------------------------------------------
    # REDUNDANCY
    # --------------------------------------------------------

    if "redundancy" in gate_features:

        feature_instructions.append(
            """
SEMANTIC REDUNDANCY:

High redundancy means recent retrieval is largely repeating
existing evidence.

High redundancy can support STOP when the evidence is already
sufficient.

High redundancy alone must NOT force STOP if an important
information gap remains.
"""
        )

    # --------------------------------------------------------
    # DISCRIMINATION
    # --------------------------------------------------------

    if "discrimination" in gate_features:

        feature_instructions.append(
            """
OPTION DISCRIMINATION:

When options are supplied, do NOT answer the question.

Instead determine whether the retrieved evidence is sufficient
to distinguish among the candidate options.

Example:

If the evidence explicitly supports one mechanism and does
not leave an important competing option unresolved, the evidence
may be sufficient.

If the evidence supports only a general topic but does not
distinguish the alternatives, return CONTINUE.

Do NOT stop merely because one option appears familiar or
because you know the medical answer from outside knowledge.

The evidence must support the discrimination.
"""
        )

    feature_instructions = "\n".join(
        feature_instructions
    )

    # ========================================================
    # PROMPT
    # ========================================================

    options_section = ""

    if use_options_in_gate and options:

        options_section = f"""
==================================================
ANSWER OPTIONS
==================================================

{option_text}

IMPORTANT:
These options are provided ONLY to evaluate whether the
retrieved evidence sufficiently discriminates among the
candidate answers.

Do NOT output the correct option.
Do NOT choose an answer.
Do NOT use an option as a shortcut for external medical
knowledge.
"""

    gate_prompt = f"""
You are a medical QA retrieval stopping controller.

Your ONLY job is to decide whether another retrieval iteration
is likely to materially improve the evidence available to a
downstream answerer.

You are NOT the final answerer.

==================================================
QUESTION
==================================================

{question}

{options_section}

==================================================
RETRIEVED EVIDENCE
==================================================

{context}

==================================================
ACTIVE GATE FEATURES
==================================================

{", ".join(gate_features)}

==================================================
ADDITIONAL SIGNALS
==================================================

{signal_text}

==================================================
RULES
==================================================

1. Use the question and retrieved evidence.
2. Do NOT produce the final answer.
3. Do NOT invent evidence.
4. Do NOT rely on external medical knowledge when the evidence
   does not contain the necessary information.
5. Decide whether additional retrieval would materially improve
   downstream reliability.
6. Be conservative about stopping.
7. If an important information gap remains, prefer CONTINUE.

{feature_instructions}

==================================================
STOP
==================================================

STOP means:

Additional retrieval is unlikely to materially improve the
reliability of the downstream answer.

==================================================
CONTINUE
==================================================

CONTINUE means:

Additional retrieval is likely to provide information that is
still missing, ambiguous, weakly supported, or necessary for
distinguishing competing interpretations.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Use exactly:

{{
    "answerable": true,
    "confidence": 90,
    "discrimination_sufficient": true,
    "stop": true,
    "reasoning": "Brief explanation."
}}

If option discrimination is not active, still return
"discrimination_sufficient": true or false based on whether
the field is relevant; it will only affect the final decision
when the feature is enabled.
"""

    # ========================================================
    # API CALL
    # ========================================================

    try:

        response = medrag_instance.generate(
            [
                {
                    "role": "user",
                    "content": gate_prompt
                }
            ]
        )

        result = parse_json_response(
            response
        )

        # ----------------------------------------------------
        # answerable
        # ----------------------------------------------------

        answerable_value = result.get(
            "answerable",
            False
        )

        if isinstance(
            answerable_value,
            str
        ):

            answerable = (
                answerable_value
                .strip()
                .lower()
                == "true"
            )

        else:

            answerable = bool(
                answerable_value
            )

        # ----------------------------------------------------
        # confidence
        # ----------------------------------------------------

        try:

            confidence = float(
                result.get(
                    "confidence",
                    0
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
                100.0,
                confidence
            )
        )

        # ----------------------------------------------------
        # discrimination
        # ----------------------------------------------------

        discrimination_value = result.get(
            "discrimination_sufficient",
            False
        )

        if isinstance(
            discrimination_value,
            str
        ):

            discrimination_sufficient = (
                discrimination_value
                .strip()
                .lower()
                == "true"
            )

        else:

            discrimination_sufficient = bool(
                discrimination_value
            )

        # ----------------------------------------------------
        # model stop
        # ----------------------------------------------------

        model_stop_value = result.get(
            "stop",
            False
        )

        if isinstance(
            model_stop_value,
            str
        ):

            model_stop = (
                model_stop_value
                .strip()
                .lower()
                == "true"
            )

        else:

            model_stop = bool(
                model_stop_value
            )

        # ----------------------------------------------------
        # Base stop decision
        # ----------------------------------------------------

        should_stop = (
            model_stop
            and answerable
            and confidence >= threshold
        )

        # ----------------------------------------------------
        # If discrimination is enabled,
        # require discrimination too.
        # ----------------------------------------------------

        if "discrimination" in gate_features:

            should_stop = (
                should_stop
                and discrimination_sufficient
            )

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------

        reasoning = str(
            result.get(
                "reasoning",
                ""
            )
        ).strip()

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        print(
            f"   [Gate Features]: "
            f"{gate_features}"
        )

        print(
            f"   [Options In Gate]: "
            f"{use_options_in_gate}"
        )

        print(
            f"   [Gate Answerable]: "
            f"{answerable}"
        )

        print(
            f"   [Gate Confidence]: "
            f"{confidence:.1f}%"
        )

        print(
            f"   [Gate Threshold]: "
            f"{threshold}%"
        )

        if "discrimination" in gate_features:

            print(
                f"   [Discrimination Sufficient]: "
                f"{discrimination_sufficient}"
            )

        print(
            f"   [Gate Model Stop]: "
            f"{model_stop}"
        )

        print(
            f"   [Gate Decision]: "
            f"{'STOP' if should_stop else 'CONTINUE'}"
        )

        print(
            f"   [Gate Reasoning]: "
            f"{reasoning}"
        )

        return (
            should_stop,
            confidence,
            answerable,
            reasoning
        )

    except Exception as e:

        print(
            f"   [Gate Error]: {e}"
        )

        print(
            "   Defaulting to CONTINUE."
        )

        return (
            False,
            0.0,
            False,
            f"Gate error: {e}"
        )

def choose_query_budget(
    confidence,
    answerable,
    iteration,
    max_iterations=3,
    coverage_stats=None,
    evidence_stats=None,
    query_budget_mode="confidence",
):
    """
    Determine the number of queries for the next iteration.

    Modes:

        confidence
            confidence-only controller.

        coverage
            Use coverage to influence the budget.

        hybrid
            Combine confidence + coverage + novelty/redundancy.

    """

    # --------------------------------------------------------
    # Stop-like condition
    # --------------------------------------------------------

    if answerable and confidence >= 85:

        return 0

    # --------------------------------------------------------
    # Confidence-only
    # --------------------------------------------------------

    if query_budget_mode == "confidence":

        if confidence < 60:
            return 4

        if confidence < 80:
            return 2

        return 1

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if query_budget_mode == "coverage":

        if coverage_stats is not None:

            weighted = coverage_stats.get(
                "weighted_coverage",
                0.0
            )

            try:
                weighted = float(
                    weighted
                )
            except (
                TypeError,
                ValueError
            ):
                weighted = 0.0

            if weighted < 0.40:
                return 4

            if weighted < 0.70:
                return 2

            return 1

        # fallback
        if confidence < 60:
            return 4

        if confidence < 80:
            return 2

        return 1

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    if query_budget_mode == "hybrid":

        coverage_score = 1.0

        novelty = None
        redundancy = None

        if coverage_stats is not None:

            try:

                coverage_score = float(
                    coverage_stats.get(
                        "weighted_coverage",
                        0.0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                coverage_score = 0.0

        if evidence_stats is not None:

            novelty = evidence_stats.get(
                "semantic_novelty"
            )

            redundancy = evidence_stats.get(
                "semantic_redundancy"
            )

            try:
                novelty = float(
                    novelty
                )
            except (
                TypeError,
                ValueError
            ):
                novelty = None

            try:
                redundancy = float(
                    redundancy
                )
            except (
                TypeError,
                ValueError
            ):
                redundancy = None

        # Very incomplete evidence
        if coverage_score < 0.40:

            if novelty is None or novelty > 0.05:
                return 4

            return 2

        # Partially complete evidence
        if coverage_score < 0.70:

            if (
                novelty is not None
                and novelty > 0.10
            ):
                return 3

            return 2

        # Nearly complete evidence
        if coverage_score < 0.90:

            if (
                redundancy is not None
                and redundancy > 0.90
            ):
                return 1

            return 2

        # Very high coverage
        return 1

    # --------------------------------------------------------
    # Unknown mode
    # --------------------------------------------------------

    raise ValueError(
        f"Unknown query_budget_mode: "
        f"{query_budget_mode}"
    )


class AdaptiveMedRAG(MedRAG):

    def adaptive_answer(
        self,
        question,
        options=None,
        k=32,
        rrf_k=100,
        save_path=None,
        max_iterations=4,
        initial_queries=2,
        confidence_threshold=85,

        # ----------------------------------------------------
        # Planner
        # ----------------------------------------------------

        use_planner=False,
        use_planner_budget=False,

        # ----------------------------------------------------
        # Evidence analysis
        # ----------------------------------------------------

        use_coverage=False,
        use_novelty=False,

        # ----------------------------------------------------
        # Controller
        # ----------------------------------------------------

        gate_features=None,
        use_options_in_gate=False,
        query_budget_mode="confidence",

        **kwargs
    ):
        """
        Adaptive i-MedRAG.

        Main experimental switches:

            use_planner
            use_planner_budget
            use_coverage
            use_novelty
            use_options_in_gate

        Gate feature configurations can be:

            ["confidence", "answerable"]

            ["confidence", "answerable",
             "coverage"]

            ["confidence", "answerable",
             "novelty", "redundancy"]

            ["confidence", "answerable",
             "coverage", "novelty",
             "redundancy"]

        When use_options_in_gate=True, the controller receives
        the answer options and additionally evaluates whether
        the evidence distinguishes the candidate answers.
        """

        # ======================================================
        # DEFAULT GATE FEATURES
        # ======================================================

        if gate_features is None:

            gate_features = [
                "confidence",
                "answerable"
            ]

        else:

            gate_features = list(
                gate_features
            )

        # ======================================================
        # QUESTION PROMPT
        # ======================================================

        if options is not None:

            options_str = "\n".join(
                [
                    key + ". " + options[key]
                    for key in sorted(options.keys())
                ]
            )

        else:

            options_str = ""

        QUESTION_PROMPT = (
            f"Here is the question:\n"
            f"{question}\n\n"
            f"{options_str}"
        )

        # ======================================================
        # STATE
        # ======================================================

        planner = None
        coverage_analyzer = None
        evidence_tracker = None

        question_plan = None

        observed_entities = []
        constraints = []
        hypotheses = []
        relationships = []
        information_needs = []

        context = ""

        last_context = None

        saved_messages = [
            {
                "role": "system",
                "content":
                    self.templates[
                        "i_medrag_system"
                    ]
            }
        ]

        stop_flag = False

        confidence = 0.0
        answerable = False

        gate_reasoning = ""

        stopping_iteration = (
            max_iterations
        )

        coverage_history = []

        query_budget_history = []

        actual_query_count = 0

        successful_retrieval_count = 0

        failed_retrieval_count = 0

        # ======================================================
        # PLANNER
        # ======================================================

        if use_planner:

            planner = QuestionPlanner(
                self
            )

            question_plan = planner.plan(
                question
            )

            observed_entities = (
                question_plan[
                    "observed_entities"
                ]
            )

            constraints = (
                question_plan[
                    "constraints"
                ]
            )

            hypotheses = (
                question_plan[
                    "hypotheses"
                ]
            )

            relationships = (
                question_plan[
                    "relationships_to_establish"
                ]
            )

            information_needs = (
                question_plan[
                    "information_needs"
                ]
            )

            retrieval_need = (
                question_plan[
                    "retrieval_need"
                ]
            )

            print(
                f"\n[Planner Retrieval Need]: "
                f"{retrieval_need}"
            )

            # --------------------------------------------------
            # NO_RAG
            # --------------------------------------------------

            if retrieval_need == "NO_RAG":

                print(
                    "\n[Planner Decision] "
                    "NO_RAG -> skipping retrieval."
                )

                # Direct answer path
                messages = [
                    {
                        "role": "system",
                        "content":
                            self.templates[
                                "i_medrag_system"
                            ]
                    },
                    {
                        "role": "user",
                        "content":
                            f"{QUESTION_PROMPT}\n\n"
                            f"{self.templates['follow_up_answer']}"
                    }
                ]

                saved_messages.extend(
                    messages[1:]
                )

                answer_response = self.generate(
                    messages,
                    **kwargs
                )

                response_message = {
                    "role": "assistant",
                    "content": answer_response
                }

                saved_messages.append(
                    response_message
                )

                messages.append(
                    response_message
                )

                # Final JSON extraction
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return ONLY valid JSON. "
                            'Return exactly this format: {"answer":""}. '
                            "The answer must be one of A, B, C, D, E, or F. "
                            "Do not include explanation."
                        )
                    }
                )

                saved_messages.append(
                    messages[-1]
                )

                final_answer = self.generate(
                    messages,
                    **kwargs
                )

                saved_messages.append(
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                )

                return (
                    final_answer,
                    saved_messages,
                    0,
                    0.0,
                    [],
                    0,
                    0,
                    0,
                    [],
                    [],
                    {
                        "use_planner":
                            use_planner,
                        "use_planner_budget":
                            use_planner_budget,
                        "use_coverage":
                            use_coverage,
                        "use_novelty":
                            use_novelty,
                        "use_options_in_gate":
                            use_options_in_gate,
                        "query_budget_mode":
                            query_budget_mode,
                        "gate_features":
                            gate_features
                    }
                )

            # --------------------------------------------------
            # PRINT SCHEMA
            # --------------------------------------------------

            print("\n[Question Plan]")

            print(
                f"   Intent: "
                f"{question_plan['intent']}"
            )

            print(
                f"   Question Target: "
                f"{question_plan['question_target']}"
            )

            print(
                f"   Retrieval Need: "
                f"{retrieval_need}"
            )

            print(
                f"   Complexity: "
                f"{question_plan['complexity']:.2f}"
            )

            print(
                f"   Planner Initial Budget: "
                f"{question_plan['initial_query_budget']}"
            )

            print(
                "\n   Information Needs:"
            )

            for item in information_needs:

                print(
                    f"      {item['id']}: "
                    f"{item['description']} "
                    f"(importance="
                    f"{item['importance']:.2f})"
                )

        # ======================================================
        # INITIAL QUERY BUDGET
        # ======================================================

        if (
            use_planner
            and use_planner_budget
        ):

            query_budget = (
                question_plan[
                    "initial_query_budget"
                ]
            )

        else:

            query_budget = initial_queries

        print(
            f"\n   [Initial Query Budget]: "
            f"{query_budget}"
        )

        # ======================================================
        # COVERAGE ANALYZER
        # ======================================================

        if use_coverage:

            if not use_planner:

                raise ValueError(
                    "use_coverage=True requires "
                    "use_planner=True because "
                    "coverage uses information needs."
                )

            coverage_analyzer = (
                EvidenceCoverageAnalyzer(
                    self
                )
            )

        # ======================================================
        # EVIDENCE TRACKER
        # ======================================================

        if use_novelty:

            evidence_tracker = (
                EvidenceTracker(
                    self.retrieval_system
                )
            )

        # ======================================================
        # MAIN LOOP
        # ======================================================

        # Hard safety limit
        for i in range(
            max_iterations + 1
        ):

            # --------------------------------------------------
            # RETRIEVAL ROUND
            # --------------------------------------------------

            if (
                i < max_iterations
                and not stop_flag
            ):

                query_budget_history.append(
                    query_budget
                )

                print(
                    f"\n   [Query Budget]: "
                    f"{query_budget}"
                )

                if not context:

                    messages = [
                        {
                            "role": "system",
                            "content":
                                self.templates[
                                    "i_medrag_system"
                                ]
                        },
                        {
                            "role": "user",
                            "content":
                                (
                                    f"{QUESTION_PROMPT}\n\n"
                                    f"{self.templates['follow_up_ask'].format(query_budget)}"
                                )
                        }
                    ]

                else:

                    messages = [
                        {
                            "role": "system",
                            "content":
                                self.templates[
                                    "i_medrag_system"
                                ]
                        },
                        {
                            "role": "user",
                            "content":
                                (
                                    f"{context}\n\n"
                                    f"{QUESTION_PROMPT}\n\n"
                                    f"{self.templates['follow_up_ask'].format(query_budget)}"
                                )
                        }
                    ]

            # --------------------------------------------------
            # FINAL ANSWER ROUND
            # --------------------------------------------------

            elif context != last_context:

                messages = [
                    {
                        "role": "system",
                        "content":
                            self.templates[
                                "i_medrag_system"
                            ]
                    },
                    {
                        "role": "user",
                        "content":
                            (
                                f"{context}\n\n"
                                f"{QUESTION_PROMPT}\n\n"
                                f"{self.templates['follow_up_answer']}"
                            )
                    }
                ]

            else:

                # Emergency final answer
                messages = [
                    {
                        "role": "system",
                        "content":
                            self.templates[
                                "i_medrag_system"
                            ]
                    },
                    {
                        "role": "user",
                        "content":
                            (
                                f"{context}\n\n"
                                f"{QUESTION_PROMPT}\n\n"
                                f"{self.templates['follow_up_answer']}"
                            )
                    }
                ]

            saved_messages.append(
                messages[-1]
            )

            last_context = context

            # --------------------------------------------------
            # MODEL CALL
            # --------------------------------------------------

            last_content = self.generate(
                messages,
                **kwargs
            )

            response_message = {
                "role": "assistant",
                "content": last_content
            }

            saved_messages.append(
                response_message
            )

            # ==================================================
            # FINAL ANSWER
            # ==================================================

            if (
                (i >= max_iterations
                 or stop_flag)
                and
                (
                    "## Answer" in last_content
                    or
                    "answer is"
                    in last_content.lower()
                )
            ):

                messages.append(
                    response_message
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return ONLY valid JSON. "
                            'Return exactly this format: {"answer":""} '
                            "where the answer must be one of A, B, C, D, E, or F. "
                            "Do not use single quotes. "
                            "Do not include explanation."
                        )
                    }
                )

                saved_messages.append(
                    messages[-1]
                )

                answer_content = self.generate(
                    messages,
                    **kwargs
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content":
                            answer_content
                    }
                )

                saved_messages.append(
                    messages[-1]
                )

                return (
                    messages[-1]["content"],
                    saved_messages,
                    stopping_iteration,
                    confidence,
                    query_budget_history,
                    actual_query_count,
                    successful_retrieval_count,
                    failed_retrieval_count,
                    evidence_tracker.history
                    if evidence_tracker
                    else [],
                    coverage_history,
                    {
                        "use_planner":
                            use_planner,
                        "use_planner_budget":
                            use_planner_budget,
                        "use_coverage":
                            use_coverage,
                        "use_novelty":
                            use_novelty,
                        "use_options_in_gate":
                            use_options_in_gate,
                        "query_budget_mode":
                            query_budget_mode,
                        "gate_features":
                            gate_features
                    }
                )

            # ==================================================
            # QUERY EXTRACTION
            # ==================================================

            elif "## Queries" in last_content:

                messages = messages[:-1]

                try:

                    parser_prompt = (
                        "Parse the following passage and extract "
                        "the queries as a list.\n\n"
                        f"{last_content}\n\n"
                        "Do not merge or split queries. "
                        "Return ONLY valid JSON using exactly:\n"
                        '{"output":["query 1","query 2"]}'
                    )

                    action_response = self.generate(
                        [
                            {
                                "role": "user",
                                "content":
                                    parser_prompt
                            }
                        ],
                        **kwargs
                    )

                    parsed = parse_json_response(
                        action_response
                    )

                    action_list = parsed.get(
                        "output",
                        []
                    )

                    if not isinstance(
                        action_list,
                        list
                    ):
                        raise ValueError(
                            "Parsed query output is not a list."
                        )

                    action_list = [
                        str(q).strip()
                        for q in action_list
                        if str(q).strip()
                    ]

                    actual_query_count += (
                        len(action_list)
                    )

                    print(
                        f"   [Actual Queries Generated]: "
                        f"{len(action_list)}"
                    )

                except Exception as e:

                    print(
                        f"Error parsing action list: {e}"
                    )

                    # Continue safely
                    continue

                # ==================================================
                # RETRIEVAL
                # ==================================================

                iteration_doc_ids = []

                for q in action_list:

                    try:

                        rag_result, retrieved_snippets, scores = (
                            self.medrag_answer(
                                q,
                                k=k,
                                rrf_k=rrf_k,
                                **kwargs
                            )
                        )

                        successful_retrieval_count += 1

                        for doc in retrieved_snippets:

                            if "id" in doc:

                                iteration_doc_ids.append(
                                    doc["id"]
                                )

                        context += (
                            f"\n\nQuery: {q}\n"
                            f"Answer: {rag_result}"
                        )

                        context = context.strip()

                    except Exception as e:

                        failed_retrieval_count += 1

                        print(
                            f"Retrieval Error: {e}"
                        )

                # ==================================================
                # EVIDENCE SIGNALS
                # ==================================================

                evidence_stats = {
                    "document_novelty": None,
                    "semantic_novelty": None,
                    "semantic_redundancy": None
                }

                # --------------------------------------------------
                # Novelty
                # --------------------------------------------------

                if use_novelty:

                    try:

                        evidence_stats = (
                            evidence_tracker.update(
                                iteration_doc_ids
                            )
                        )

                        print(
                            f"   [Document Novelty]: "
                            f"{evidence_stats['document_novelty']}"
                        )

                        print(
                            f"   [Semantic Novelty]: "
                            f"{evidence_stats['semantic_novelty']}"
                        )

                        print(
                            f"   [Semantic Redundancy]: "
                            f"{evidence_stats['semantic_redundancy']}"
                        )

                    except Exception as e:

                        print(
                            f"[Evidence Tracker Error]: "
                            f"{e}"
                        )

                # --------------------------------------------------
                # Coverage
                # --------------------------------------------------

                coverage_stats = {
                    "coverage": [],
                    "constraint_coverage": [],
                    "core_coverage": 0.0,
                    "weighted_coverage": 0.0,
                    "reasoning": ""
                }

                if use_coverage:

                    try:

                        coverage_stats = (
                            coverage_analyzer.analyze(
                                question=question,
                                question_target=(
                                    question_plan.get(
                                        "question_target",
                                        ""
                                    )
                                ),
                                information_needs=(
                                    information_needs
                                ),
                                constraints=(
                                    constraints
                                ),
                                relationships=(
                                    relationships
                                ),
                                context=context
                            )
                        )

                    except TypeError:

                        # Compatibility with your current
                        # coverage module.
                        try:

                            coverage_stats = (
                                coverage_analyzer.analyze(
                                    question=question,
                                    core_requirements=(
                                        information_needs
                                    ),
                                    supporting_constraints=(
                                        constraints
                                    ),
                                    context=context
                                )
                            )

                        except Exception as e:

                            print(
                                f"[Coverage Error]: {e}"
                            )

                    except Exception as e:

                        print(
                            f"[Coverage Error]: {e}"
                        )

                    coverage_history.append(
                        coverage_stats
                    )

                    print(
                        f"\n   [Core Coverage]: "
                        f"{coverage_stats.get('core_coverage', 0.0):.2f}"
                    )

                    print(
                        f"   [Weighted Coverage]: "
                        f"{coverage_stats.get('weighted_coverage', 0.0):.2f}"
                    )

                    print(
                        "   [Information Need Coverage]:"
                    )

                    for item in coverage_stats.get(
                        "coverage",
                        []
                    ):

                        print(
                            f"      "
                            f"{item.get('id', '')} | "
                            f"{item.get('status', '')} | "
                            f"{float(item.get('confidence', 0.0)):.2f}"
                        )

                # ==================================================
                # STOPPING GATE
                # ==================================================

                print(
                    f"\n--- Checking Gate after "
                    f"Iteration {i + 1} ---"
                )

                (
                    stop_flag,
                    confidence,
                    answerable,
                    gate_reasoning
                ) = check_stopping_gate(
                    self,
                    question,
                    context,
                    threshold=confidence_threshold,
                    gate_features=gate_features,
                    coverage_stats=coverage_stats,
                    evidence_stats=evidence_stats,
                    options=options,
                    use_options_in_gate=(
                        use_options_in_gate
                    )
                )

                # --------------------------------------------------
                # STOP
                # --------------------------------------------------

                if stop_flag:

                    stopping_iteration = (
                        i + 1
                    )

                    query_budget = 0

                # --------------------------------------------------
                # CONTINUE
                # --------------------------------------------------

                else:

                    query_budget = (
                        choose_query_budget(
                            confidence=confidence,
                            answerable=answerable,
                            iteration=i + 1,
                            max_iterations=max_iterations,
                            coverage_stats=(
                                coverage_stats
                                if use_coverage
                                else None
                            ),
                            evidence_stats=(
                                evidence_stats
                                if use_novelty
                                else None
                            ),
                            query_budget_mode=(
                                query_budget_mode
                            )
                        )
                    )

                    print(
                        f"   [Next Query Budget]: "
                        f"{query_budget}"
                    )

        # ======================================================
        # FINAL FAILSAFE
        # ======================================================

        return (
            messages[-1]["content"],
            saved_messages,
            max_iterations,
            confidence,
            query_budget_history,
            actual_query_count,
            successful_retrieval_count,
            failed_retrieval_count,
            evidence_tracker.history
            if evidence_tracker
            else [],
            coverage_history,
            {
                "use_planner":
                    use_planner,
                "use_planner_budget":
                    use_planner_budget,
                "use_coverage":
                    use_coverage,
                "use_novelty":
                    use_novelty,
                "use_options_in_gate":
                    use_options_in_gate,
                "query_budget_mode":
                    query_budget_mode,
                "gate_features":
                    gate_features
            }
        )