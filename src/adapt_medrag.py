from .medrag import MedRAG

import json
import time
import re

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

# ============================================================
# CONTROLLER UTILITIES
# ============================================================

def clamp01(value, default=0.5):
    """
    Safely convert a value to [0, 1].
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return max(0.0, min(1.0, value))


def confidence_to_01(confidence):
    """
    Convert confidence from 0-100 to 0-1.
    """

    try:
        return clamp01(
            float(confidence) / 100.0
        )
    except (TypeError, ValueError):
        return 0.5


# ============================================================
# DETERMINISTIC QUERY-BUDGET CONTROLLER
# ============================================================
def choose_query_budget(
    confidence,
    answerable,
    discrimination,
    novelty=None,
    iteration=1,
    max_iterations=3,
    has_options=True,
):
    """
    Stage-1 confidence controller.

    STOP when evidence is judged sufficient.
    """

    if iteration >= max_iterations:
        return 0

    c = confidence / 100.0
    d = 1.0 if discrimination else 0.0

    # --------------------------------------------
    # STOP
    # --------------------------------------------

    if answerable and c >= 0.80:
        if not has_options or d >= 0.70:
            return 0

    # --------------------------------------------
    # CONTINUE
    # --------------------------------------------

    if not answerable:
        return 2

    if has_options and d < 0.70:
        return 2

    if c < 0.50:
        return 4

    if c < 0.70:
        return 2

    return 1

def check_stopping_gate(
    medrag_instance,
    question,
    context,
    gate_features=None,
    coverage_stats=None,
    evidence_stats=None,
    options=None,
    use_options_in_gate=False,
):
    """
    LLM-based SIGNAL ESTIMATOR.

    IMPORTANT:
        This function does NOT decide STOP / CONTINUE.

    The LLM only estimates:
        - confidence
        - answerability
        - option discrimination

    Python decides the actual retrieval budget afterward.

    Active features:
        confidence
        answerable
        coverage
        novelty
        redundancy
        discrimination
    """

    if gate_features is None:
        gate_features = [
            "confidence",
            "answerable"
        ]

    # --------------------------------------------------------
    # Automatically enable discrimination when options exist
    # --------------------------------------------------------

    if use_options_in_gate and "discrimination" not in gate_features:
        gate_features = list(gate_features)
        gate_features.append("discrimination")

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not context or not context.strip():
        return (
            0.0,      # confidence
            False,    # answerable
            False,    # discrimination
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

        option_text = "\n".join(option_lines)

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

                try:
                    item_confidence = float(
                        item.get("confidence", 0.0)
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    item_confidence = 0.0

                requirements.append(
                    f"- {item.get('id', '')}: "
                    f"{item.get('status', 'MISSING')} "
                    f"(confidence={item_confidence:.2f})"
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

        if use_options_in_gate and options:

            signal_text.append(
                f"""
OPTION DISCRIMINATION:

The downstream task is multiple-choice.

Candidate options:

{option_text}

Evaluate ONLY whether the retrieved evidence contains
enough information to distinguish among the candidates.

Do NOT answer the question.
Do NOT identify the correct option.
Do NOT rely on outside medical knowledge.
"""
            )

        else:

            signal_text.append(
                "OPTION DISCRIMINATION: unavailable."
            )

    signal_text = "\n".join(signal_text)

    # ========================================================
    # FEATURE INSTRUCTIONS
    # ========================================================

    feature_instructions = []

    if "confidence" in gate_features:

        feature_instructions.append(
            """
CONFIDENCE:

Estimate how strongly the RETRIEVED EVIDENCE supports
a reliable downstream answer.

This is evidence confidence, not confidence from your
own medical knowledge.

If the evidence is incomplete or indirect, lower the score.
"""
        )

    if "answerable" in gate_features:

        feature_instructions.append(
            """
ANSWERABILITY:

Determine whether the currently retrieved evidence contains
enough information for a downstream QA system to answer
reliably without another retrieval iteration.

Answerable means the evidence is sufficient.

It does NOT mean that you personally know the answer.
"""
        )

    if "coverage" in gate_features:

        feature_instructions.append(
            """
COVERAGE:

Use the supplied coverage information.

Missing or partial important requirements indicate that
important evidence may still be absent.
"""
        )

    if "novelty" in gate_features:

        feature_instructions.append(
            """
NOVELTY:

Low novelty means the newest retrieved evidence is largely
repeating previously retrieved information.

High novelty means the newest retrieval adds substantially
different information.

Do not interpret high novelty as bad evidence.
It indicates that retrieval is still finding new material.
"""
        )

    if "redundancy" in gate_features:

        feature_instructions.append(
            """
REDUNDANCY:

High redundancy means recent retrieval overlaps strongly
with already retrieved evidence.

Low redundancy means the latest retrieval is more distinct.
"""
        )

    if "discrimination" in gate_features:

        feature_instructions.append(
            """
DISCRIMINATION:

When candidate options are supplied, determine whether the
retrieved evidence itself is sufficient to distinguish the
candidates.

Do NOT choose the correct option.

For example:

Strong discrimination:
The evidence explicitly supports one candidate mechanism,
organism, diagnosis, or treatment while ruling out important
alternatives.

Weak discrimination:
The evidence discusses only the general topic and does not
allow reliable separation of the candidates.
"""
        )

    feature_instructions = "\n".join(
        feature_instructions
    )

    # ========================================================
    # OPTIONS SECTION
    # ========================================================

    options_section = ""

    if use_options_in_gate and options:

        options_section = f"""
==================================================
ANSWER OPTIONS
==================================================

{option_text}

These options are provided ONLY to measure evidence
discrimination.

Do NOT output the correct answer.
Do NOT rank the options.
Do NOT use an option as a substitute for missing evidence.
"""

    # ========================================================
    # PROMPT
    # ========================================================

    gate_prompt = f"""
You are an evidence assessment module inside an adaptive
medical retrieval system.

Your role is NOT to control retrieval directly.

Your role is ONLY to estimate the current state of the
retrieved evidence.

The Python controller will use your measurements to decide
whether more retrieval is necessary.

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
ACTIVE FEATURES
==================================================

{", ".join(gate_features)}

==================================================
ADDITIONAL SIGNALS
==================================================

{signal_text}

==================================================
INSTRUCTIONS
==================================================

1. Judge the RETRIEVED EVIDENCE.
2. Use the question as context.
3. Do NOT use outside medical knowledge to fill evidence gaps.
4. Do NOT produce the final answer.
5. Do NOT select an answer option.
6. Do NOT decide the retrieval budget.
7. Be conservative.
8. If the evidence is indirect, incomplete, contradictory,
   or insufficient, lower confidence.

{feature_instructions}

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Exactly:

{{
    "answerable": true,
    "confidence": 85,
    "discrimination_sufficient": true,
    "reasoning": "Brief explanation."
}}

CONFIDENCE:
0 = no reliable support
100 = very strong direct support

DISCRIMINATION_SUFFICIENT:
true = evidence can reliably distinguish the candidates
false = evidence cannot reliably distinguish the candidates

Do NOT include:
- stop
- query budget
- final answer
- option letter
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

        if response is None:
            raise ValueError(
                "Gate returned None."
            )

        response = str(
            response
        ).strip()

        result = parse_json_response(
            response
        )

        # ====================================================
        # ANSWERABILITY
        # ====================================================

        answerable_value = result.get(
            "answerable",
            False
        )

        if isinstance(
            answerable_value,
            str
        ):

            answerable = (
                answerable_value.strip().lower()
                == "true"
            )

        else:

            answerable = bool(
                answerable_value
            )

        # ====================================================
        # CONFIDENCE
        # ====================================================

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

        # ====================================================
        # DISCRIMINATION
        # ====================================================

        discrimination_value = result.get(
            "discrimination_sufficient",
            False
        )

        if isinstance(
            discrimination_value,
            str
        ):

            discrimination_sufficient = (
                discrimination_value.strip().lower()
                == "true"
            )

        else:

            discrimination_sufficient = bool(
                discrimination_value
            )

        # ====================================================
        # REASONING
        # ====================================================

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

        if "discrimination" in gate_features:

            print(
                f"   [Discrimination Sufficient]: "
                f"{discrimination_sufficient}"
            )

        print(
            f"   [Gate Reasoning]: "
            f"{reasoning}"
        )

        return (
            confidence,
            answerable,
            discrimination_sufficient,
            reasoning
        )

    except Exception as e:

        print(
            f"   [Gate Error]: {e}"
        )

        print(
            "   Returning conservative evidence signals."
        )

        return (
            0.0,
            False,
            False,
            f"Gate error: {e}"
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
        start_time = time.perf_counter()
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
        confidence_history = []
        confidence_history = []
        novelty_history = []
        discrimination_history = []
        query_budget_history = []
        gate_reasoning_history = []

        actual_query_count = 0
        successful_retrieval_count = 0
        failed_retrieval_count = 0

        self.llm_call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_history = []

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
                latency_seconds = (
                    time.perf_counter() - start_time
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
                            gate_features,
                    
                        "confidence_history":
                            confidence_history,

                        "novelty_history":
                            novelty_history,

                        "discrimination_history":
                            discrimination_history,

                        "query_budget_history":
                            query_budget_history,
                        "llm_calls": self.llm_call_count,
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": self.completion_tokens,
                        "total_tokens": self.total_tokens,
                        "latency_seconds": latency_seconds,
                                                            
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
                            f"Retrieval Error: {e}")
                
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
                    confidence,
                    answerable,
                    discrimination_sufficient,
                    gate_reasoning
                ) = check_stopping_gate(
                    self,
                    question,
                    context,
                    gate_features=gate_features,
                    coverage_stats=coverage_stats,
                    evidence_stats=evidence_stats,
                    options=options,
                    use_options_in_gate=use_options_in_gate
                )
                

                confidence_history.append(confidence)

                discrimination_history.append(
                    discrimination_sufficient
                )

                novelty_history.append(
                    evidence_stats.get("semantic_novelty")
                    if evidence_stats
                    else None
                )

                gate_reasoning_history.append(
                    gate_reasoning
                )

            query_budget = choose_query_budget(
                confidence=confidence,
                answerable=answerable,
                discrimination=discrimination_sufficient,
                novelty=(
                    evidence_stats.get("semantic_novelty")
                    if evidence_stats
                    else None
                ),
                iteration=i + 1,
                max_iterations=max_iterations,
                has_options=(options is not None)
            )

            stop_flag = (
                query_budget == 0
            )

            if stop_flag:

                stopping_iteration = (
                    i + 1
                )

            print(
                f"   [Next Query Budget]: "
                f"{query_budget}"
            )
        # ======================================================
        # FINAL FAILSAFE
        # ======================================================
        latency_seconds = (
            time.perf_counter() - start_time
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
                    gate_features,

                "confidence_history":
                    confidence_history,

                "novelty_history":
                    novelty_history,

                "discrimination_history":
                    discrimination_history,

                "query_budget_history":
                    query_budget_history,

                "llm_calls": self.llm_call_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "latency_seconds": latency_seconds,
                },
            
        )