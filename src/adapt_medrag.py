from .medrag import MedRAG

import json
import time
import re

from .evidence_tracker import EvidenceTracker
from .evidence_coverage import EvidenceCoverageAnalyzer
from .question_planner import QuestionPlanner
from .Q_network import TrainedStopRAGController




def parse_json_response(response):
    """
    Robustly extract a JSON object from an LLM response.
    """

    if response is None:
        raise ValueError("LLM returned None.")

    text = str(response).strip()

    if not text:
        raise ValueError("LLM returned empty response.")

    
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

    
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    
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
    Deterministic adaptive query-budget controller.

    Novelty is used as an information-gain signal:

        low novelty  -> new retrieval is becoming repetitive
        high novelty -> retrieval is still finding new information

    Returns:
        0 = STOP
        1 = retrieve 1 more query
        2 = retrieve 2 more queries
        4 = retrieve 4 more queries
    """

    if iteration >= max_iterations:
        return 0


    c = max(0.0, min(1.0, confidence / 100.0))
    d = 1.0 if discrimination else 0.0

    novelty_available = novelty is not None

    if novelty_available:
        try:
            n = float(novelty)
            n = max(0.0, min(1.0, n))
        except (TypeError, ValueError):
            novelty_available = False
            n = 0.5
    else:
        n = 0.5


    
    if not answerable:
        # if c < 0.50:
        #     return 4
        return 2

    
    if has_options and d < 0.70:
        return 2

    
    if c >= 0.8:

        
        if novelty_available:
            if n <= 0.30:
                return 0

            
            # return 1
            return 2

        
        return 0

    

    if c < 0.50:
        # return 4
        return 2

    

    if c < 0.70:

        
        if novelty_available and n > 0.50:
            return 2

        return 2

    

    
    if novelty_available and n > 0.50:
        return 2

    
    return 2

def check_stopping_gate(
    medrag_instance,
    question,
    context,
    gate_features=None,
    coverage_stats=None,
    evidence_stats=None,
    options=None,
    use_options_in_gate=False,
    pre_retrieval=False,
):
    

    

    if gate_features is None:

        gate_features = [
            "confidence",
            "answerable"
        ]

    else:

        gate_features = list(
            gate_features
        )

   

    if (
        use_options_in_gate
        and
        "discrimination" not in gate_features
    ):

        gate_features.append(
            "discrimination"
        )

   

    option_text = ""

    if (
        options is not None
        and
        use_options_in_gate
    ):

        option_lines = []

        for key in sorted(
            options.keys()
        ):

            option_lines.append(
                f"{key}. {options[key]}"
            )

        option_text = "\n".join(
            option_lines
        )


    signal_text = []

  

    if pre_retrieval:

        signal_text.append(
            """
    For PRE-RETRIEVAL assessment, you MAY use the model's pretrained
    medical knowledge to judge whether the answer choices can be
    distinguished reliably without retrieving new evidence.

    Do NOT provide the actual answer.
    Do NOT identify which option is correct.
    Do NOT rank the answer choices.

    Assess whether the available question information, answer choices,
    and pretrained knowledge are sufficient for reliable discrimination.
    """
        )



    else:

        

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
                            item.get(
                                "confidence",
                                0.0
                            )
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        item_confidence = 0.0

                    requirements.append(
                        f"- {item.get('id', '')}: "
                        f"{item.get('status', 'MISSING')} "
                        f"(confidence="
                        f"{item_confidence:.2f})"
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

   

    if "discrimination" in gate_features:

        if (
            use_options_in_gate
            and
            options
        ):

            signal_text.append(
                f"""
OPTION DISCRIMINATION:

The downstream task is multiple-choice.

Candidate options:

{option_text}

Evaluate ONLY whether the available information is
sufficient to distinguish among the candidates.

Do NOT answer the question.
Do NOT identify the correct option.
Do NOT rely on outside medical knowledge.
"""
            )

        else:

            signal_text.append(
                "OPTION DISCRIMINATION: unavailable."
            )

    signal_text = "\n".join(
        signal_text
    )


    feature_instructions = []


    if "confidence" in gate_features:

        if pre_retrieval:

            feature_instructions.append(
                """
CONFIDENCE:

Estimate how likely the question is to be answerable
reliably WITHOUT retrieving additional information.

Judge the apparent sufficiency of the question and answer
choices for a downstream QA system.

This is NOT confidence in a selected answer.

Do NOT solve the question.
Do NOT choose an option.

High confidence means retrieval is likely unnecessary.
Low confidence means retrieval may be useful.
"""
            )

        else:

            feature_instructions.append(
                """
CONFIDENCE:

Estimate how strongly the CURRENT RETRIEVED EVIDENCE supports
a reliable downstream answer.

This is evidence confidence, not confidence from your own
medical knowledge.

If the evidence is incomplete, indirect, ambiguous,
or contradictory, lower the score.
"""
            )



    if "answerable" in gate_features:

        if pre_retrieval:

            feature_instructions.append(
                """
ANSWERABILITY:

Determine whether a downstream medical QA system is likely
to answer the question reliably WITHOUT retrieval.

true:
The question is likely answerable from information already
available to the model.

false:
Additional retrieval is likely useful or necessary.

Do NOT actually answer the question.
"""
            )

        else:

            feature_instructions.append(
                """
ANSWERABILITY:

Determine whether the CURRENT RETRIEVED EVIDENCE contains
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
currently available information is sufficient to distinguish
the candidates.

Do NOT choose the correct option.

Strong discrimination:
The available information supports one candidate and/or
rules out important alternatives.

Weak discrimination:
The available information is too general, incomplete, or
ambiguous to reliably distinguish the candidates.
"""
        )

    feature_instructions = "\n".join(
        feature_instructions
    )



    options_section = ""

    if (
        use_options_in_gate
        and
        options
    ):

        options_section = f"""
==================================================
ANSWER OPTIONS
==================================================

{option_text}

These options are provided ONLY to assess information
sufficiency and discrimination.

Do NOT output the correct answer.
Do NOT rank the options.
Do NOT use an option as a substitute for missing evidence.
"""


    if pre_retrieval:

        information_section = """
==================================================
CURRENT INFORMATION
==================================================

No retrieved evidence is available.

This is the PRE-RETRIEVAL state.
"""

    else:

        information_section = f"""
==================================================
RETRIEVED EVIDENCE
==================================================

{context}
"""



    gate_prompt = f"""
You are an evidence assessment module inside an adaptive
medical retrieval system.

Your role is NOT to control retrieval directly.

Your role is ONLY to estimate the current information state.

The Python controller will use your measurements later to
decide whether additional retrieval is necessary.

==================================================
QUESTION
==================================================

{question}

{options_section}

{information_section}

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

1. Assess only the information currently available.
2. Use the question as context.
3. Do NOT use outside medical knowledge to compensate for
   missing information.
4. Do NOT produce the final answer.
5. Do NOT select an answer option.
6. Do NOT decide STOP or CONTINUE.
7. Be conservative.
8. If the available information is incomplete, indirect,
   contradictory, or insufficient, lower confidence.

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
0 = information is completely insufficient
100 = information is highly sufficient for a reliable answer

ANSWERABLE:
true = a downstream QA system is likely able to answer
reliably from the currently available information

false = additional information/retrieval is likely needed

DISCRIMINATION_SUFFICIENT:
true = the available information can reliably distinguish
the candidate answers

false = the available information cannot reliably distinguish
the candidate answers

Do NOT include:
- stop
- continue
- query budget
- final answer
- option letter as your answer
"""



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



        reasoning = str(
            result.get(
                "reasoning",
                ""
            )
        ).strip()



        print(
            f"   [Assessment Mode]: "
            f"{'PRE-RETRIEVAL' if pre_retrieval else 'EVIDENCE'}"
        )

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



        use_planner=False,
        use_planner_budget=False,


        use_coverage=False,
        use_novelty=False,



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


        if gate_features is None:

            gate_features = [
                "confidence",
                "answerable"
            ]

        else:

            gate_features = list(
                gate_features
            )


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


        if use_novelty:

            evidence_tracker = (
                EvidenceTracker(
                    self.retrieval_system
                )
            )


        PRE_RETRIEVAL_CONFIDENCE_THRESHOLD = 85.0

        pre_retrieval_assessment = check_stopping_gate(
            self,
            question,
            context="",
            gate_features=[
                "confidence",
                "answerable",
                "discrimination"
            ],
            options=options,
            use_options_in_gate=True,
            pre_retrieval=True
        )

        pre_confidence = pre_retrieval_assessment[0]
        pre_answerable = pre_retrieval_assessment[1]
        pre_discrimination = pre_retrieval_assessment[2]
        pre_reasoning = pre_retrieval_assessment[3]

        print("\n--- Pre-Retrieval Confidence Check ---")
        print(f"   Confidence: {pre_confidence}%")
        print(f"   Answerable: {pre_answerable}")
        print(f"   Discrimination: {pre_discrimination}")
        print(f"   Reasoning: {pre_reasoning}")

        if (
            pre_confidence is not None
            and pre_confidence >= PRE_RETRIEVAL_CONFIDENCE_THRESHOLD
            and pre_answerable
            and pre_discrimination
        ):
            print(
                f"[Confidence-Adaptive] "
                f"PRE-RETRIEVAL STOP "
                f"(confidence={pre_confidence}%)"
            )

            stopping_iteration = 0
            query_budget_history = []

            
        else:
            print(
                "[Confidence-Adaptive] "
                "Pre-retrieval confidence insufficient. "
                "Starting retrieval."
            )

           
        for i in range(
            max_iterations + 1
        ):

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

                    
                    continue

                

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
                
                

                evidence_stats = {
                    "document_novelty": None,
                    "semantic_novelty": None,
                    "semantic_redundancy": None
                }

                
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
    def set_stop_rag_controller(
        self,
        checkpoint_path="./stop_rag_ckpt_v2/best_stop_rag_qnetwork.pt",
        model_name="microsoft/deberta-v3-base",
        max_length=1024,
        device=None
    ):

        self.stop_rag_controller = (
            TrainedStopRAGController(
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                max_length=max_length,
                device=device
            )
        )

        print(
            "\n[Stop-RAG Controller] "
            "Loaded."
        )

    def trained_stop_rag_decision(
        self,
        question,
        options,
        context
    ):
        """
        Use the trained Q-network to decide whether
        another retrieval round is necessary.
        """

        if not hasattr(
            self,
            "stop_rag_controller"
        ):

            raise RuntimeError(
                "Stop-RAG controller has not been loaded. "
                "Call set_stop_rag_controller() first."
            )

        decision = (
            self.stop_rag_controller.decide(
                question=question,
                options=options,
                evidence=context
            )
        )

        print(
            "\n[Learned Stop-RAG]"
        )

        print(
            f"   Q(STOP): "
            f"{decision['q_stop']:.4f}"
        )

        print(
            f"   Q(CONTINUE): "
            f"{decision['q_continue']:.4f}"
        )

        print(
            f"   Margin: "
            f"{decision['margin']:.4f}"
        )

        print(
            f"   Decision: "
            f"{decision['action']}"
        )

        return decision
    def learned_stop_rag_answer(
        self,
        question,
        options=None,
        k=32,
        rrf_k=100,
        max_iterations=3,
        initial_queries=2,
        **kwargs
    ):

        start_time = time.perf_counter()

        

        self.llm_call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_history = []

        

        context = ""

        saved_messages = [
            {
                "role": "system",
                "content":
                    self.templates[
                        "i_medrag_system"
                    ]
            }
        ]

        query_budget_history = []

        q_history = []

        actual_query_count = 0
        successful_retrieval_count = 0
        failed_retrieval_count = 0

        stopping_iteration = (
            max_iterations
        )

        

        query_budget = initial_queries

        

        if options is not None:

            options_str = "\n".join(
                [
                    f"{key}. {options[key]}"
                    for key in sorted(
                        options.keys()
                    )
                ]
            )

        else:

            options_str = ""

        QUESTION_PROMPT = (
            f"Here is the question:\n"
            f"{question}\n\n"
            f"{options_str}"
        )

        

        for i in range(max_iterations + 1):

            print("\n")
            print("=" * 80)
            print(
                f"LEARNED STOP-RAG STATE {i}"
            )
            print("=" * 80)

            print(
                f"[Current evidence chars]: "
                f"{len(context)}"
            )

            

            decision = self.trained_stop_rag_decision(
                question=question,
                options=options,
                context=context
            )

            q_history.append(decision)

            print(
                f"\n[Learned Stop-RAG]"
            )
            print(
                f"   Q(STOP): "
                f"{decision['q_stop']:.4f}"
            )
            print(
                f"   Q(CONTINUE): "
                f"{decision['q_continue']:.4f}"
            )
            print(
                f"   Margin: "
                f"{decision['margin']:.4f}"
            )
            print(
                f"   Margin threshold: "
                f"{decision.get('margin_threshold', 0.0):.4f}"
            )
            print(
                f"   Decision: "
                f"{decision['action']}"
            )

            

            if decision["action"] == "STOP":

                print(
                    "[Stop-RAG] STOP selected."
                )

                stopping_iteration = i

                break

            

            if i >= max_iterations:

                print(
                    "[Stop-RAG] Maximum retrieval "
                    "iterations reached."
                )

                stopping_iteration = i

                break

            

            print(
                "[Stop-RAG] CONTINUE selected."
            )

            query_budget = initial_queries

            query_budget_history.append(
                query_budget
            )

            print(
                f"[Query Budget]: "
                f"{query_budget}"
            )

            

            if not context:

                user_content = (
                    f"{QUESTION_PROMPT}\n\n"
                    f"{self.templates['follow_up_ask'].format(query_budget)}"
                )

            else:

                user_content = (
                    f"{context}\n\n"
                    f"{QUESTION_PROMPT}\n\n"
                    f"{self.templates['follow_up_ask'].format(query_budget)}"
                )

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
                    "content": user_content
                }
            ]

            saved_messages.append(
                messages[-1]
            )

            

            query_response = self.generate(
                messages,
                **kwargs
            )

            saved_messages.append(
                {
                    "role": "assistant",
                    "content": query_response
                }
            )

            

            queries = []

            for line in query_response.splitlines():

                line = line.strip()

                
                match = re.match(
                    r"^##\s*Query\s*\d+\s*:\s*(.+?)\s*$",
                    line,
                    flags=re.IGNORECASE
                )

                if match:

                    query = match.group(1).strip()

                    if query:
                        queries.append(query)

            

            if not queries:

                fallback_patterns = [
                    r"^\s*Query\s*\d+\s*:\s*(.+?)\s*$",
                    r"^\s*\d+\.\s*(.+?)\s*$",
                    r"^\s*-\s*(.+?)\s*$",
                ]

                for pattern in fallback_patterns:

                    matches = re.findall(
                        pattern,
                        query_response,
                        flags=re.IGNORECASE |
                              re.MULTILINE
                    )

                    for match in matches:

                        query = match.strip()

                        if query:
                            queries.append(query)

                    if queries:
                        break

            

            queries = queries[:query_budget]

            print(
                f"[Queries generated]: "
                f"{len(queries)}"
            )

           

            if not queries:

                print(
                    "\n[ERROR] Query generation returned "
                    "zero queries."
                )

                print(
                    "\n[Raw query response]:"
                )

                print(query_response)

                raise RuntimeError(
                    "Stop-RAG decided CONTINUE, "
                    "but no retrieval queries were generated."
                )


            for query in queries:

                actual_query_count += 1

                print(
                    f"\n[Retrieving query "
                    f"{actual_query_count}]:"
                )

                print(
                    f"   {query}"
                )

                try:

                   
                    rag_result, retrieved_snippets, scores = (
                        self.medrag_answer(
                            query,
                            k=k,
                            rrf_k=rrf_k
                        )
                    )

                    if rag_result:

                        successful_retrieval_count += 1

                        
                        context += (
                            f"\n\n"
                            f"Query: {query}\n"
                            f"Answer: {rag_result}"
                        )

                    else:

                        failed_retrieval_count += 1

                        print(
                            "[Retrieval] No result returned."
                        )

                except Exception as e:

                    failed_retrieval_count += 1

                    print(
                        f"[Retrieval error]: {e}"
                    )

            

            context = context.strip()

            print(
                f"\n[Evidence chars after retrieval]: "
                f"{len(context)}"
            )

            print(
                f"[Successful retrievals]: "
                f"{successful_retrieval_count}"
            )

            print(
                f"[Failed retrievals]: "
                f"{failed_retrieval_count}"
            )


        

        final_messages = [
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
            final_messages[-1]
        )

        answer_response = self.generate(
            final_messages,
            **kwargs
        )

        saved_messages.append(
            {
                "role": "assistant",
                "content":
                    answer_response
            }
        )

        

        final_messages.append(
            {
                "role": "assistant",
                "content":
                    answer_response
            }
        )

        final_messages.append(
            {
                "role": "user",
                "content":
                    (
                        "Return ONLY valid JSON. "
                        'Return exactly this format: {"answer":""}. '
                        "The answer must be one of A, B, C, D, E, or F. "
                        "Do not include explanation."
                    )
            }
        )

        saved_messages.append(
            final_messages[-1]
        )

        final_answer = self.generate(
            final_messages,
            **kwargs
        )

        saved_messages.append(
            {
                "role": "assistant",
                "content":
                    final_answer
            }
        )

        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        

        controller_config = {

            "controller":
                "trained_stop_rag",

            "model":
                "microsoft/deberta-v3-base",

            "checkpoint":
                "./stop_rag_ckpt_v2/best_stop_rag_qnetwork.pt",

            "uses_llm_confidence":
                False,

            "uses_answerability":
                False,

            "uses_discrimination":
                False,

            "uses_retrieval_scores":
                False,

            "q_history":
                q_history,

            "llm_calls":
                self.llm_call_count,

            "prompt_tokens":
                self.prompt_tokens,

            "completion_tokens":
                self.completion_tokens,

            "total_tokens":
                self.total_tokens,

            "latency_seconds":
                latency_seconds
        }

        return (
            final_answer,
            saved_messages,
            stopping_iteration,
            query_budget_history,
            actual_query_count,
            successful_retrieval_count,
            failed_retrieval_count,
            q_history,
            controller_config
        )



    def collect_full_trajectory(
        self,
        question_id,
        question,
        options=None,
        k=10,
        rrf_k=100,
        n_rounds=3,
        n_queries=2,
        **kwargs
    ):
        """
        Collect one complete fixed-budget i-MedRAG trajectory.

        Stage 1 MDP dataset:

            Actions: STOP / CONTINUE
            Queries per iteration: fixed at n_queries
            Maximum iterations: n_rounds

        IMPORTANT:
            This method NEVER performs adaptive stopping.

        It is only for constructing the offline MDP dataset.
        """

        start_time = time.perf_counter()

        

        self.llm_call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_history = []

        

        evidence_tracker = EvidenceTracker(
            self.retrieval_system
        )

        

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
        
        
        

        context = ""

        saved_messages = [
            {
                "role": "system",
                "content": self.templates[
                    "i_medrag_system"
                ]
            }
        ]

        trajectory = []

        total_queries = 0
        successful_retrievals = 0
        failed_retrievals = 0

        accumulated_documents = []


        
        s0_assessment = check_stopping_gate(
            self,
            question,
            context="",
            gate_features=[
                "confidence",
                "answerable",
                "discrimination"
            ],
            options=options,
            use_options_in_gate=True,
            pre_retrieval=True
        )
        s0_state = {
            "state_id": f"{question_id}_S0",
            "iteration": 0,

            "question": question,
            "options": options,

            "queries": [],
            "requested_query_count": 0,
            "actual_query_count": 0,

            "confidence": s0_assessment[0],
            "answerable": s0_assessment[1],
            "discrimination": s0_assessment[2],
            "assessment_reasoning": s0_assessment[3],
            "assessment_type": "pre_retrieval",

            "retrieval_records": [],
            "iteration_documents": [],
            "iteration_document_ids": [],
            "accumulated_document_ids": [],

            "context": "",

            "evidence_stats": {
                "num_documents": 0,
                "num_new_documents": 0,
                "document_novelty": 0.0,
                "semantic_novelty": None,
                "semantic_redundancy": None
            },

            "novelty_available": False,

            "cost": {
                "queries_so_far": 0,
                "llm_calls_so_far": 0,
                "prompt_tokens_so_far": 0,
                "completion_tokens_so_far": 0,
                "total_tokens_so_far": 0,
                "latency_seconds_so_far": 0.0
            }
        }

        trajectory.append(s0_state)
        
        for iteration in range(
            1,
            n_rounds + 1
        ):

            iteration_start = time.perf_counter()

            print("\n")
            print("=" * 90)
            print(
                f"[MDP TRAJECTORY] "
                f"Iteration {iteration}/{n_rounds}"
            )
            print("=" * 90)

            

            query_budget = n_queries

            

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

            saved_messages.append(
                messages[-1]
            )

            
            query_generation_response = self.generate(
                messages,
                **kwargs
            )

            query_generation_message = {
                "role": "assistant",
                "content": query_generation_response
            }

            saved_messages.append(
                query_generation_message
            )

           

            parser_prompt = (
                "Parse the following passage and extract "
                "the queries as a list.\n\n"
                f"{query_generation_response}\n\n"
                "Do not merge or split queries. "
                "Return ONLY valid JSON using exactly:\n"
                '{"output":["query 1","query 2"]}'
            )

            try:

                parser_response = self.generate(
                    [
                        {
                            "role": "user",
                            "content": parser_prompt
                        }
                    ],
                    **kwargs
                )

                parsed = parse_json_response(
                    parser_response
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

            except Exception as e:

                print(
                    f"[Query Parsing Error]: {e}"
                )

                action_list = []

            

            action_list = action_list[
                :n_queries
            ]

            actual_generated_queries = len(
                action_list
            )

            print(
                f"[Queries Generated]: "
                f"{actual_generated_queries}"
            )

            total_queries += (
                actual_generated_queries
            )

            

            iteration_documents = []

            iteration_doc_ids = []

            retrieval_records = []

            for query in action_list:

                try:

                    (
                        rag_result,
                        retrieved_snippets,
                        scores
                    ) = self.medrag_answer(
                        query,
                        k=k,
                        rrf_k=rrf_k,
                        **kwargs
                    )

                    successful_retrievals += 1

                    query_record = {
                        "query": query,
                        "intermediate_answer":
                            rag_result,
                        "retrieved_documents": []
                    }

                    

                    for j, doc in enumerate(
                        retrieved_snippets
                    ):

                        doc_copy = dict(doc)

                        if j < len(scores):

                            doc_copy[
                                "retrieval_score"
                            ] = scores[j]

                        iteration_documents.append(
                            doc_copy
                        )

                        query_record[
                            "retrieved_documents"
                        ].append(
                            doc_copy
                        )

                        if "id" in doc:

                            iteration_doc_ids.append(
                                str(doc["id"])
                            )

                    retrieval_records.append(
                        query_record
                    )

                    

                    context += (
                        f"\n\nQuery: {query}\n"
                        f"Answer: {rag_result}"
                    )

                    context = context.strip()

                except Exception as e:

                    failed_retrievals += 1

                    retrieval_records.append({
                        "query": query,
                        "error": str(e),
                        "retrieved_documents": []
                    })

                    print(
                        f"[Retrieval Error]: {e}"
                    )
            

            accumulated_documents.extend(
                iteration_documents
            )

            deduplicated_documents = {}

            for doc in accumulated_documents:

                doc_id = (
                    doc.get("id")
                    or doc.get("doc_id")
                )

                if doc_id is not None:

                    deduplicated_documents[
                        str(doc_id)
                    ] = doc

                else:

                    
                    fallback_id = str(
                        hash(
                            doc.get(
                                "content",
                                ""
                            )
                        )
                    )

                    deduplicated_documents[
                        fallback_id
                    ] = doc

            accumulated_documents = list(
                deduplicated_documents.values()
            )    

            

            try:

                evidence_stats = (
                    evidence_tracker.update(
                        iteration_doc_ids
                    )
                )

            except Exception as e:

                print(
                    f"[Evidence Tracker Error]: {e}"
                )

                evidence_stats = {
                    "num_documents":
                        len(iteration_doc_ids),

                    "num_new_documents":
                        None,

                    "document_novelty":
                        None,

                    "semantic_novelty":
                        None,

                    "semantic_median_novelty":
                        None,

                    "high_semantic_novelty_fraction":
                        None,

                    "semantic_redundancy":
                        None,

                    "novelty_values":
                        [],

                    "found_embeddings":
                        False
                }

           

            iteration_latency = (
                time.perf_counter()
                - iteration_start
            )
            
            current_llm_calls = self.llm_call_count
            current_prompt_tokens = self.prompt_tokens
            current_completion_tokens = self.completion_tokens
            current_total_tokens = self.total_tokens

            previous_total_tokens = (
                trajectory[-1]["cost"]["total_tokens_so_far"]
                if trajectory
                else 0
            )

            previous_llm_calls = (
                trajectory[-1]["cost"]["llm_calls_so_far"]
                if trajectory
                else 0
            )

            previous_prompt_tokens = (
                trajectory[-1]["cost"]["prompt_tokens_so_far"]
                if trajectory
                else 0
            )

            previous_completion_tokens = (
                trajectory[-1]["cost"]["completion_tokens_so_far"]
                if trajectory
                else 0
            )

            previous_queries = (
                trajectory[-1]["cost"]["queries_so_far"]
                if trajectory
                else 0
            )

            current_queries = len(action_list)

            queries_so_far = (
                previous_queries + current_queries
            )

            
            state_assessment = check_stopping_gate(
                self,
                question,
                context=context,
                gate_features=[
                    "confidence",
                    "answerable",
                    "discrimination"
                ],
                evidence_stats=evidence_stats,
                options=options,
                use_options_in_gate=True,
                pre_retrieval=False
            )
            state = {
                "state_id":
                    f"{question_id}_S{iteration}",

                "iteration":
                    iteration,

                "question":
                    question,

                "options":
                    options,

                

                "queries": action_list,

                "requested_query_count":
                    n_queries,

                "actual_query_count":
                    current_queries,

                "confidence": state_assessment[0],
                "answerable": state_assessment[1],
                "discrimination": state_assessment[2],
                "assessment_reasoning": state_assessment[3],
                "assessment_type": "evidence_based",

                

                "retrieval_records":
                    retrieval_records,

                "iteration_documents":
                    iteration_documents,

                "iteration_document_ids":
                    iteration_doc_ids,

                "accumulated_document_ids": [
                    str(doc["id"])
                    for doc in accumulated_documents
                    if doc.get("id") is not None
                ],

                

                "context":
                    context,

                

                "evidence_stats":
                    evidence_stats,

                "novelty_available":
                    evidence_stats.get(
                        "semantic_novelty"
                    ) is not None,


                

                "cost": {

                    

                    "queries_so_far":
                        queries_so_far,

                    "llm_calls_so_far":
                        current_llm_calls,

                    "prompt_tokens_so_far":
                        current_prompt_tokens,

                    "completion_tokens_so_far":
                        current_completion_tokens,

                    "total_tokens_so_far":
                        current_total_tokens,

                    "latency_seconds_so_far":
                        time.perf_counter() - start_time,

                    
                    "queries_this_iteration":
                        current_queries,

                    "llm_calls_this_iteration":
                        current_llm_calls
                        - previous_llm_calls,

                    "prompt_tokens_this_iteration":
                        current_prompt_tokens
                        - previous_prompt_tokens,

                    "completion_tokens_this_iteration":
                        current_completion_tokens
                        - previous_completion_tokens,

                    "total_tokens_this_iteration":
                        current_total_tokens
                        - previous_total_tokens,

                    "latency_seconds_this_iteration":
                        iteration_latency,

                    "successful_retrievals_so_far":
                        successful_retrievals,

                    "failed_retrievals_so_far":
                        failed_retrievals,

                    "successful_retrievals_this_iteration":
                        sum(
                            1 for r in retrieval_records
                            if "error" not in r
                        ),

                    "failed_retrievals_this_iteration":
                        sum(
                            1 for r in retrieval_records
                            if "error" in r
                        ),
                }
            }

            trajectory.append(
                state
            )

            

            unique_doc_count = len(set(iteration_doc_ids))

            print(f"  Queries: {actual_generated_queries}")
            print(f"  Retrieved documents (raw): {len(iteration_doc_ids)}")
            print(f"  Unique documents: {unique_doc_count}")
            print(f"  New unique documents: {evidence_stats['num_new_documents']}")
            print(f"  Document novelty: {evidence_stats['document_novelty']}")
            print(f"  Semantic novelty: {evidence_stats['semantic_novelty']}")
            print(f"  Semantic redundancy: {evidence_stats['semantic_redundancy']}")

        

        final_messages = [
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
            final_messages[-1]
        )

        final_response = self.generate(
            final_messages,
            **kwargs
        )

        saved_messages.append({
            "role": "assistant",
            "content": final_response
        })

        total_latency = (
            time.perf_counter()
            - start_time
        )

        

        result = {

            "question":
                question,

            "options":
                options,

            "num_rounds":
                n_rounds,

            "queries_per_round":
                n_queries,

            "trajectory":
                trajectory,

            "final_answer":
                final_response,

            "metrics": {

                "total_queries":
                    total_queries,

                "successful_retrievals":
                    successful_retrievals,

                "failed_retrievals":
                    failed_retrievals,

                "llm_calls":
                    self.llm_call_count,

                "prompt_tokens":
                    self.prompt_tokens,

                "completion_tokens":
                    self.completion_tokens,

                "total_tokens":
                    self.total_tokens,

                "latency_seconds":
                    total_latency
            },

            "history":
                saved_messages
        }

        return result