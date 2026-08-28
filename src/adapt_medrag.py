from .medrag import MedRAG
import json
import os
import requests
import re, ast
from .evidence_tracker import EvidenceTracker
from .evidence_coverage import EvidenceCoverageAnalyzer
from .question_planner import QuestionPlanner
from .utils import robust_parse_json
def check_stopping_gate(
    medrag_instance,
    question,
    context,
    threshold=85,
    gate_features=None,
    coverage_stats=None,
    evidence_stats=None,
):
    """
    Configurable stopping gate.

    Available features:
        confidence
        answerable
        coverage
        novelty
        redundancy

    Examples:

        ["confidence", "answerable"]

        ["confidence", "answerable", "coverage"]

        ["confidence", "answerable", "coverage",
         "novelty", "redundancy"]

    The gate NEVER receives answer choices.
    """

    if gate_features is None:
        gate_features = [
            "confidence",
            "answerable",
        ]

    if not context or context.strip() == "":
        return (
            False,
            0.0,
            False,
            "No retrieved evidence available."
        )

    # ==========================================================
    # Build information given to the gate
    # ==========================================================

    signal_text = []

    # ----------------------------------------------------------
    # Coverage
    # ----------------------------------------------------------

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
                    f"- {item['id']}: "
                    f"{item['status']} "
                    f"(confidence={item['confidence']:.2f})"
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

    # ----------------------------------------------------------
    # Novelty
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Redundancy
    # ----------------------------------------------------------

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

    signal_text = "\n".join(signal_text)

    # ==========================================================
    # Feature instructions
    # ==========================================================

    feature_instructions = []

    if "confidence" in gate_features:
        feature_instructions.append(
            """
CONFIDENCE:
Estimate how strongly the retrieved evidence supports
a reliable answer.
"""
        )

    if "answerable" in gate_features:
        feature_instructions.append(
            """
ANSWERABILITY:
Determine whether the current evidence is sufficient
to answer the question reliably without more retrieval.
"""
        )

    if "coverage" in gate_features:
        feature_instructions.append(
            """
REQUIREMENT COVERAGE:
Use the requirement coverage information.

A high confidence score must NOT override an important
core requirement that is still MISSING or PARTIAL.

Pay particular attention to high-importance requirements.
"""
        )

    if "novelty" in gate_features:
        feature_instructions.append(
            """
SEMANTIC NOVELTY:
Determine whether the latest retrieval appears to add
meaningfully different information.

Very low novelty suggests additional retrieval may be
becoming repetitive.
"""
        )

    if "redundancy" in gate_features:
        feature_instructions.append(
            """
SEMANTIC REDUNDANCY:
High redundancy means the newly retrieved evidence is
largely similar to evidence already available and
additional retrieval may have diminishing value.
"""
        )

    feature_instructions = "\n".join(
        feature_instructions
    )

    # ==========================================================
    # Prompt
    # ==========================================================

    gate_prompt = f"""
You are a strict medical QA retrieval stopping controller.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

ACTIVE GATE FEATURES:
{", ".join(gate_features)}

{signal_text}

TASK:

Decide whether another retrieval iteration is necessary.

The retrieval system follows Question-Only Retrieval (QOR).

Therefore:

1. Use only the question and retrieved evidence.
2. Do NOT use answer choices.
3. Do NOT rely on outside medical knowledge.
4. Do NOT produce the final answer.
5. Your job is ONLY to determine whether additional retrieval
   would materially improve the evidence available for solving
   the question.

{feature_instructions}

STOP means:

Additional retrieval is unlikely to materially improve
the reliability of the answer.

CONTINUE means:

Additional retrieval is likely to materially improve
the evidence available for answering the question.

IMPORTANT:

High confidence by itself is NOT sufficient if other enabled
signals indicate important evidence gaps.

Be conservative.

Return ONLY valid JSON:

{{
    "answerable": true,
    "confidence": 90,
    "stop": true,
    "reasoning": "Brief explanation."
}}
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
                "Empty gate response."
            )

        response = str(response).strip()

        # Remove code fences
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

        # Parse JSON
        try:
            result = robust_parse_json(response)
        except Exception as e:
            print(f"Failed to parse gate JSON: {e}")
            raise
        
        # ------------------------------------------------------
        # Answerability
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Confidence
        # ------------------------------------------------------

        confidence = float(
            result.get(
                "confidence",
                0
            )
        )

        confidence = max(
            0.0,
            min(
                100.0,
                confidence
            )
        )

        # ------------------------------------------------------
        # Model stop decision
        # ------------------------------------------------------

        model_stop = result.get(
            "stop",
            False
        )

        if isinstance(
            model_stop,
            str
        ):
            model_stop = (
                model_stop.strip().lower()
                == "true"
            )
        else:
            model_stop = bool(
                model_stop
            )

        # Final decision
        should_stop = (
            model_stop
            and answerable
            and confidence >= threshold
        )

        reasoning = str(
            result.get(
                "reasoning",
                ""
            )
        )

        # ------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------

        print(
            f"   [Gate Features]: "
            f"{gate_features}"
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
    max_iterations=3
):
    """
    Decide how many new follow-up queries to generate
    for the NEXT retrieval iteration.
    """

    # If evidence is already sufficient, no more queries.
    if answerable and confidence >= 85:
        return 0

    # Very weak evidence -> search aggressively.
    if confidence < 60:
        return 4

    # Moderate evidence -> normal search.
    if confidence < 80:
        return 2

    # Strong evidence, but not sufficient to stop.
    return 1


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
    use_coverage=False, use_novelty=False, **kwargs):    
        """
        The spliced iterative loop with the dynamic stopping gate.
        """

        if options is not None:
            options_str = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options_str = ''

        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options_str}"

        # ==========================================================
        # COMPONENT CONFIGURATION
        # ==========================================================

        planner = None
        coverage_analyzer = None
        evidence_tracker = None

        question_plan = None

        core_requirements = []
        supporting_constraints = []

        coverage_history = []
        context = ""
        last_context = None
        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

        stop_flag = False
        confidence = 0
        answerable = False
        gate_reasoning = ""
        stopping_iteration = max_iterations
        coverage_history = []

            # Track what happened at every iteration
        query_budget_history = []
        actual_query_count = 0
        successful_retrieval_count = 0
        failed_retrieval_count = 0

        # ----------------------------------------------------------
        # Which gate information will actually be used?
        # ----------------------------------------------------------

        gate_features = [
            "confidence",
            "answerable"
        ]

        if use_coverage:
            gate_features.append("coverage")

        if use_novelty:
            gate_features.append("novelty")
            gate_features.append("redundancy")


        # ----------------------------------------------------------
        # Question planner
        # ----------------------------------------------------------

        if use_planner:

            planner = QuestionPlanner(self)

            question_plan = planner.plan(question)

            core_requirements = (
                question_plan["core_requirements"]
            )

            supporting_constraints = (
                question_plan["supporting_constraints"]
            )

            print("\n[Question Plan]")

            print(
                f"   Complexity: "
                f"{question_plan['complexity']:.2f}"
            )

            print(
                f"   Planner Initial Budget: "
                f"{question_plan['initial_query_budget']}"
            )

            print("\n   Core Requirements:")

            for req in core_requirements:

                print(
                    f"      {req['id']}: "
                    f"{req['description']} "
                    f"(importance={req['importance']:.2f})"
                )


        # ----------------------------------------------------------
        # Initial query budget
        # ----------------------------------------------------------

        if use_planner and use_planner_budget:

            query_budget = (
                question_plan["initial_query_budget"]
            )

        else:

            query_budget = initial_queries


        print(
            f"\n   [Initial Query Budget]: "
            f"{query_budget}"
        )


        # ----------------------------------------------------------
        # Coverage analyzer
        # ----------------------------------------------------------

        if use_coverage:

            if not use_planner:
                raise ValueError(
                    "Coverage requires the planner because "
                    "coverage needs core requirements."
                )

            coverage_analyzer = (
                EvidenceCoverageAnalyzer(self)
            )


        # ----------------------------------------------------------
        # Evidence tracker
        # ----------------------------------------------------------

        if use_novelty:

            evidence_tracker = EvidenceTracker(
                self.retrieval_system
            )



        # 2. We use a failsafe max_iterations to prevent infinite API billing
        for i in range(max_iterations + 3):

            # 3. THE SPLICE: If we haven't hit the max rounds AND the gate hasn't triggered
            if i < max_iterations and not stop_flag:
                query_budget_history.append(query_budget)

                print(
                    f"   [Query Budget]: {query_budget}"
                )
                if context == "":
                    messages = [
                        {"role": "system", "content": self.templates["i_medrag_system"]},
                        {"role": "user", "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(query_budget)}"}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": self.templates["i_medrag_system"]},
                        {"role": "user", "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(query_budget)}"}
                    ]

            # 4. Trigger the final answer generation if the gate said STOP or we ran out of rounds
            elif context != last_context:
                messages = [
                    {"role": "system", "content": self.templates["i_medrag_system"]},
                    {"role": "user", "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}"}
                ]
            elif len(messages) == 1:
                messages = [
                    {"role": "system", "content": self.templates["i_medrag_system"]},
                    {"role": "user", "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}"}
                ]

            saved_messages.append(messages[-1])
            last_context = context

            # Generate LLM response (Queries or Answer)
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)

            # --- EXTRACT FINAL ANSWER ---
            if (i >= max_iterations or stop_flag) and ("## Answer" in last_content or "answer is" in last_content.lower()):
                messages.append(response_message)
                messages.append({
                    "role": "user",
                    "content": (
                    "Return ONLY valid JSON. "
                    'Return exactly this format: {"answer":""} '
                    "where the answer must be one of A, B, C, D, E, or F. "
                    "Do not use single quotes. "
                    "Do not include explanation."
                ) if options else "Output the answer in JSON: {'answer': your_answer}"
                })
                
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                messages.append({"role": "assistant", "content": answer_content})
                saved_messages.append(messages[-1])

                # We return exactly how many iterations it took to prove early stopping!
                return (
                    messages[-1]["content"],
                    saved_messages,
                    stopping_iteration,
                    confidence,
                    query_budget_history,
                    actual_query_count,
                    successful_retrieval_count,
                    failed_retrieval_count,
                    evidence_tracker.history if evidence_tracker else [],
                    coverage_history,
                    {
                        "use_planner": use_planner,
                        "use_planner_budget": use_planner_budget,
                        "use_coverage": use_coverage,
                        "use_novelty": use_novelty,
                        "question_plan": question_plan
                    }
                )
            # --- EXTRACT QUERIES & RETRIEVE ---
            elif "## Queries" in last_content:
                messages = messages[:-1]
                try:
                    action_str = self.generate([{
                        "role": "user",
                        "content": f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}"
                    }], **kwargs)
                    action_str = re.search(r"output\": (\[.*\])", action_str, re.DOTALL).group(1)
                    action_list = [re.sub(r'^\d+\.\s*', '', s.strip()) for s in eval(action_str)]
                    actual_query_count += len(action_list)

                    print(
                        f"   [Actual Queries Generated]: {len(action_list)}"
                    )
                except Exception as E:
                    print(f"Error parsing queries: {E}")
                    continue

                # Retrieve documents for each query
                iteration_doc_ids = []
                for q in action_list:

                    if q.strip() == "":
                        continue

                    try:
                        rag_result, retrieved_snippets, scores = self.medrag_answer(
                            q,
                            k=k,
                            rrf_k=rrf_k,
                            **kwargs
                        )

                        successful_retrieval_count += 1

                        for doc in retrieved_snippets:
                            iteration_doc_ids.append(doc["id"])

                        context += f"\n\nQuery: {q}\nAnswer: {rag_result}"
                        context = context.strip()

                    except Exception as E:
                        failed_retrieval_count += 1
                        print(f"Retrieval Error: {E}")
                evidence_stats = {
                    "document_novelty": None,
                    "semantic_novelty": None,
                    "semantic_redundancy": None
                }

                if use_novelty:

                    evidence_stats = evidence_tracker.update(
                        iteration_doc_ids
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
                coverage_stats = {
                    "coverage": [],
                    "constraint_coverage": [],
                    "core_coverage": None,
                    "weighted_coverage": None,
                    "reasoning": ""
                }

                if use_coverage:

                    coverage_stats = coverage_analyzer.analyze(
                        question=question,
                        core_requirements=core_requirements,
                        supporting_constraints=supporting_constraints,
                        context=context
                    )

                    coverage_history.append(
                        coverage_stats
                    )

                    print(
                        f"   [Core Coverage]: "
                        f"{coverage_stats['core_coverage']:.2f}"
                    )

                    print(
                        f"   [Weighted Coverage]: "
                        f"{coverage_stats['weighted_coverage']:.2f}"
                    )

                    print(
                        "   [Requirement Coverage]:"
                    )

                    for item in coverage_stats["coverage"]:

                        print(
                            f"      {item['id']} | "
                            f"{item['status']} | "
                            f"{item['confidence']:.2f}"
                        )

                

                # 5. THE GATE: Check if the newly retrieved context is sufficient
                print(f"\n--- Checking Gate after Iteration {i+1} ---")

                
                stop_flag, confidence, answerable, gate_reasoning = (
                    check_stopping_gate(
                        self,
                        question,
                        context,
                        threshold=confidence_threshold,
                        gate_features=gate_features,
                        coverage_stats=coverage_stats,
                        evidence_stats=evidence_stats
                    )
                )

                if stop_flag:

                    stopping_iteration = i + 1

                    query_budget = 0

                else:


                    query_budget = choose_query_budget(
                        confidence=confidence,
                        answerable=answerable,
                        iteration=i + 1,
                        max_iterations=max_iterations
                    )

                    print(
                        f"   [Next Query Budget]: {query_budget}"
                    )

            else:
                messages.append(response_message)
                continue

        return (
            messages[-1]["content"],
            saved_messages,
            max_iterations,
            confidence,
            query_budget_history,
            actual_query_count,
            successful_retrieval_count,
            failed_retrieval_count,
            evidence_tracker.history if evidence_tracker else [],
            coverage_history,
            {
                "use_planner": use_planner,
                "use_planner_budget": use_planner_budget,
                "use_coverage": use_coverage,
                "use_novelty": use_novelty,
                "question_plan": question_plan
            }
        )
