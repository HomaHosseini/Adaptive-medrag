from .medrag import MedRAG
import json
import os
import requests
import re, ast
from .evidence_tracker import EvidenceTracker

def check_stopping_gate(medrag_instance,question,context,threshold=85):
    """
    Question-only stopping gate.

    The gate receives:
        - the original question
        - accumulated retrieved evidence

    It does NOT receive the answer choices.

    Returns:
        should_stop
        confidence
        answerable
        reasoning
    """

    if not context or context.strip() == "":
        return False, 0, False, "No retrieved evidence available."

    gate_prompt = f"""
You are a strict medical QA stopping controller.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

TASK:
Determine whether the retrieved evidence is sufficient to answer
the medical question reliably.

The retrieval process follows Question-Only Retrieval (QOR).
Therefore, you must judge sufficiency using ONLY:

1. The question
2. The retrieved evidence

Do NOT use the answer choices.
Do NOT rely on outside medical knowledge to fill missing information.

The purpose of this controller is NOT to produce the final answer.
Its only purpose is to decide whether another retrieval iteration
is necessary.

Definitions:

answerable=true:
The retrieved evidence contains enough relevant and specific
information that a medical QA system could reliably answer the
question without needing additional retrieval.

answerable=false:
Important information is missing, evidence is too general,
evidence is contradictory, or additional retrieval would
reasonably improve the answer.

Confidence:
0-20   = evidence is essentially insufficient
21-40  = weak evidence
41-60  = partial evidence
61-80  = useful evidence but important uncertainty remains
81-90  = strong evidence
91-100 = very strong/direct evidence

IMPORTANT:
- High confidence alone does NOT mean STOP.
- Only answerable=true AND confidence >= the threshold should STOP.
- If the evidence does not directly address an important part
  of the question, prefer CONTINUE.
- Be conservative.

Return ONLY valid JSON:

{{
    "answerable": true,
    "confidence": 90,
    "reasoning": "Brief explanation of why the retrieved evidence is or is not sufficient."
}}
"""

    messages = [
        {
            "role": "user",
            "content": gate_prompt
        }
    ]

    try:
        response = medrag_instance.generate(messages).strip()

        # Remove markdown code fences
        response = re.sub(r"^```json\s*", "", response)
        response = re.sub(r"^```\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        response = response.strip()

        result = json.loads(response)

        answerable_value = result.get("answerable", False)

        # Handle true/false returned as string
        if isinstance(answerable_value, str):
            answerable = answerable_value.strip().lower() == "true"
        else:
            answerable = bool(answerable_value)

        confidence = float(
            result.get("confidence", 0)
        )

        confidence = max(0, min(100, confidence))

        reasoning = str(
            result.get("reasoning", "")
        )

        should_stop = (
            answerable
            and confidence >= threshold
        )

        print(
            f"   [Gate Answerable]: {answerable}"
        )

        print(
            f"   [Gate Confidence]: {confidence:.1f}%"
        )

        print(
            f"   [Gate Threshold]: {threshold}%"
        )

        print(
            f"   [Gate Decision]: "
            f"{'STOP' if should_stop else 'CONTINUE'}"
        )

        print(
            f"   [Gate Reasoning]: {reasoning}"
        )

        return (
            should_stop,
            confidence,
            answerable,
            reasoning
        )

    except Exception as e:

        print(f"   [Gate Error]: {e}")
        print("   Defaulting to CONTINUE.")

        return (
            False,
            0,
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

    def adaptive_answer(self, question, options=None, k=32, rrf_k=100, save_path=None, max_iterations=4,  initial_queries=2,confidence_threshold=85, **kwargs):
        """
        The spliced iterative loop with the dynamic stopping gate.
        """
        if options is not None:
            options_str = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options_str = ''

        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options_str}"

        evidence_tracker = EvidenceTracker(
            self.retrieval_system
        )
        context = ""
        last_context = None
        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

        stop_flag = False
        confidence = 0
        answerable = False
        gate_reasoning = ""
        stopping_iteration = max_iterations
        query_budget = initial_queries

        # Track what happened at every iteration
        query_budget_history = []
        actual_query_count = 0

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
                return messages[-1]["content"], saved_messages, stopping_iteration ,confidence,query_budget_history, actual_query_count, evidence_tracker.history

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
                    if q.strip() == "": continue
                    try:
                        rag_result, retrieved_snippets, scores = self.medrag_answer(q, k=k,rrf_k=rrf_k, **kwargs)
                        for doc in retrieved_snippets:
                            iteration_doc_ids.append(doc["id"])
                        context += f"\n\nQuery: {q}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        print(f"Retrieval Error: {E}")
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

                # 5. THE GATE: Check if the newly retrieved context is sufficient
                print(f"\n--- Checking Gate after Iteration {i+1} ---")

                
                stop_flag, confidence, answerable, gate_reasoning = (
                    check_stopping_gate(
                        self,
                        question,
                        context,
                        threshold=confidence_threshold
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

        return messages[-1]["content"], saved_messages, max_iterations, confidence,query_budget_history, actual_query_count, evidence_tracker.history

