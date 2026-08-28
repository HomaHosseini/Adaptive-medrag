from src.medrag import MedRAG
import os
from src.adapt_medrag import AdaptiveMedRAG
from datasets import load_dataset

os.environ["ARVAN_API_KEY"] = "d60b8d53-5f6c-5eb6-828a-031d503221d7"
os.environ["ARVAN_ENDPOINT"] = "https://arvancloudai.ir/gateway/models/Llama-3.3-70B-Instruct-R1AI/wXwicG2w6RhESLlIg7KlsetpzCShAX1k0bXHGZLX9nDHLxr4mJwXDMW-2Vunju0LPAXGRBNA8Hg4NhbIgiU-6UwAscgx9QDw1vINQTl1636Buf89X2lEaj3jhmNVJ9Q5qjRgH_G-9CxWta0kJqLV_LtR3XQypA0QnGRFDXtMfKtcE4Q0w5zta-2-Ul-b1rMjyoCaLU2oe4h7NM8-23TJZsUjYp9JrcetdbTBdzh2TDjNK1Iy0I10_c_8RQIN0B9LAEWZihx5ela0-k8AEHGCTwvU5zqzRXjNpwAdEQ/v1"


dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="test"
    )
subset = dataset.select(range(2))

test_questions  =[
        {
            "id": f"MedQA_{i+1}",
            "question": item["question"],
            "options": item["options"],
            "ground_truth": item["answer_idx"]
        }
        for i, item in enumerate(subset)
    ]




MODEL_NAME = "Arvan/Llama-3-3-70B-Instruct-R1AI"

adaptive_medrag = AdaptiveMedRAG(
    llm_name=MODEL_NAME,
    rag=True,
    follow_up=True,
    retriever_name="MedCPT",
    corpus_name="Textbooks",
    corpus_cache=True
)

# test_questions = [
#     {
#         "id": "Q1",
#         "question":"A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given?",
#         "options":{
#         "A":"Gentamicin",
#         "B":"Ciprofloxacin",
#         "C":"Ceftriaxone",
#         "D":"Trimethoprim"
#         }
#     },
#     {
#         "id": "Q2",
#         "question":"A 4670-g (10-lb 5-oz) male newborn is delivered at term to a 26-year-old woman after prolonged labor. Apgar scores are 9 and 9 at 1 and 5 minutes. Examination in the delivery room shows swelling, tenderness, and crepitus over the left clavicle. There is decreased movement of the left upper extremity. Movement of the hands and wrists are normal. A grasping reflex is normal in both hands. An asymmetric Moro reflex is present. The remainder of the examination shows no abnormalities and an anteroposterior x-ray confirms the diagnosis. Which of the following is the most appropriate next step in management?",
#         "options":{
#         "A":"Nerve conduction study",
#         "B":"Surgical fixation",
#         "C":"Physical therapy",
#         "D":"Pin sleeve to the shirt",
#         "E":"Splinting of the arm",
#         "F":"MRI of the clavicle",
#         }
#     }
#]
from src.question_planner import QuestionPlanner
from src.evidence_coverage import EvidenceCoverageAnalyzer

planner = QuestionPlanner(adaptive_medrag)
coverage_analyzer = EvidenceCoverageAnalyzer(adaptive_medrag)

q = test_questions[1]

plan = planner.plan(q["question"])

context = """
PASTE YOUR ITERATION 1 RETRIEVED EVIDENCE HERE
"""

coverage = coverage_analyzer.analyze(
    question=q["question"],
    core_requirements=plan["core_requirements"],
    supporting_constraints=plan["supporting_constraints"],
    context=context,
)

print("\n" + "=" * 70)
print("CORE COVERAGE")
print("=" * 70)

for item in coverage["coverage"]:
    print(
        f"{item['id']} | "
        f"{item['status']} | "
        f"{item['confidence']:.2f}"
    )
    print(
        f"   {item['evidence_summary']}"
    )

print(
    f"\nCore coverage: "
    f"{coverage['core_coverage']:.2f}"
)

print(
    f"Weighted coverage: "
    f"{coverage['weighted_coverage']:.2f}"
)

print("\nCONSTRAINT COVERAGE")

for item in coverage["constraint_coverage"]:
    print(
        f"{item['id']} | "
        f"{item['status']} | "
        f"{item['confidence']:.2f}"
    )
    print(
        f"   {item['evidence_summary']}"
    )

print(
    "\nReasoning:",
    coverage["reasoning"])