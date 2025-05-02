import json
import json
import os
import re
import ast
from argparse import ArgumentParser
from flask import Flask, jsonify, request
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from vqa_metric import (vqa_evaluation, cn_vqa_evaluation, math_expression_evaluation,
                        vqa_evaluation_case_sensitive, counting_evaluation, 
                        cn_math_expression_evaluation)
from IoUscore_metric import (vqa_with_position_evaluation, calculate_iou,
                            extract_coordinates)
from TEDS_metric import (TEDS, convert_markdown_table_to_html, convert_str_to_dict,
                        convert_str_to_multi_dict, generate_combinations, dict_to_html,
                        compute_f1_score, doc_parsing_evaluation, wrap_html_table)
from page_ocr_metric import cal_per_metrics
from spotting_metric import extract_bounding_boxes_robust, spotting_evaluation

app = Flask(__name__)
data_index = {}
executor = None
teds_evaluator = TEDS(n_jobs=os.cpu_count())

def extract_response(query: str, prompt: str) -> str:
    tag_patterns = [
        r"<answer>(.*?)</answer>",
        r"\|Answer\|>(.*?)$",
        r"Assistant:\s*(.*?)(<\|im_end\|>|$)"
    ]
    for pattern in tag_patterns:
        match = re.search(pattern, query, re.DOTALL|re.IGNORECASE)
        if match:
            return match.group(1).strip()

    if query.startswith(prompt):
        return query[len(prompt):].strip()
    
    segments = [s.strip() for s in query.split("\n") if s.strip()]
    return segments[-1] if segments else ""

def handle_generic_vqa(response, answer, item):
    if item.get("eval") == "multiple choice":
        pred_char = ''.join(filter(str.isalpha, response)).upper()
        if isinstance(answer, list):
            return 1.0 if any(pred_char == a.upper() for a in answer) else 0.0
        return 1.0 if pred_char == str(answer).upper() else 0.0

    if isinstance(answer, list):
        return max(1.0 if ans.lower() in response.lower() else 0.0 for ans in answer)
    return 1.0 if response.strip().lower() == str(answer).lower() else 0.0

def handle_math(response, answer, item):
    try:
        return float(math_expression_evaluation(response, [answer]))
    except Exception as e:
        print(f"Math evaluation error: {e}")
        return 0.0

def handle_table_parsing(response, answer, item):
    try:
        if "markdown" in item.get("question", "").lower():
            pred_html = convert_markdown_table_to_html(response)
        else:
            pred_dict = convert_str_to_multi_dict(response) or {}
            pred_html = dict_to_html(pred_dict)

        if isinstance(answer, list):
            answer = answer[0]  
        gt_html = wrap_html_table(convert_markdown_table_to_html(answer))
        
        return teds_evaluator.evaluate(wrap_html_table(pred_html), gt_html)
    except Exception as e:
        print(f"Table parsing error: {e}")
        return 0.0

def handle_text_spotting(response, answer, item):
    try:
        pred_boxes = extract_bounding_boxes_robust(response)
        if not pred_boxes:
            return 0.0

        gt_boxes = [{
            "text": box.get("text", ""),
            "bbox": box["bbox"]
        } for box in answer]
        
        return spotting_evaluation(pred_boxes, gt_boxes)
    except Exception as e:
        print(f"Text spotting error: {e}")
        return 0.0

def handle_kie(response, answer, item, lang="en"):
    try:
        if isinstance(answer, str):
            answer = ast.literal_eval(answer) if lang == "cn" else json.loads(answer)
        answer = generate_combinations(answer)
 
        pred_dict = convert_str_to_dict(response) or {}
        return max(compute_f1_score(pred_dict, gt) for gt in answer)
    except Exception as e:
        print(f"KIE error: {e}")
        return 0.0

HANDLERS = {
    # en
    "APP agent en": handle_generic_vqa,
    "ASCII art classification en": handle_generic_vqa,
    "key information extraction en": partial(handle_kie, lang="en"),
    "key information mapping en": partial(handle_kie, lang="en"),
    "math QA en": handle_math,
    "full-page OCR en": lambda r,a,i: cal_per_metrics(r, a[0])["f_measure"],
    "reasoning VQA en": handle_generic_vqa,
    "fine-grained text recognition en": lambda r,a,i: cal_per_metrics(r, a[0])["f_measure"],
    "science QA en": handle_generic_vqa,
    "table parsing en": handle_table_parsing,
    "text counting en": lambda r,a,i: float(counting_evaluation(r, a, i.get("eval"))),
    "text grounding en": lambda r,a,i: calculate_iou(extract_coordinates(r), a),
    "text recognition en": handle_generic_vqa,
    "text spotting en": handle_text_spotting,
    "document classification en": handle_generic_vqa,
    "cognition VQA en": handle_generic_vqa,
    "VQA with position en": lambda r,a,i: vqa_with_position_evaluation(convert_str_to_dict(r), i),
    "chart parsing en": handle_table_parsing,
    "document parsing en": lambda r,a,i: doc_parsing_evaluation(r, a[0]),
    "formula recognition en": handle_math,
    "diagram QA en": handle_generic_vqa,
    
    # cn
    "cognition VQA cn": lambda r,a,i: cn_vqa_evaluation(r, a),
    "key information extraction cn": partial(handle_kie, lang="cn"),
    "formula recognition cn": lambda r,a,i: float(cn_math_expression_evaluation(r, a)),
    "full-page OCR cn": lambda r,a,i: cal_per_metrics(r, a[0])["f_measure"],
    "reasoning VQA cn": lambda r,a,i: cn_vqa_evaluation(r, a),
    "text translation cn": lambda r,a,i: cal_per_metrics(r, a[0])["bleu"],
    "table parsing cn": handle_table_parsing,
    "handwritten answer extraction cn": lambda r,a,i: 1.0 if any(ans in r for ans in a) else 0.0,
    "document parsing cn": lambda r,a,i: doc_parsing_evaluation(r, a[0]),
    
    # defult
    "__default__": lambda r,a,i: (
        print(f"Unknown task type: question_type={i.get('question_type', 'unknown')}"),
        0.0
    )[1]
}

@app.route("/get_reward", methods=["POST"])
def get_reward():
    data = request.get_json()
    if not data or "query" not in data or "prompts" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    rewards = []
    for q, prompt_json in zip(data["query"], data["prompts"]):
        print(f"\n[DEBUG] raw prompt JSON: {prompt_json[:100]}...")
        try:
            p = json.loads(prompt_json)
        except json.JSONDecodeError:
            p = None

        if isinstance(p, dict) and "message" in p and "image_path" in p:
            key_dict = {"question": p["message"], "image_path": p["image_path"]}
            key = json.dumps(key_dict, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(p, dict) and "message" in p:
            key = json.dumps(p["message"], ensure_ascii=False, separators=(",", ":"))
        else:
            key = prompt_json.strip()

        print(f"[DEBUG] The key used for looking up the index: {key[:50]}...")
        item = data_index.get(key)
        if not item and isinstance(p, dict) and "message" in p:
            # fallback to question-only key
            fallback_key = json.dumps(p["message"], ensure_ascii=False, separators=(",", ":"))
            item = data_index.get(fallback_key)

        if not item:
            print(f"[WARN] No match found for key: {key[:50]}...")
            rewards.append(0.0)
            continue

        try:
            response = extract_response(q, item["question"])
            qtype = item.get("question_type", item.get("type"))
            answer = item.get("answers", item.get("answer", []))

            print(f"[DEBUG] type: {qtype} | answer: {answer} | predict: {response}")

            handler = HANDLERS.get(qtype, HANDLERS["__default__"])
            reward = handler(response, answer, item)
            rewards.append(float(reward))
        except Exception as e:
            print(f"[ERROR] Handle exceptions: {str(e)}")
            rewards.append(0.0)

    return jsonify({"rewards": rewards})

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-file", type=str, default="/home/wmk/code/OCRBench_v2/pred_folder/internvl2_5_26b.json", help="data_path")
    parser.add_argument("--port", type=int, default=5000, help="port")
    args = parser.parse_args()

    with open(args.data_file) as f:
        dataset = json.load(f)

    print("\n=== Loading Data_idx ===")
    valid_count = 0
    for item in dataset:
        if "question" not in item or "image_path" not in item:
            continue
        key = json.dumps({"question": item["question"], "image_path": item["image_path"]}, ensure_ascii=False, separators=(",", ":"))
        data_index[key] = item
        valid_count += 1

    print(f"total num of data: {len(dataset)}")
    print(f"valid num of data: {valid_count}")

    executor = ProcessPoolExecutor(max_workers=8)
    print("\n=== Seevice Lauching ===")
    app.run(host='0.0.0.0', port=args.port, threaded=False)
