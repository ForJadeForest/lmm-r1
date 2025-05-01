import json
import numpy as np
import argparse
from tqdm import tqdm
from vqa_metric import vqa_evaluation, vqa_evaluation_case_sensitive, math_expression_evaluation

def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)
    task_type_list = [ "APP agent en", "ASCII art classification en", "math QA en", "reasoning VQA en", "science QA en", "text recognition en", 
                      "document classification en", "cognition VQA en", "diagram QA en", "formula recognition en"]
    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        if data_item["type"] == "APP agent en" or data_item["type"] == "ASCII art classification en" or data_item["type"] == "math QA en" \
            or data_item["type"] == "reasoning VQA en" or data_item["type"] == "science QA en" \
            or data_item["type"] == "text recognition en" or data_item["type"] == "document claassification en" \
            or data_item["type"] == "cognition VQA en" or data_item["type"] == "diagram QA en":
            if "eval" in data_item.keys():
                if data_item["eval"] == "multiple choice":
                    if not isinstance(data_item["answers"], list):
                        data_item["answers"] = [data_item["answers"]]
                    assert len(data_item["answers"]) == 1

                    if not isinstance(data_item["predict"], str):
                        data_item["score"] = 0
                    else:
                        predict = ''.join(c for c in data_item["predict"] if c.isalpha())

                        if predict == data_item["answers"][0]:
                            data_item["score"] = 1
                        else:
                            data_item["score"] = 0
                elif data_item["eval"] == "case sensitive":
                    data_item["score"] = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
                else:
                    raise ValueError("No such evaluation method") 
            else:
                data_item["score"] = vqa_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "formula recognition en":
            data_item["score"] = math_expression_evaluation(data_item["predict"], data_item["answers"])

        else:
            raise ValueError("Unknown task type!")
        
        res_data_list.append(data_item)
    
    for task_name in task_type_list:
        print("\n" + task_name)
        mean_score, total_len = 0, .0
        for item in res_data_list:
            if item["type"] == task_name:
                total_len += 1
                mean_score += item["score"]
        
        mean_score = mean_score / total_len if total_len > 0 else 0
        print(f"Task {task_name}, total instructions: {total_len}, average score: {mean_score:.3f}\n")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(predict_file, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prediction JSON files and evaluate results.")
    parser.add_argument(
        "--input_path", type=str, default="/home/wmk/code/OCR-R1/reward/Exact Match.json", help="Path to the input prediction JSON file."
    )
    parser.add_argument(
        "--output_path", type=str, default="/home/wmk/code/OCR-R1/reward/Exact Match_result.json", help="Path to save the results JSON file."
    )

    args = parser.parse_args()

    process_predictions(args.input_path, args.output_path)

print("End of Code!")