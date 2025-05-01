import json
import ast
from itertools import product
import numpy as np
import argparse
from tqdm import tqdm
from vqa_metric import counting_evaluation
from IOU import convert_str_to_dict

def generate_combinations(input_dict):
    """
    Function to generate all possible combinations of values from a dictionary.
    """
    kie_answer = input_dict
    if not isinstance(kie_answer, dict):
        kie_answer = kie_answer.strip('"')
        try:
            kie_answer = json.loads(kie_answer)
        except json.JSONDecodeError:
            try:
                kie_answer = ast.literal_eval(kie_answer)
                if not isinstance(kie_answer, dict):
                    kie_answer = ast.literal_eval(kie_answer)
            except (ValueError, SyntaxError):
                print(f"Unable to parse 'answers' field: {kie_answer}")
                return {}
        
        # Ensure the parsed result is a dictionary.
        if not isinstance(kie_answer, dict):
            print("Parsed 'answers' is still not a dictionary.")
            raise ValueError("Input could not be parsed into a dictionary.")
    
        keys = list(kie_answer.keys())
        
        value_lists = []
        for single_key in keys:
            sinlge_value = kie_answer[single_key]
            if not isinstance(sinlge_value, list):
                sinlge_value = [sinlge_value]
            value_lists.append(sinlge_value)
    
        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))
    
        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result
    
    else:
        keys = list(input_dict.keys())
        value_lists = [input_dict[key] for key in keys]

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result

def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score for KIE task between predicted and ground truth dictionaries.

    Args:
        preds (dict): The predicted key-value pairs.
        gts (dict): The ground truth key-value pairs.
        ignores (list): The list of keys to ignore during evaluation.

    Returns:
        dict: A dictionary where keys are field names and values are their corresponding F1-scores.
    """
    # Optionally remove ignored keys from predictions and ground truths
    keys = set(preds.keys()).union(set(gts.keys())) - set(ignores)
    f1_scores = {}

    for key in keys:
        pred_value = preds.get(key, None)
        gt_value = gts.get(key, None)

        if pred_value:
            pred_value = pred_value.lower().strip().replace("\n"," ").replace(" ", "")
        if gt_value:
            gt_value = gt_value.lower().strip().replace("\n"," ").replace(" ", "")

        if pred_value is None and gt_value is None:
            continue
        elif pred_value is None:
            precision = 0.0
            recall = 0.0
        elif gt_value is None:
            # false positive
            precision = 0.0
            recall = 0.0
        else:
            if pred_value == gt_value:
                # True positive
                precision = 1.0
                recall = 1.0
            else:
                precision = 0.0
                recall = 0.0

        # Compute F1-score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[key] = f1_score

    if len(f1_scores) == 0:
        return 0
    average_f1 = sum(f1_scores.values()) / len(f1_scores)

    return average_f1

def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)
    task_type_list = [ "text counting en","key information extraction en","key information mapping en"]
    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        if data_item["type"] == "text counting en":
            data_item["score"] = counting_evaluation(data_item["predict"], data_item["answers"], data_item["eval"])

        elif data_item["type"] == "key information extraction en" or data_item["type"] == "key information mapping en":
            assert len(data_item["answers"]) == 1
            answers = generate_combinations(data_item["answers"][0])
            
            if type(answers)==list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)
                    
                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

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
        "--input_path", type=str, default="/home/wmk/code/OCR-R1/reward/F1.json", help="Path to the input prediction JSON file."
    )
    parser.add_argument(
        "--output_path", type=str, default="/home/wmk/code/OCR-R1/reward/F1_result.json", help="Path to save the results JSON file."
    )

    args = parser.parse_args()

    process_predictions(args.input_path, args.output_path)

print("End of Code!")