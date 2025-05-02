import json
import math
import re
from tqdm import tqdm
def extract_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    return None
def counting_evaluation(predict, answers, eval_method):
    score = 0
    
    if isinstance(predict, str):
        predict_processed = predict.lower().strip().replace("\n", " ")
    elif math.isnan(predict):
        return 0
    else:
        predict_processed = int(predict)
    if type(answers)==list:
        temp_score = 0
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n"," ")
            if eval_method == "exact match":
                if answer in predict:
                    score = 1
                else:
                    score = 0
            elif eval_method == "regression":
                predict_number = extract_first_number(predict_processed)
                if predict_number:
                    
                    answer = int(answer)
                    
                    if predict_number <= 0 or predict_number >= 2 * answer:
                        score = 0
                    else:
                        iou = 1 - abs(predict_number - answer) / answer
                        if iou > 0.5:
                            score = iou
                        else:
                            score = 0
                else:
                    score = 0
            if score > temp_score:
                temp_score = score
        score = temp_score

    else:
        answers = answers.lower().strip().replace("\n"," ")
        predict = predict.lower().strip().replace("\n"," ")
        if eval_method == "exact match":
            if answer in predict:
                score = 1
            else:
                score = 0
        elif eval_method == "regression":
            predict = extract_first_number(predict)
            if predict:
                answer = int(answer)
                if predict <= 0 or predict >= 2 * answer:
                    score = 0
                else:
                    iou = 1 - abs(predict - answer) / answer
                    
                    if iou > 0.5:
                        score = iou
                    else:
                        score = 0
            else:
                score = 0
    return score

def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)

    

    task_type_list = [ "text counting en"]

    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        
        if data_item["type"] == "text counting en":
            data_item["score"] = counting_evaluation(data_item["predict"], data_item["answers"], data_item["eval"])
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
process_predictions("/home/lrz/reward/test_L1.json","/home/lrz/reward/output_L1.json")