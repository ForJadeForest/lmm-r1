import nltk
from nltk.metrics import precision, recall, f_measure
import numpy as np
import jieba
import re
import json
from tqdm import tqdm
from nltk.translate import meteor_score
def get_value_or_zero(value):
    return 0.0 if value is None else value
def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))
def cal_per_metrics(pred, gt):
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics
def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)

    
    task_type_list = ["full-page OCR en", "fine-grained text recognition en","text translation cn"]

    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        
        if data_item["type"] == "fine-grained text recognition en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            elif len(data_item["predict"]) == 0:
                data_item["score"] = 0
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) + 
                    get_value_or_zero(ocr_metric["meteor"]) + 
                    get_value_or_zero(ocr_metric["f_measure"]) + 
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
        elif data_item["type"] == "full-page OCR en":
            if not data_item["predict"]:
                data_item["score"] == 0
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) + 
                    get_value_or_zero(ocr_metric["meteor"]) + 
                    get_value_or_zero(ocr_metric["f_measure"]) + 
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
        elif data_item["type"] == "text translation cn":
            if len(data_item["predict"]) == 0:
                data_item["score"] = 0
            elif len(data_item["answers"][0]) == 0:
                data_item["score"] = 0
                data_item["ignore"] = "True"
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4 

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
process_predictions("/home/lrz/reward/test_bleu.json","/home/lrz/reward/output_bleu.json")