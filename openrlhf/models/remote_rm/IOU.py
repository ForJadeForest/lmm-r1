import os
import re
import ast
import ipdb
import json
import numpy as np
import shutil
import zipfile
import argparse
import editdistance
from vqa_metric import vqa_evaluation
from spotting_eval.script import default_evaluation_params,validate_data,evaluate_method
import spotting_eval.rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from tqdm import tqdm

def get_anls(s1, s2):
    try:
        s1 = s1.lower()
        s2 = s2.lower()
    except:
        pass
    if s1 == s2:
        return 1.0
    iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
    anls = iou
    return anls

def calculate_iou(box1, box2):

    try:
        box1 = [int(coordinate) for coordinate in box1]
        box2 = [int(coordinate) for coordinate in box2]
    except:
        return 0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
  
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou

def vqa_position_reward(pred_box, gt_box, answer_sim, temperature=0.7):
    """
    VQA带位置奖励函数（强化学习优化版）
    :param pred_box: 预测框坐标[x1,y1,x2,y2]
    :param gt_box: 真实框坐标
    :param answer_sim: 答案相似度（0-1）
    :param temperature: 奖励曲线调整参数（0.5-1.0）
    """
    iou = calculate_iou(pred_box, gt_box)
    
    # 动态温度调整：IOU越高，温度系数越小，奖励曲线越陡峭
    adj_temp = temperature * (1 - 0.2 * iou)  
    
    # 组合奖励公式
    reward = answer_sim * (iou ** (1/adj_temp)) 
    return min(reward, 1.0)

def text_grounding_reward(pred_polygons, gt_polygons, iou_thresh=0.5):
    """
    文本定位奖励函数（多边形IOU版本）
    :param pred_polygons: 预测多边形列表[[x1,y1,...],...]
    :param gt_polygons: 真实多边形列表
    :param iou_thresh: 有效匹配阈值
    """
    matched_gt = set()
    total_reward = 0.0
    coverage_bonus = 0.0
    
    for pred in pred_polygons:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_polygons):
            if idx in matched_gt:
                continue
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        if best_iou >= iou_thresh:
            total_reward += best_iou
            matched_gt.add(best_idx)
            # 方向一致性奖励（假设多边形为四边形）
            pred_dir = np.arctan2(pred[3]-pred[1], pred[2]-pred[0])
            gt_dir = np.arctan2(gt[3]-gt[1], gt[2]-gt[0])
            angle_diff = abs(pred_dir - gt_dir) % (2*np.pi)
            total_reward += 0.2 * (1 - angle_diff/np.pi)
    
    # 覆盖率奖励
    coverage = len(matched_gt) / len(gt_polygons)
    coverage_bonus = 0.3 * coverage ** 2
    
    return min(total_reward / len(pred_polygons) + coverage_bonus, 1.0)

def text_spotting_reward(detection_iou, recognition_acc, continuity=0.9):
    """
    文本识别综合奖励函数
    :param detection_iou: 检测框IOU（0-1）
    :param recognition_acc: 识别准确率（0-1）
    :param continuity: 连续性因子（0.9-1.0）
    """
    # 动态权重：当检测IOU低时更关注检测质量
    detection_weight = 0.6 + 0.2 * (1 - detection_iou)
    recognition_weight = 1 - detection_weight
    
    # 基础奖励
    base_reward = (detection_weight * detection_iou + 
                  recognition_weight * recognition_acc)
    
    # 连续性奖励：鼓励连续正确预测
    continuity_bonus = 0.1 * (continuity ** 3)
    
    # 字符级精度奖励
    char_level_bonus = 0.15 * (recognition_acc ** 2)
    
    return min(base_reward + continuity_bonus + char_level_bonus, 1.0)

def vqa_with_position_evaluation(predict, img_metas):

    score_content, score_bbox = .0, .0
    if "answer" in predict.keys():
        score_content = vqa_evaluation(predict["answer"], img_metas["answers"])
    if "bbox" in predict.keys():
        gt_bbox = img_metas["bbox"]
        try:
            predict_bbox_list = ast.literal_eval(predict["bbox"])
            score_bbox = calculate_iou(predict_bbox_list, gt_bbox)
        except:
            score_bbox = 0
    return 0.5 * score_content + 0.5 * score_bbox


def extract_coordinates(text):
    # Regex pattern to match coordinates in either (x1, y1, x2, y2) or [x1, y1, x2, y2] format

    pattern = r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]'

    matches = list(re.finditer(pattern, text))
    coords_list = []
    coords_set = set()
    for match in matches:

        x1, y1, x2, y2 = map(int, match.groups())

        if all(0 <= n <= 1000 for n in [x1, y1, x2, y2]):
            coords = (x1, y1, x2, y2)

            if coords in coords_set:
                coords_list = [c for c in coords_list if c != coords]

            coords_list.append(coords)
            coords_set.add(coords)
    if coords_list:
        last_coords = coords_list[-1]
        return list(last_coords)
    else:
        return None
#-----------Text Spotting------------#
def zip_folder(source_folder, destination_zip):
    abs_source = os.path.abspath(source_folder)
    abs_destination = os.path.abspath(destination_zip)

    with zipfile.ZipFile(abs_destination, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(abs_source):
            for file in files:
                abs_file_path = os.path.join(root, file)

                relative_path = os.path.relpath(abs_file_path, abs_source)
                zf.write(abs_file_path, relative_path)

def convert_str_to_dict(predict_str: str):
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    match = re.search(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        content = predict_str.strip()

    data = {}
    success = False

    # try parsing with JSON
    try:
        data = json.loads(content)
        success = True
    except json.JSONDecodeError:
        pass

    # try parsing with ast.literal_eval
    if not success:
        try:
            data = ast.literal_eval(content)
            if isinstance(data, dict):
                success = True
        except (ValueError, SyntaxError):
            pass

    # try parsing with regex
    if not success:
        key_value_pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
        matches = re.findall(key_value_pattern, content)
        try:
            for key, value in matches:
                data[key.strip()] = value.strip()
        except:
            return {}

    if not data:
        return {}

    try:
        result = {k.strip(): str(v).strip() for k, v in data.items()}
    except:
        return {}
    return result

def spotting_evaluation(prediction_list, img_metas):
    score = 0

    submit_path = "/home/wmk/code/OCR-R1/reward/spotting_eval/submit"
    gt_path = "/home/wmk/code/OCR-R1/reward/spotting_eval/gt"
    submit_zip_path = "/home/wmk/code/OCR-R1/reward/spotting_eval/submit.zip"
    gt_zip_path = "/home/wmk/code/OCR-R1/reward/spotting_eval/gt.zip"
    for file_path in [submit_path, gt_path, submit_zip_path, gt_zip_path]:
        if "zip" in file_path:
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            if os.path.exists(file_path):
                shutil.rmtree(file_path)
            os.makedirs(file_path)

    res_submit_list = []
    for item in prediction_list:
        if len(item) != 5:
            ipdb.set_trace()
        x1, y1, x2, y2, rec = item
        if x1 >= x2 or y1 >= y2:
            continue
        
        res_submit_list.append(",".join([str(x1),str(y1),str(x2),str(y1),str(x2),str(y2),str(x1),str(y2),rec]))

    res_gt_list = []
    for bbox, rec in zip(img_metas["bbox"], img_metas["content"]):
        x_coords = bbox[0::2]
        y_coords = bbox[1::2]

        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        res_gt_list.append(",".join([str(x1),str(y1),str(x2),str(y1),str(x2),str(y2),str(x1),str(y2),rec]))

    if len(res_submit_list) == 0 or len(res_gt_list) == 0:
        return 0

    with open(os.path.join(submit_path,"res_img_0.txt"), "w") as f:
        for item in res_submit_list[:-1]:
            f.write(item + "\n")
        f.write(res_submit_list[-1])
    
    with open(os.path.join(gt_path,"gt_img_0.txt"), "w") as f:
        for item in res_gt_list[:-1]:
            f.write(item + "\n")
        f.write(res_gt_list[-1])

    zip_folder(submit_path, submit_zip_path)
    zip_folder(gt_path, gt_zip_path)

    command = {
        'g': gt_zip_path,        
        's': submit_zip_path,    
        'o': './',               
        'p': '{"IOU_CONSTRAINT":0.5}'  
    }

    # run rrc_evaluation_funcs
    result = rrc_evaluation_funcs.main_evaluation(command,default_evaluation_params,validate_data,evaluate_method)
    score = result["method"]["hmean"]
    return score

def extract_bounding_boxes_robust(predict_str):
    """
    Extract coordinates and text content from the given prediction string, 
    handling potential format issues.

    Args:
    predict_str (str): Model prediction output as a string.

    Returns:
    list: Extracted data in the format [[x1, y1, x2, y2, text_content], ...].
          Returns None if no valid data is extracted.
    """
    results = []
    seen = set()

    # try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(predict_str)
    except Exception:
        data = None

    if data is not None:
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 5:
                    x1_str, y1_str, x2_str, y2_str = item[:4]
                    text_content = item[4]

                    x1_str = str(x1_str).strip()
                    y1_str = str(y1_str).strip()
                    x2_str = str(x2_str).strip()
                    y2_str = str(y2_str).strip()
                    text_content = str(text_content).replace("\n", "").strip().strip('"').strip("'")

                    try:
                        x1 = int(x1_str)
                        y1 = int(y1_str)
                        x2 = int(x2_str)
                        y2 = int(y2_str)

                        if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
                            continue

                        key = (x1, y1, x2, y2, text_content)
                        if key in seen:
                            continue

                        seen.add(key)
                        results.append([x1, y1, x2, y2, text_content])
                    except ValueError:
                        continue
    else:
        # try parsing with regular expression
        
        list_content = predict_str
        items = re.findall(r'[\[\(]\s*([^\[\]\(\)]*?)\s*[\]\)]', list_content)

        if not items:
            return None

        for item in items:
            parts = item.split(',', 4)
            if len(parts) < 5:
                continue

            x1_str, y1_str, x2_str, y2_str, text_content = parts

            x1_str = x1_str.strip()
            y1_str = y1_str.strip()
            x2_str = x2_str.strip()
            y2_str = y2_str.strip()
            text_content = text_content.replace("\n", "").strip().strip('"').strip("'")

            try:
                x1 = int(x1_str)
                y1 = int(y1_str)
                x2 = int(x2_str)
                y2 = int(y2_str)

                if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
                    continue

                key = (x1, y1, x2, y2, text_content)
                if key in seen:
                    continue

                seen.add(key)
                results.append([x1, y1, x2, y2, text_content])
            except ValueError:
                continue

    if not results:
        return None

    return results


def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)
    task_type_list = [ "VQA with position en", "text grounding en", "text spotting en"]
    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        if data_item["type"] == "VQA with position en":  
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                pred_dict = convert_str_to_dict(data_item["predict"])
                data_item["score"] = vqa_with_position_evaluation(pred_dict, data_item)

        elif data_item["type"] == "text spotting en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_bounding_boxes_robust(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = spotting_evaluation(predict_bbox, data_item)

        elif data_item["type"] == "text grounding en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_coordinates(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = calculate_iou(predict_bbox, data_item["answers"])
        
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
        "--input_path", type=str, default="/home/wmk/code/OCR-R1/reward/IOU.json", help="Path to the input prediction JSON file."
    )
    parser.add_argument(
        "--output_path", type=str, default="/home/wmk/code/OCR-R1/reward/IOU_result.json", help="Path to save the results JSON file."
    )

    args = parser.parse_args()

    process_predictions(args.input_path, args.output_path)

print("End of Code!")
