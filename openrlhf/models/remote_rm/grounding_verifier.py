import json
import os
import random
import re
from argparse import ArgumentParser
import numpy as np
from flask import Flask, jsonify, request
from loguru import logger

app = Flask(__name__)

dataset_entries = []  # Will store the dataset entries
message_to_entry = {}  # Dictionary for fast lookup: message -> entry

def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<|end_of_sentence|>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()

def extract_json_from_response(response):
    """Extract JSON content from the response."""
    # Look for JSON block in markdown format
    json_match = re.search(r'```(?:json)?\n(.*?)```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # If no markdown format, try to find JSON directly
        json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    
    try:
        return json.loads(json_str)
    except:
        return None

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Both boxes should be in [x1, y1, x2, y2] format.
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def adjust_bbox_to_original(bbox, meta_info):
    """
    Adjust the bounding box from resized image coordinates to original image coordinates.
    
    bbox: [x1, y1, x2, y2] in resized image coordinates
    meta_info: Dictionary containing resize information
    """
    resized_width, resized_height = meta_info["resized_img_size"]
    original_width, original_height = meta_info["original_img_size"]
    
    # Calculate scale factors
    width_scale = original_width / resized_width
    height_scale = original_height / resized_height
    
    # Apply scaling to each coordinate
    original_bbox = [
        int(bbox[0] * width_scale),
        int(bbox[1] * height_scale),
        int(bbox[2] * width_scale),
        int(bbox[3] * height_scale)
    ]
    
    return original_bbox

def find_dataset_entry(message_content):
    """
    Find the corresponding dataset entry based on the message content using dictionary lookup.
    
    message_content: The message content string
    returns: The dataset entry with meta_info or None if not found
    """
    # Direct dictionary lookup - O(1) operation
    if message_content in message_to_entry:
        return message_to_entry[message_content]
    
    # Try to parse the message if it's a JSON string
    try:
        parsed_message = json.dumps(json.loads(message_content))
        if parsed_message in message_to_entry:
            return message_to_entry[parsed_message]
    except:
        pass
    
    # Fallback to approximate matching if dictionary lookup fails
    logger.warning("No exact match found in dictionary. Trying approximate matching.")
    for entry_message, entry in message_to_entry.items():
        if message_content in entry_message or entry_message in message_content:
            return entry
    
    logger.error(f"No matching dataset entry found for message: {message_content[:100]}...")
    return None

def verify_grounding(response, label, meta_info):
    """
    Verify if the predicted bounding box has IoU > 0.5 with the ground truth.
    
    response: The model's response string
    label: Ground truth bounding box [x1, y1, x2, y2]
    meta_info: Dictionary containing resize information
    """
    # Extract JSON content from the response
    json_data = extract_json_from_response(response)
    if not json_data:
        return 0.0
    
    # Count number of bounding boxes in the prediction
    bbox_count = sum(1 for pred in json_data if "bbox_2d" in pred)
    
    # If more than one bounding box is detected, return 0.0 reward
    if bbox_count > 1:
        logger.warning(f"Multiple bounding boxes detected ({bbox_count}). Returning 0.0 reward.")
        return 0.0
    
    # Get resized dimensions from meta_info
    resized_width, resized_height = meta_info["resized_img_size"]
    
    # Parse the ground truth label
    # The ground truth is already in the original image coordinates
    gt_bbox = eval(label)
    # Assert ground truth format is valid
    assert len(gt_bbox) == 4, f"Unexpected ground truth format: {gt_bbox}"
    assert gt_bbox[2] >= gt_bbox[0] and gt_bbox[3] >= gt_bbox[1], f"Invalid ground truth box: {gt_bbox}"
    
    # Go through each predicted box
    best_iou = 0.0
    for pred in json_data:
        if "bbox_2d" in pred:
            pred_bbox = pred["bbox_2d"]
            
            # Validate bounding box format
            if len(pred_bbox) != 4:
                logger.warning(f"Skipping invalid bbox format: {pred_bbox}")
                continue
                
            # Check if the box coordinates are valid
            if pred_bbox[2] < pred_bbox[0] or pred_bbox[3] < pred_bbox[1]:
                # Invalid box, skip
                logger.warning(f"Skipping invalid predicted box: {pred_bbox}")
                continue

            # Adjust bbox to original image size
            adjusted_pred_bbox = adjust_bbox_to_original(pred_bbox, meta_info)
            
            # Calculate IoU
            iou = calculate_iou(adjusted_pred_bbox, gt_bbox)
            best_iou = max(best_iou, iou)
    
    # Return 1.0 if IoU > 0.5, otherwise 0.0
    return 1.0 if best_iou > 0.5 else 0.0

@app.route("/get_reward", methods=["POST"])
def get_reward():
    # Get JSON data from request
    data = request.get_json()
    # Check for required fields
    if "query" not in data or "prompts" not in data or "labels" not in data:
        return jsonify({"error": "query, prompts, and labels fields are required"}), 400
    
    rewards = []
    for q, prompt_str, label in zip(data["query"], data["prompts"], data["labels"]):
        try:
            # prompt_str is the message content
            # Find the corresponding dataset entry with meta_info using fast dictionary lookup
            dataset_entry = find_dataset_entry(prompt_str)
            
            if not dataset_entry:
                logger.error(f"Could not find dataset entry for message: {prompt_str[:100]}...")
                rewards.append(0.0)
                continue
            
            # Get meta_info from the dataset entry
            meta_info = dataset_entry.get("meta_info", {})
            if not meta_info:
                logger.error(f"No meta_info found in dataset entry")
                rewards.append(0.0)
                continue
            
            # Get model response
            response = get_response_from_query(q) or q
            if response is None:
                logger.error(f"Response not found from {q}")
                rewards.append(0.0)
                continue
            
            # Verify grounding and calculate reward
            reward = verify_grounding(response, label, meta_info)
            
            # Randomly log some examples
            if random.randint(1, 2) == 1:
                info = f"Query: {q}\n\nPrompt: {prompt_str[:200]}...\n\nLabel: {label}\n\nResponse: {response}\n\nReward: {reward}\n\n"
                info = re.sub(r"<\|.*?\|>", "", info)
                logger.info(info)
            
            rewards.append(reward)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            rewards.append(0.0)
    
    # Return rewards
    return jsonify({"rewards": rewards})

def load_dataset(dataset_path):
    """Load dataset from file and build dictionary for fast lookup."""
    global dataset_entries, message_to_entry
    
    if dataset_path.endswith("json"):
        with open(dataset_path, "r") as f:
            dataset_entries = json.load(f)
    elif dataset_path.endswith("jsonl"):
        dataset_entries = []
        with open(dataset_path, "r") as f:
            for line in f:
                dataset_entries.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format for dataset: {dataset_path}")
    
    # Build dictionary for fast lookup
    message_to_entry = {}
    for entry in dataset_entries:
        if "message" in entry:
            message = entry["message"]
            message_to_entry[message] = entry
            
            # Also try to normalize JSON strings for more robust matching
            try:
                if isinstance(message, str):
                    parsed = json.loads(message)
                    normalized = json.dumps(parsed)
                    message_to_entry[normalized] = entry
            except:
                pass
    
    logger.info(f"Loaded {len(dataset_entries)} entries from dataset {dataset_path}")
    logger.info(f"Built lookup dictionary with {len(message_to_entry)} keys")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--prompt-template", type=str, default="chatml", help="Prompt template"
    )
    parser.add_argument("--log_file", type=str, default="grounding_verifier.log", help="Log file path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (json or jsonl)")
    args = parser.parse_args()
    
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    logger.remove()
    logger.add(args.log_file, level="DEBUG")
    
    # Load dataset and build lookup dictionary
    load_dataset(args.dataset)
    
    # Define response prefix based on prompt template
    if args.prompt_template == "chatml":
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template == "qwen1":
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template == "base":
        response_prefix = r"Assistant: "
    elif args.prompt_template == "phi3" or args.prompt_template == "phi4":
        response_prefix = r"<|assistant|>\n"
    else:
        raise ValueError(f"Unknown chat format: {args.prompt_template}")
    
    logger.info(f"Starting grounding verification service with prompt template: {args.prompt_template}")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)