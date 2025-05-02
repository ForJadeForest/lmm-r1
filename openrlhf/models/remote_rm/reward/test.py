import requests
import json

def get_reward_from_server(question: str, image_path: str, prediction: str,
                           url: str = "http://127.0.0.1:5000/get_reward",
                           headers: dict = None) -> float:
    if headers is None:
        headers = {"Content-Type": "application/json"}

    payload = {
        "query": [f"<answer>{prediction}</answer>"],
        "prompts": [
            json.dumps(
                {"message": question, "image_path": image_path},
                ensure_ascii=False,
                separators=(",", ":")
            )
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload, ensure_ascii=False))
    response.raise_for_status()

    data = response.json()
    rewards = data.get("rewards", [])
    if not rewards:
        raise ValueError("No rewards returned from server")

    return rewards[0]

def load_json_file(file_path: str) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}.")

if __name__ == "__main__":
    json_file_path = "/home/wmk/data/internvl2_5_26b.json"
    
    try:
        data = load_json_file(json_file_path)
    except Exception as e:
        print(f"Failed to load JSON file: {str(e)}")
        exit(1)

    for item in data:
        try:
            reward = get_reward_from_server(
                question=item["question"],
                image_path=item["image_path"],
                prediction=item["predict"]
            )
            print(f"ID: {item['id']} | Prediction: '{item['predict']}' | Reward: {reward}")
        except KeyError as e:
            print(f"Missing required field in item {item.get('id', 'unknown')}: {str(e)}")
        except Exception as e:
            print(f"Error processing ID {item.get('id', 'unknown')}: {str(e)}")

# if __name__ == "__main__":
#     question = "what is the radio program focus on? Output the answer with 'answer' and 'bbox'. 'bbox' refers to the bounding box position of the 'answer' content in the image. The output format is \"answer:gt, bbox:(x1,y1,x2,y2)\", where the bbox is the coordinates of the top-left corner and the bottom-right corners. The 'bbox' should be normalized coordinates ranging from 0 to 1000 by the image width and height.\nYour answer should be in the JSON format:\n{\n    \"answer\": \"..\",  # The answer\n    \"bbox\": \"(x1,y1,x2,y2)\"     # The bounding box position of the 'answer'\n}\n"
#     image_path = "EN_part/vqa-pos_textvqa/textvqa_0085.jpg"
#     prediction = "{\n    \"answer\": \"Sport\", \n    \"bbox\": \"(0.2,0.3,0.4,0.5)\"\n}"

#     reward = get_reward_from_server(question, image_path, prediction)
#     print(f"Reward for prediction '{prediction}': {reward}")