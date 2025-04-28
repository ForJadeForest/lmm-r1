from datasets import load_dataset


ds = load_dataset("/path/to/COCO-Text")

# Filter for only one word ds
ds = ds.filter(lambda x: len(x['ocr_info']) == 1)

# Change the bounding box to a [x1, y1, x2, y2] format

def convert_to_absolute_bbox(item):
    result = item.copy()
    
    for i, ocr in enumerate(result['ocr_info']):
        # Extract normalized coordinates
        width = ocr['bounding_box']['width']
        height = ocr['bounding_box']['height']
        top_left_x = ocr['bounding_box']['top_left_x']
        top_left_y = ocr['bounding_box']['top_left_y']
        
        # Convert to absolute coordinates and round to integers
        x_min = round(top_left_x * result['image_width'])
        y_min = round(top_left_y * result['image_height'])
        x_max = round(x_min + (width * result['image_width']))
        y_max = round(y_min + (height * result['image_height']))
        
        # Replace with tuple format
        result['ocr_info'][i]['bounding_box'] = (x_min, y_min, x_max, y_max)
    
    return result

ds = ds.map(convert_to_absolute_bbox)


# 记录resize的尺寸和对应参数
from qwen_vl_utils import smart_resize

def get_resized_img_size(img, max_pixels=2000 * 28 * 28, min_pixels=4 * 28 * 28):
    width, height = img.size
    resized_height, resized_width = smart_resize(height, width, max_pixels=max_pixels, min_pixels=min_pixels)
    return resized_width, resized_height


ds = ds.map(
    lambda x: {
        "resized_img_size": get_resized_img_size(x["image"]),
        "resize_paras": {
            "max_pixels": 2000 * 28 * 28,
            "min_pixels": 4 * 28 * 28,
        },
        "original_img_size": x["image"].size,
    }
)


def transform_to_json(ds, save_dir):
    import json
    from tqdm import tqdm
    import os
    data = []
    img_save_dir = os.path.join(save_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)
    for i, item in enumerate(tqdm(ds, desc="Transforming to JSON", total=len(ds))):
        new_item = {}
        bounding_box = item['ocr_info'][0]['bounding_box']
        new_item['label'] = str(bounding_box)
        print(new_item['label'])
        new_item['meta_info'] = {
            "resized_img_size": item['resized_img_size'],
            "resize_paras": item['resize_paras'],
            "original_img_size": item['original_img_size'],
        }
        need_detection_word = item['ocr_info'][0]['word']
        # save image to file and use image path in message
        
        image_path = os.path.join("/lnt/workspace/pengyingzhe.py/code/dev/lmm-r1", img_save_dir, f"image_{i}.png")
        if not os.path.exists(image_path):
            item['image'].save(image_path)
        new_item['message'] = json.dumps([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": f"Where is the region of the text '{need_detection_word}'? Output its bbox coordinates using JSON format."},
                    {"type": "image", "image": image_path}
                ]
            }
        ])
        data.append(new_item)
    with open(os.path.join(save_dir, "data.json"), "w") as f:
        json.dump(data, f)
    return data


train_data = transform_to_json(ds['train'], "data/coco_text/coco_text_train_data")
validation_data = transform_to_json(ds['validation'], "data/coco_text/coco_text_validation_data")