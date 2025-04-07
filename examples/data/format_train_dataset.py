import json
import argparse
from pathlib import Path


def process_dataset(
    input_file: str,
    old_image_root: str,
    new_image_root: str,
    output_file: str = None,
    change_system_prompt: bool = False,
    system_prompt: str = None,
):
    """
    Process the training dataset by:
    1. Replacing or removing system prompts
    2. Replacing image root paths
    3. Keeping answer field unchanged

    Args:
        input_file: Path to the input JSONL file
        old_image_root: Original image root path to be replaced
        new_image_root: New image root path to replace with
        output_file: Path to output JSONL file. If None, will use input filename with .processed.jsonl suffix
        change_system_prompt: Whether to change system prompt
        system_prompt: New system prompt to use if use_system_prompt is True
    """
    if output_file is None:
        output_file = str(Path(input_file).with_suffix(".processed.jsonl"))

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line.strip())

            # Parse the messages string into a list
            messages = json.loads(data["messages"])

            # Process each message
            processed_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    if change_system_prompt and system_prompt:
                        processed_messages.append({"role": "system", "content": system_prompt})
                    else:
                        processed_messages.append(msg)

                elif msg["role"] == "user":
                    # Process user content which may contain image paths
                    processed_content = []
                    for content in msg["content"]:
                        if content["type"] == "image":
                            # Replace image path
                            new_path = content["image"].replace(old_image_root, new_image_root)
                            processed_content.append({"type": "image", "image": new_path})
                        else:
                            processed_content.append(content)
                    processed_messages.append({"role": "user", "content": processed_content})

            # Create new data entry
            new_data = {"messages": json.dumps(processed_messages), "answer": data["answer"]}

            # Write to output file
            f_out.write(json.dumps(new_data) + "\n")

    print(f"Processed dataset saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process training dataset")
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("old_image_root", type=str, help="Original image root path")
    parser.add_argument("new_image_root", type=str, help="New image root path")
    parser.add_argument("--output_file", type=str, help="Path to output JSONL file", default=None)
    parser.add_argument("--change_system_prompt", action="store_true", help="Whether to keep system prompt")
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="New system prompt to use if use_system_prompt is True",
        default="You are a helpful assistant.",
    )

    args = parser.parse_args()

    process_dataset(
        args.input_file,
        args.old_image_root,
        args.new_image_root,
        args.output_file,
        args.change_system_prompt,
        args.system_prompt,
    )


if __name__ == "__main__":
    main()
