
import os
import sys
import json
import time
import logging
import traceback
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

def setup_logger(log_dir="logs"):
    """
    Configure logging system
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_inference_qwen_{timestamp}.log")

    logger = logging.getLogger("batch_inference_qwen")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file

def load_model_and_processor(model_path):
    """
    Load model and processor
    """
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def run_inference(model, processor, data, logger):
    """
    Run inference
    """
    results = []
    for item in data:
        try:
            image_path = item["image"]
            prompt="""You are a multimodal AI assistant specializing in precise physical reasoning and geometric estimation.
You will be given:
1. An image containing physical objects.
2. A specific question about a target object in the image.
Your task is to answer the question by deriving the result strictly from the visual content provided.
================================================
Reasoning and Answering Guidelines
================================================
- You must first provide an explicit reasoning process explaining how you calculated or deduced the answer.
- Your reasoning must be strictly grounded in the observable pixel data and the geometric relationships within the image.
- **Do NOT** rely on generic parametric knowledge (e.g., "standard sizes") if it conflicts with or is unsupported by the visual data.
- You must demonstrate a logical path from visual observation to the final conclusion.
================================================
Important Constraints (READ CAREFULLY)
================================================
1. **SINGLE DETERMINISTIC VALUE:** Your final answer must be a single, exact value or category.
   - **NO Ranges:** Do not output "20-30cm". Output a single number like "25cm".
   - **NO Uncertainty:** Do not use words like "approximately", "about", "around", "maybe", or "estimated".
   - **NO Verbosity:** Do not write a full sentence after the required prefix.
2. **STRICT FORMATTING:**
   - The content of the `answer` field **MUST** start with the exact phrase: "**The answer is **".
   - Immediately following this phrase, output **ONLY** the value (and unit if applicable).
   - Example (Correct): "The answer is 14.5cm"
   - Example (Correct): "The answer is Winter"
   - Example (WRONG): "The answer is about 14.5cm"
   - Example (WRONG): "The answer is 14cm to 15cm"
   - Example (WRONG): "The answer is the bottle is 14.5cm tall"
3. **MANDATORY ANSWER:**
   - You **MUST** provide a result. Do not answer "Unknown" or "Cannot determine". If the visual evidence is subtle, provide your best specific estimation.
================================================
Output Format (JSON ONLY)
================================================
1. Output **ONLY** a raw JSON object.
2. **Do NOT** use Markdown code blocks (no ```json).
3. **Do NOT** include any introductory or concluding text.
4. Start with `{` and end with `}`.
JSON Structure:
{
  "reasoning": "Step-by-step derivation based strictly on visual evidence.",
  "answer": "The answer is [Exact_Value]"
}
Question:
"""
            question = prompt + item["question"]

            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                results.append({"id": item["id"], "error": "Image not found"})
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

            start_time = time.time()
            generated_ids = model.generate(**inputs, max_new_tokens=4096)
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            inference_time = time.time() - start_time

            results.append({
                "id": item["id"],
                "image": image_path,
                "question": question,
                "response": output_text[0],
                "error": None,
                "inference_time": round(inference_time, 2)
            })
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            logger.error(traceback.format_exc())
            results.append({"id": item["id"], "error": str(e)})

    return results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    logger, log_file = setup_logger(os.path.join(script_dir, "logs"))
    logger.info("=" * 60)
    logger.info("Qwen model batch inference script started")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    result_dir = os.path.join(script_dir, "result_experiment_qwen/geometric")
    os.makedirs(result_dir, exist_ok=True)

    ###################################
    ## Modify input JSON
    ###################################
    test_data_path = os.path.join(script_dir, "json/geometric.json")
    if not os.path.exists(test_data_path):
        logger.error(f"Test data file does not exist: {test_data_path}")
        return

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    logger.info(f"Number of test data items: {len(test_data)}")

    model_configs = [
        {"name": "Qwen3-VL-2B", "path": "Qwen3-VL-2B/Qwen/Qwen3-VL-2B-Instruct"},
        {"name": "Qwen3-VL-4B", "path": "Qwen3-VL-4B/Qwen/Qwen3-VL-4B-Instruct"},
        {"name": "Qwen2.5-VL-3B", "path": "Qwen2.5-VL-3B/Qwen/Qwen2.5-VL-3B-Instruct"},
        # {"name": "Qwen3-VL-8B", "path": "Qwen3-VL-8B"},
    ]

    for config in model_configs:
        model_name = config["name"]
        model_path = config["path"]

        logger.info(f"Loading model: {model_name}")
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            continue

        try:
            model, processor = load_model_and_processor(model_path)
            results = run_inference(model, processor, test_data, logger)

            result_file = os.path.join(result_dir, f"{model_name}_results.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Results saved to: {result_file}")
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()