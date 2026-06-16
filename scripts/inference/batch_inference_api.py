import json
import base64
import os
import asyncio
import aiohttp
import time
import random
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any
from prompts import PROMPT_TEMPLATE

# Configurations
CONFIGS = [ 
    {
        "name": "openai",
        "base_url": "https://xxxxxxxxx",
        "api_key": "sk-xxxxxxxxx",
        "concurrency_limit": 2, 
        "request_delay": 60.0, 
        "models": [
            # "gpt-5.2-2025-12-11",
            # "claude-sonnet-4-5-20250929",
            # "claude-sonnet-4-5-20250929-thinking",
            # "gemini-3-flash-preview",
            "gemini-3-pro"
        ]
    },
]

INPUT_DIR = "json"
INPUT_FILES = ["contectual_inference.json", "physical_logic.json", "physical_property.json", "geometric.json"]
# INPUT_FILES = ["test.json"]
BASE_OUTPUT_DIR = "result_api"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_media_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.webp':
        return 'image/webp'
    elif ext == '.gif':
        return 'image/gif'
    return 'image/jpeg'

import re

def parse_json_response(response_text):
    try:
        # Basic cleanup
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
            
        cleaned_text = cleaned_text.strip()
        
        # Attempt direct parse
        return json.loads(cleaned_text)
    except Exception:
        # Fallback 1: Try to fix common LaTeX/escape issues
        try:
            fixed_text = cleaned_text.replace(r'\(', '(').replace(r'\)', ')').replace(r'\[', '[').replace(r'\]', ']')
            fixed_text = fixed_text.replace(r'\frac', 'frac').replace(r'\times', '*').replace(r'\approx', '~')

            
            return json.loads(fixed_text)
        except Exception:
            pass

        # Fallback 2: Try to find the first valid JSON object
        try:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response_text[start:end+1]
                # Try to fix escapes in the extracted string too
                try:
                    return json.loads(json_str)
                except:
                     fixed_json_str = json_str.replace(r'\(', '(').replace(r'\)', ')').replace(r'\[', '[').replace(r'\]', ']')
                     return json.loads(fixed_json_str)
        except Exception:
            pass
        
        return response_text

async def fetch_inference(session: aiohttp.ClientSession, 
                          config: Dict, 
                          model: str, 
                          item: Dict, 
                          semaphore: asyncio.Semaphore,
                          logger: logging.Logger) -> Dict:
    
    image_path = item['image']
    question = item['question']
    
    # Apply Prompt Template
    full_question = PROMPT_TEMPLATE.format(question=question)
    
    item_id = item['id']
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"[{model}] Image file not found: {image_path}")
        return {
            "id": item_id,
            "error": "Image file not found"
        }

    # Prepare payload
    try:
        base64_image = encode_image(image_path)
        media_type = get_image_media_type(image_path)
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }

        async with semaphore:
            request_delay = config.get('request_delay', 0)
            if request_delay > 0:
                actual_delay = request_delay * (0.5 + random.random())
                await asyncio.sleep(actual_delay)

            for attempt in range(3):
                try:
                    async with session.post(config['base_url'], json=payload, headers=headers, timeout=600) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            if 'choices' in response_json and len(response_json['choices']) > 0:
                                content = response_json['choices'][0]['message']['content']
                                parsed_content = parse_json_response(content)
                                return {
                                    "id": item_id,
                                    "question": question,
                                    "response": parsed_content,
                                    "raw_response": content,
                                    "model": model
                                }
                            else:
                                logger.error(f"[{model}] Unexpected response format for {item_id}: {response_json}")
                                return {
                                    "id": item_id,
                                    "error": "Unexpected response format",
                                    "raw": response_json
                                }
                        elif response.status == 429:
                            logger.warning(f"[{model}] Rate limit hit for {item_id}, retrying in {2 ** attempt}s...")
                            await asyncio.sleep(2 ** attempt)
                        else:
                            error_text = await response.text()
                            logger.error(f"[{model}] Error {response.status} for {item_id}: {error_text[:200]}")
                            # Store the last error to return if all retries fail
                            last_error = f"HTTP {response.status}: {error_text}"
                            # Increased backoff for 500 errors or other non-429 errors
                            await asyncio.sleep(2 * (attempt + 1))
                except Exception as e:
                    logger.warning(f"[{model}] Exception for {item_id} (Attempt {attempt+1}): {e}")
                    last_error = str(e)
                    await asyncio.sleep(1)
            
            logger.error(f"[{model}] Failed after retries for {item_id}. Last error: {last_error if 'last_error' in locals() else 'Unknown error'}")
            return {
                "id": item_id,
                "error": f"Failed after retries. Last error: {last_error if 'last_error' in locals() else 'Unknown error'}"
            }

    except Exception as e:
        logger.error(f"[{model}] Critical Error for {item_id}: {e}")
        return {
            "id": item_id,
            "error": str(e)
        }

async def process_model(model: str, config: Dict, test_data: List[Dict], semaphore: asyncio.Semaphore, output_dir: str, logger: logging.Logger):
    logger.info(f"Starting model: {model}")
    safe_model_name = model.replace("/", "_")
    output_file = os.path.join(output_dir, f"result_{safe_model_name}.json")
    
    if os.path.exists(output_file):
         logger.info(f"Skipping {model}, output file already exists: {output_file}")
         return

    async with aiohttp.ClientSession() as session:
        tasks = []
        total_items = len(test_data)
        completed_count = 0
        
        # Track statistics
        stats = {
            "total": total_items,
            "successful": 0,
            "failed": 0,
            "avg_inference_time": 0.0,
            "total_inference_time": 0.0
        }
        
        async def process_item_wrapper(item):
            nonlocal completed_count
            result = await fetch_inference(session, config, model, item, semaphore, logger)
            completed_count += 1
            if completed_count % 5 == 0 or completed_count == total_items:
                logger.info(f"[{model}] Progress: {completed_count}/{total_items} ({(completed_count/total_items)*100:.1f}%)")
            return result

        for item in test_data:
            task = process_item_wrapper(item)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_results = [r for r in results if "error" not in r]
        stats["successful"] = len(successful_results)
        stats["failed"] = len(results) - len(successful_results)
        stats["avg_inference_time"] = round(total_time / len(results), 2) if results else 0
        
        # Prepare final output structure
        final_output = {
            "model_name": model,
            "model_type": "api", # Assuming API models
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": stats,
            "results": results
        }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        logger.info(f"Finished model: {model} (Saved to {output_file})")

async def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("BatchInference")

    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)

    print(f"Processing {len(INPUT_FILES)} datasets: {INPUT_FILES}")

    for filename in INPUT_FILES:
        dataset_name = os.path.splitext(filename)[0]
        input_path = os.path.join(INPUT_DIR, filename)
        output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n{'='*50}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"Input: {input_path}")
        print(f"Output Directory: {output_dir}")
        print(f"{'='*50}")

        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            continue

        with open(input_path, 'r') as f:
            test_data = json.load(f)

        tasks = []
        
        for config in CONFIGS:
            semaphore = asyncio.Semaphore(config.get("concurrency_limit", 10))
            
            for model in config['models']:
                tasks.append(process_model(model, config, test_data, semaphore, output_dir, logger))
        
        print(f"Launching {len(tasks)} model tasks for {dataset_name}...")
        await asyncio.gather(*tasks)
        print(f"Completed processing for {dataset_name}")

    print("\nAll datasets processed.")

if __name__ == "__main__":
    asyncio.run(main())
