import os
import sys
import json
import time
import logging
import gc
import traceback
import torch
from datetime import datetime
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from prompts import PROMPT_TEMPLATE

# ==================== Log Configuration ====================
def setup_logger(log_dir="logs"):
    """Configure logging system"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_inference_{timestamp}.log")
    
    logger = logging.getLogger("batch_inference")
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# ==================== InternVL Image Preprocessing ====================
PROMPT_TEMPLATE = """
You are a multimodal AI assistant specializing in precise physical reasoning and geometric estimation.

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
4. Start with `{{` and end with `}}`.

JSON Structure:
{{
  "reasoning": "Step-by-step derivation based strictly on visual evidence.",
  "answer": "The answer is [Exact_Value]"
}}

Question:
{question}
"""

def parse_json_response(response_text):
    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        return json.loads(cleaned_text)
    except Exception as e:
        return response_text

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ==================== Model Configuration ====================
MODEL_CONFIGS = [
    # # Transferred from 4090
    # {
    #     "model_path": "InternVL3_5-1B/OpenGVLab/InternVL3_5-1B",
    #     "model_type": "internvl",
    #     "name": "InternVL3_5-1B"
    # },   
    # {
    #     "model_path": "InternVL3_5-2B/OpenGVLab/InternVL3_5-2B",
    #     "model_type": "internvl",
    #     "name": "InternVL3_5-2B"
    # },
    # {
    #     "model_path": "InternVL3_5-8B/OpenGVLab/InternVL3_5-8B",
    #     "model_type": "internvl",
    #     "name": "InternVL3_5-8B"
    # },
    
    # {
    #     "model_path": "InternVL3-1B/OpenGVLab/InternVL3-1B",
    #     "model_type": "internvl",
    #     "name": "InternVL3-1B"
    # },    
    {
        "model_path": "InternVL3-2B/OpenGVLab/InternVL3-2B",
        "model_type": "internvl",
        "name": "InternVL3-2B"
    },
    # {
    #     "model_path": "InternVL3-8B/OpenGVLab/InternVL3-8B",
    #     "model_type": "internvl",
    #     "name": "InternVL3-8B"
    # },
    
    # {
    #     "model_path": "LLaVA-OV-7B/llava-hf/llava-onevision-qwen2-7b-ov-hf",
    #     "model_type": "llava_onevision",
    #     "name": "llava-onevision-qwen2-7b-ov-hf"
    # },
    
   
    
    # # InternVL Models
    # {
    #     "model_path": "InternVL3_5-14B",
    #     "model_type": "internvl",
    #     "name": "InternVL3_5-14B"
    # },
    {
        "model_path": "InternVL3_5-38B",
        "model_type": "internvl",
        "name": "InternVL3_5-38B"
    },
    {
        "model_path": "InternVL3-14B",
        "model_type": "internvl",
        "name": "InternVL3-14B"
    },
    # {
    #     "model_path": "InternVL3-38B",
    #     "model_type": "internvl",
    #     "name": "InternVL3-38B"
    # },
    {
        "model_path": "InternVL3-78B",
        "model_type": "internvl",
        "name": "InternVL3-78B"
    },
    # # LLaVA OneVision Models
    # {
    #     "model_path": "llava-onevision-qwen2-72b-ov-hf",
    #     "model_type": "llava_onevision",
    #     "name": "llava-onevision-qwen2-72b-ov-hf"
    # },
    # # LLaVA Next Models
    # {
    #     "model_path": "llava-next-72b-hf",
    #     "model_type": "llava_next",
    #     "name": "llava-next-72b-hf"
    # },
]

# ==================== Inference Functions ====================
def inference_internvl(model, tokenizer, image_path, question, logger):
    """InternVL Model Inference"""
    from transformers import GenerationConfig
    
    # Load image
    pixel_values = load_image_internvl(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    # Generation configuration
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    # Construct question format
    formatted_question = f"<image>\n{question}"
    
    # Inference
    if hasattr(model, 'module'):
        response = model.module.chat(tokenizer, pixel_values, formatted_question, generation_config)
    else:
        response = model.chat(tokenizer, pixel_values, formatted_question, generation_config)
    
    return response

def inference_llava_onevision(model, processor, image_path, question, logger):
    """LLaVA OneVision Model Inference"""
    # Load image
    image = Image.open(image_path)
    
    # Construct conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    
    # Apply template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    # Inference
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
        )
    
    # Decode
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract content after assistant
    if "assistant" in generated_text.lower():
        parts = generated_text.lower().split("assistant")
        if len(parts) > 1:
            # Find corresponding position in original text
            idx = generated_text.lower().rfind("assistant")
            response = generated_text[idx + len("assistant"):].strip()
        else:
            response = generated_text
    else:
        response = generated_text
    
    return response

def inference_llava_next(model, processor, image_path, question, logger):
    """LLaVA Next Model Inference"""
    # Load image
    image = Image.open(image_path)
    
    # Construct conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    
    # Apply template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    # Inference
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
        )
    
    # Decode
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract content after assistant
    if "assistant" in generated_text.lower():
        idx = generated_text.lower().rfind("assistant")
        response = generated_text[idx + len("assistant"):].strip()
    else:
        response = generated_text
    
    return response

# ==================== Model Loading Functions ====================
def load_internvl_model(model_path, logger):
    """Load InternVL Model"""
    from transformers import AutoModel, AutoTokenizer
    
    logger.info(f"Start loading InternVL model: {model_path}")
    start_time = time.time()
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2", # Enable flash attention
        device_map="auto"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    load_time = time.time() - start_time
    logger.info(f"InternVL model loaded, time taken: {load_time:.2f} seconds")
    
    return model, tokenizer

def load_llava_onevision_model(model_path, logger):
    """Load LLaVA OneVision Model"""
    from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
    
    logger.info(f"Start loading LLaVA OneVision model: {model_path}")
    start_time = time.time()
    
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", # Enable flash attention
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    load_time = time.time() - start_time
    logger.info(f"LLaVA OneVision model loaded, time taken: {load_time:.2f} seconds")
    
    return model, processor

def load_llava_next_model(model_path, logger):
    """Load LLaVA Next Model"""
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    
    logger.info(f"Start loading LLaVA Next model: {model_path}")
    start_time = time.time()
    
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", # Enable flash attention
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    load_time = time.time() - start_time
    logger.info(f"LLaVA Next model loaded, time taken: {load_time:.2f} seconds")
    
    return model, processor

# ==================== GPU Memory Cleanup Function ====================
def cleanup_gpu_memory(model=None, tokenizer_or_processor=None, logger=None):
    """
    Thoroughly clean GPU memory (for distributed models with device_map="auto")
    """
    if logger:
        logger.info("Cleaning GPU memory...")
        # Print VRAM before cleaning
        for i in range(torch.cuda.device_count()):
            mem_before = torch.cuda.memory_allocated(i) / 1024**3
            logger.info(f"  VRAM usage on GPU {i} before cleaning: {mem_before:.2f} GB")
    
    # 1. Attempt to move model to CPU
    if model is not None:
        try:
            if hasattr(model, 'to'):
                model.to('cpu')
        except:
            pass
        
        # 2. Clear accelerate hooks
        try:
            from accelerate.hooks import remove_hook_from_module
            for module in model.modules():
                remove_hook_from_module(module, recurse=True)
        except:
            pass
        
        # 3. Delete model
        del model
    
    # 4. Delete tokenizer/processor
    if tokenizer_or_processor is not None:
        del tokenizer_or_processor
    
    # 5. Force multiple garbage collections
    for _ in range(5):
        gc.collect()
    
    # 6. Synchronize all CUDA devices
    for i in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(i):
                torch.cuda.synchronize()
        except:
            pass
    
    # 7. Empty cache of all GPUs
    torch.cuda.empty_cache()
    
    # 8. Attempt to clean IPC memory
    try:
        torch.cuda.ipc_collect()
    except:
        pass
    
    # 9. Garbage collect and clear cache again
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()
    
    # 10. Wait to ensure memory is fully released
    time.sleep(5)
    
    # 11. Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print VRAM after cleaning
    if logger:
        for i in range(torch.cuda.device_count()):
            mem_after = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"  GPU {i} after cleaning: Allocated {mem_after:.2f} GB, Reserved {mem_reserved:.2f} GB")
        logger.info("GPU memory cleanup complete")

# ==================== Main Function ====================
def main():
    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Initialize logging
    logger, log_file = setup_logger(os.path.join(script_dir, "logs"))
    logger.info("=" * 60)
    logger.info("Batch inference script started")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    # Check CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    
    #################################################
    ### Modify paths
    #################################################
    ## Document paths
    # Create result directory
    result_dir = os.path.join(script_dir, "result_20test/20test_dick")
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Result directory: {result_dir}")
    
    # Load test data
    test_data_path = os.path.join(script_dir, "json/20test_new.json")
    logger.info(f"Loading test data: {test_data_path}")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Number of test data items: {len(test_data)}")
    
    # Record overall statistics
    total_start_time = time.time()
    all_results_summary = {}
    
    # Process each model sequentially
    for model_idx, model_config in enumerate(MODEL_CONFIGS):
        model_name = model_config["name"]
        model_path = model_config["model_path"]
        model_type = model_config["model_type"]
        
        logger.info("=" * 60)
        logger.info(f"[{model_idx + 1}/{len(MODEL_CONFIGS)}] Start processing model: {model_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model type: {model_type}")
        logger.info("=" * 60)
        
        # Check model path
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}, skipping this model")
            continue
        
        # Check VRAM status before loading
        logger.info("VRAM status before loading:")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: Allocated {mem_allocated:.2f} GB, Reserved {mem_reserved:.2f} GB, Total {mem_total:.2f} GB")
        
        # Load model
        model_load_start = time.time()
        try:
            if model_type == "internvl":
                model, tokenizer_or_processor = load_internvl_model(model_path, logger)
            elif model_type == "llava_onevision":
                model, tokenizer_or_processor = load_llava_onevision_model(model_path, logger)
            elif model_type == "llava_next":
                model, tokenizer_or_processor = load_llava_next_model(model_path, logger)
            else:
                logger.error(f"Unknown model type: {model_type}")
                continue
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            # Need to clear VRAM even on failure (partially loaded model)
            cleanup_gpu_memory(logger=logger)
            continue
        
        model_load_time = time.time() - model_load_start
        logger.info(f"Total model loading time: {model_load_time:.2f} seconds")
        
        # Store results of current model
        model_results = []
        inference_times = []
        
        # Iterate through test data
        for data_idx, data_item in enumerate(test_data):
            item_id = data_item["id"]
            image_path = data_item["image"]
            question = data_item["question"]
            
            logger.info(f"  [{data_idx + 1}/{len(test_data)}] Processing data ID: {item_id}")
            logger.debug(f"    Image path: {image_path}")
            logger.debug(f"    Question: {question}")
            
            # Inject Prompt
            full_question = PROMPT_TEMPLATE.format(question=question)
            
            # Check if image exists
            if not os.path.exists(image_path):
                logger.error(f"    Image does not exist: {image_path}")
                model_results.append({
                    "id": item_id,
                    "image": image_path,
                    "question": question,
                    "response": None,
                    "error": "Image does not exist",
                    "inference_time": 0
                })
                continue
            
            # Inference
            inference_start = time.time()
            try:
                if model_type == "internvl":
                    response = inference_internvl(model, tokenizer_or_processor, image_path, full_question, logger)
                elif model_type == "llava_onevision":
                    response = inference_llava_onevision(model, tokenizer_or_processor, image_path, full_question, logger)
                elif model_type == "llava_next":
                    response = inference_llava_next(model, tokenizer_or_processor, image_path, full_question, logger)
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                logger.info(f"    Inference completed, time taken: {inference_time:.2f} seconds")
                logger.debug(f"    Answer: {response[:200]}..." if len(response) > 200 else f"    Answer: {response}")
                
                # Parse JSON response
                parsed_response = parse_json_response(response)
                
                model_results.append({
                    "id": item_id,
                    "image": image_path,
                    "question": question,
                    "response": parsed_response,
                    "error": None,
                    "inference_time": round(inference_time, 2)
                })
                
            except Exception as e:
                inference_time = time.time() - inference_start
                logger.error(f"    Inference failed: {e}")
                logger.error(traceback.format_exc())
                
                model_results.append({
                    "id": item_id,
                    "image": image_path,
                    "question": question,
                    "response": None,
                    "error": str(e),
                    "inference_time": round(inference_time, 2)
                })
        
        # Statistics
        successful_count = sum(1 for r in model_results if r["error"] is None)
        failed_count = len(model_results) - successful_count
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        logger.info(f"Model {model_name} inference completed:")
        logger.info(f"  Successful: {successful_count}/{len(test_data)}")
        logger.info(f"  Failed: {failed_count}/{len(test_data)}")
        logger.info(f"  Average inference time: {avg_inference_time:.2f} seconds")
        
        # Save results
        result_file = os.path.join(result_dir, f"{model_name}_results.json")
        result_data = {
            "model_name": model_name,
            "model_path": model_path,
            "model_type": model_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": {
                "total": len(test_data),
                "successful": successful_count,
                "failed": failed_count,
                "avg_inference_time": round(avg_inference_time, 2),
                "model_load_time": round(model_load_time, 2)
            },
            "results": model_results
        }
        # Save results
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {result_file}")
        
        # Record in summary
        all_results_summary[model_name] = {
            "successful": successful_count,
            "failed": failed_count,
            "avg_inference_time": round(avg_inference_time, 2),
            "model_load_time": round(model_load_time, 2)
        }
        
        # Clean GPU memory
        cleanup_gpu_memory(model, tokenizer_or_processor, logger)
    
    # Overall statistics
    total_time = time.time() - total_start_time
    logger.info("=" * 60)
    logger.info("Batch inference completed!")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("=" * 60)
    
    # Save summary report
    summary_file = os.path.join(result_dir, "summary.json")
    summary_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": round(total_time, 2),
        "total_time_minutes": round(total_time / 60, 2),
        "test_data_count": len(test_data),
        "models_processed": len(all_results_summary),
        "model_results": all_results_summary
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Summary report saved to: {summary_file}")
    
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("Results summary:")
    logger.info("-" * 80)
    logger.info(f"{'Model Name':<40} {'Success':<8} {'Failed':<8} {'Avg Inf(s)':<12} {'Load(s)':<10}")
    logger.info("-" * 80)
    for model_name, stats in all_results_summary.items():
        logger.info(f"{model_name:<40} {stats['successful']:<8} {stats['failed']:<8} {stats['avg_inference_time']:<12} {stats['model_load_time']:<10}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()