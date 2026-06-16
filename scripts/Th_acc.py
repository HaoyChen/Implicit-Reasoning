import json
import re
import argparse
import sys

def clean_model_output(response_data):
    if not response_data:
        return None

    ans = None

    if isinstance(response_data, dict):
        if "answer" in response_data:
            ans = str(response_data["answer"])
            
    elif isinstance(response_data, str):
        cleaned_str = re.sub(r'```json\s*', '', response_data)
        cleaned_str = re.sub(r'```\s*', '', cleaned_str)

        try:
            data = json.loads(cleaned_str)
            if isinstance(data, dict) and "answer" in data:
                ans = str(data["answer"])
        except json.JSONDecodeError:

            match = re.search(r'"answer"\s*:\s*"([^"]+)"', cleaned_str, re.IGNORECASE)
            if match:
                ans = str(match.group(1))

    if ans is None:
        return None 

    ans = ans.strip()
    prefixes_to_remove = ["The answer is", "the answer is", "Answer is", "answer is", "The answer:"]
    for p in prefixes_to_remove:
        if ans.lower().startswith(p.lower()):
            ans = ans[len(p):].strip()
            break

    ans = ans.strip('.\'" ')
    return ans

def evaluate_pair(gt_str, pred_str):
    gt_clean = str(gt_str).strip().lower()
    pred_clean = str(pred_str).strip().lower()

    scores = {5: 0.0, 10: 0.0, 20: 0.0, 30: 0.0}

    if not gt_clean or not pred_clean:
        if gt_clean == pred_clean:
            return {5: 1.0, 10: 1.0, 20: 1.0, 30: 1.0}
        return scores

    num_pattern = re.compile(r'^([-+]?\d*\.?\d+)\s*([a-z0-9\^³²°%]*)$')
    gt_match = num_pattern.match(gt_clean)

    if gt_match:
        gt_val = float(gt_match.group(1))

        pred_match = re.search(r'([-+]?\d*\.?\d+)', pred_clean)
        if pred_match:
            pred_val = float(pred_match.group(1))

            if gt_val == 0:

                acc = 1.0 if pred_val == 0 else 0.0
                return {5: acc, 10: acc, 20: acc, 30: acc}
            else:
                error = abs(gt_val - pred_val) / abs(gt_val)
                return {
                    5: 1.0 if error <= 0.05 else 0.0,
                    10: 1.0 if error <= 0.10 else 0.0,
                    20: 1.0 if error <= 0.20 else 0.0,
                    30: 1.0 if error <= 0.30 else 0.0,
                }

    if gt_clean in pred_clean or pred_clean in gt_clean:
        return {5: 1.0, 10: 1.0, 20: 1.0, 30: 1.0}

    return scores

def main():
    parser = argparse.ArgumentParser(description="Vision-Language Model accuracy evaluation script")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth JSON file")
    parser.add_argument("--pred", type=str, required=True, help="Path to the model prediction JSON file")
    parser.add_argument("--out_txt", type=str, default="Accuracy_Evaluation_Report.txt", help="Filename for the output summary report text file")
    parser.add_argument("--out_json", type=str, default="Detailed_Comparison_Results.json", help="Filename for the output detailed comparison records JSON file")
    args = parser.parse_args()



    try:
        with open(args.gt, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(args.pred, 'r', encoding='utf-8') as f:
            raw_pred_data = json.load(f)
    except Exception as e:
        print(f"file open failed: {e}")
        sys.exit(1)
        
    if isinstance(raw_pred_data, dict) and "results" in raw_pred_data:
        pred_data = raw_pred_data["results"]
    elif isinstance(raw_pred_data, list):
        pred_data = raw_pred_data
    else:
        print("error: Unrecognizable prediction file format")
        sys.exit(1)


    gt_map = {str(item["id"]): str(item["gt"]) for item in gt_data if "id" in item and "gt" in item}

    total_samples = 0
    valid_evaluated = 0
    correct = {5: 0, 10: 0, 20: 0, 30: 0}
    
    category_stats = {}
    
    detailed_results_log = []

    for item in pred_data:
        pred_id = str(item.get("id", ""))
        
        base_id = pred_id[:-4] if pred_id.endswith("_cot") else pred_id

        if base_id not in gt_map:
            continue
            
        task_match = re.match(r'^([a-zA-Z]+)', base_id)
        task_type = task_match.group(1) if task_match else "Unknown"
        

        if task_type not in category_stats:
            category_stats[task_type] = {
                "total": 0,
                "correct": {5: 0, 10: 0, 20: 0, 30: 0}
            }

        total_samples += 1
        category_stats[task_type]["total"] += 1
        gt_val = gt_map[base_id]
        
        raw_response = item.get("response")
        pred_ans = clean_model_output(raw_response)
        
        record_entry = {
            "original_id": pred_id,
            "task_type": task_type,
            "ground_truth": gt_val,
            "extracted_prediction": pred_ans if pred_ans is not None else "[EXTRACTION FAILED]",
            "raw_model_response": raw_response 
        }

        if pred_ans is not None:
            valid_evaluated += 1
            scores = evaluate_pair(gt_val, pred_ans)
            for threshold in [5, 10, 20, 30]:
                correct[threshold] += scores[threshold]
                category_stats[task_type]["correct"][threshold] += scores[threshold]
                record_entry[f"is_correct_at_{threshold}%"] = bool(scores[threshold])
        else:
            for threshold in [5, 10, 20, 30]:
                record_entry[f"is_correct_at_{threshold}%"] = False
                
        detailed_results_log.append(record_entry)


    if total_samples > 0:
        acc_5 = (correct[5] / total_samples) * 100
        acc_10 = (correct[10] / total_samples) * 100
        acc_20 = (correct[20] / total_samples) * 100
        acc_30 = (correct[30] / total_samples) * 100
    else:
        acc_5 = acc_10 = acc_20 = acc_30 = 0.0


    report = f"""==================================================
          Accuracy Evaluation Report
==================================================
Total Samples Processed : {total_samples}
Valid Evaluated Samples : {valid_evaluated} (Failed Extr. Counted as 0%)
--------------------------------------------------
Global Accuracy @ 5%    : {acc_5:6.2f} %
Global Accuracy @ 10%   : {acc_10:6.2f} %
Global Accuracy @ 20%   : {acc_20:6.2f} %
Global Accuracy @ 30%   : {acc_30:6.2f} %
==================================================
          Per-Task Accuracy Breakdown
==================================================\n"""


    for task, stats in sorted(category_stats.items()):
        t_total = stats["total"]
        if t_total > 0:
            t_acc_5 = (stats["correct"][5] / t_total) * 100
            t_acc_10 = (stats["correct"][10] / t_total) * 100
            t_acc_20 = (stats["correct"][20] / t_total) * 100
            t_acc_30 = (stats["correct"][30] / t_total) * 100
            
            report += f"Task: {task:<12} (Total: {t_total})\n"
            report += f"  @ 5%: {t_acc_5:6.2f}% | @ 10%: {t_acc_10:6.2f}% | @ 20%: {t_acc_20:6.2f}% | @ 30%: {t_acc_30:6.2f}%\n"
            report += "-" * 50 + "\n"

    print(report)
    
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(report)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(detailed_results_log, f, ensure_ascii=False, indent=2)
        
    print(f"\n[INFO] the summary report has been saved to: {args.out_txt}")
    print(f"[INFO] the detailed comparison results have been saved to: {args.out_json}")

if __name__ == "__main__":
    main()