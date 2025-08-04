from utils import EM_score, RL_score, logger
import json
import os
from typing import Dict, List, Tuple

def calculate_metrics(answer_folder: str, dataset_name: str) -> Tuple[float, float, int]:
    """
    计算指定数据集的评估指标
    
    Args:
        answer_folder: 答案文件夹路径
        dataset_name: 数据集名称 (NovelQA/InfiniteChoice/InfiniteQA/NarrativeQA)
    
    Returns:
        Tuple[float, float, int]: (平均EM分数, 平均RL分数, 总问题数)
    """
    answer_folder_path = os.path.join(answer_folder, dataset_name)
    logger.info(f"answer_folder: {answer_folder_path}")
    
    total_em_score = 0
    total_rl_score = 0
    total_num = 0
    
    for file in os.listdir(answer_folder_path):
        if not file.endswith(".json"):
            continue
            
        with open(os.path.join(answer_folder_path, file), "r") as f:
            answer = json.load(f)
            
        for qa in answer:
            if dataset_name == "InfiniteQALoader":
                max_rl_score = 0
                max_em_score = 0
                for ans in qa["answer"]:
                    # take the first sentence as the answer, as the ground truth is a single sentence.
                    max_rl_score = max(max_rl_score, RL_score(ans, qa["output_text"].split("\n")[0].split(".")[0].strip()))
                    max_em_score = max(max_em_score, EM_score(ans, qa["output_text"].split("\n")[0].split(".")[0].strip()))
                total_em_score += max_em_score
                total_rl_score += max_rl_score
            elif dataset_name == "NarrativeQA":
                candidate_em_score = []
                candidate_rl_score = []
                for ans in qa["answer"]:
                    candidate_em_score.append(EM_score(ans, qa["output_text"]))
                    candidate_rl_score.append(RL_score(ans, qa["output_text"]))
                total_em_score += max(candidate_em_score)
                total_rl_score += max(candidate_rl_score)
            else:
                total_em_score += EM_score(qa["answer"], qa["output_text"])
                total_rl_score += RL_score(qa["answer"], qa["output_text"])
            total_num += 1
    
    return total_em_score / total_num, total_rl_score / total_num, total_num

def calculate_time_cost(cache_folder: str, dataset_name: str) -> Tuple[float, float, int]:
    """
    计算时间开销
    
    Args:
        cache_folder: 缓存文件夹路径
        dataset_name: 数据集名称
    
    Returns:
        Tuple[float, float, int]: (平均build时间, 平均extract时间, 总样本数)
    """
    cache_folder_path = os.path.join(cache_folder, dataset_name)
    build_time = 0
    extract_time = 0
    total_count = 0
    
    for folder in os.listdir(cache_folder_path):
        try:
            with open(os.path.join(cache_folder_path, folder, "time_cost.txt"), "r") as f:
                time_cost = f.readlines()
                total_count += 1
                for line in time_cost:
                    if "build" in line:
                        build_time += float(line.split("||")[1])
                    elif "extract" in line:
                        extract_time += float(line.split("||")[1])
        except Exception as e:
            logger.error(f"Error reading time_cost.txt for folder {folder}: {e}")
            continue
    
    return build_time / total_count, extract_time / total_count, total_count