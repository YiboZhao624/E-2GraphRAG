from typing import List
from transformers import AutoTokenizer
from rouge import Rouge

def sequential_split(text:str, tokenizer:AutoTokenizer,
                     length:int, overlap:int)->List[str]:
    '''
    Split the text into chunks of length length with overlap.
    '''
    chunks = []
    text_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    for i in range(0, len(text_ids), length - overlap):
        chunk = tokenizer.decode(text_ids[i:i+length])
        chunks.append(chunk)
    return chunks

def EM_score(pred, gold):
    return standardize_answer(pred) == standardize_answer(gold)

def standardize_answer(answer):
    return answer.strip().lower()

def RL_score(pred, gold):
    """
    Calculate Rouge-L score using rouge library
    Args:
        pred: prediction string
        gold: gold reference string
    Returns:
        float: Rouge-L F1 score
    """
    # Standardize inputs
    pred = standardize_answer(pred)
    gold = standardize_answer(gold)
    
    # Handle empty strings
    if not pred or not gold:
        return 0.0
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(pred, gold)[0]
        return round(scores['rouge-l']['f'], 4)  # 取4位小数
    except:
        return 0.0


import time
import multiprocessing as mp
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Optional

class Timer:
    """计时器类，用于跟踪任务执行时间"""
    def __init__(self):
        self.manager = mp.Manager()
        self.times = self.manager.dict()
    
    @contextmanager
    def timer(self, name: str):
        """上下文管理器形式的计时器"""
        try:
            start_time = time.time()
            yield
        finally:
            self.times[name] = time.time() - start_time
    
    def __getitem__(self, key: str) -> float:
        return self.times.get(key, 0.0)
    
    def summary(self) -> str:
        """返回格式化的时间统计信息"""
        return "\n".join(f"{task}: {duration:.2f}秒" 
                        for task, duration in self.times.items())

def timed(timer: Timer, name: Optional[str] = None):
    """函数装饰器，用于计时"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_name = name or func.__name__
            with timer.timer(task_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    print(RL_score("The answer is 10.", "The answer is 10."))
    print(RL_score("The answer is 10.", "The answer is 11."))
    print(RL_score("The answer is 10.", "The answer is 100."))
    print(RL_score("The answer is 10.", "The answer is 100."))
    print(RL_score("The answer is 10.", "The answer is 100."))

    print(EM_score("The answer is 10.", "The answer is 10."))
    print(EM_score("The answer is 10.", "The answer is 11."))
    print(EM_score("The answer is 10.", "The answer is 100."))
    print(EM_score("The answer is 10.", "The answer is 100."))
    print(EM_score("The answer is 10.", "The answer is 100."))