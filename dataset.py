# Dataset utilities for loading and processing benchmark datasets

from typing import Dict, List, Union, Optional, Any
from datasets import load_dataset

# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {
        "dataset_path": "gsm8k",
        "dataset_name": "main",
        "question_key": "question", 
        "answer_key": "answer", 
        "module_type": "math"
    },
    "math": {
        "dataset_path": "hendrycks/math",  # Actual Math dataset path
        "dataset_name": "algebra",  # Default subset, can be overridden
        "question_key": "problem", 
        "answer_key": "solution", 
        "module_type": "math"
    },
    "bbh": {
        "dataset_path": "lukaemon/bbh",
        "dataset_name": None,
        "question_key": "input", 
        "answer_key": "target", 
        "module_type": "multi-choice"
    },
    "mmlu": {
        "dataset_path": "cais/mmlu",
        "dataset_name": "all",  # Default subset, can be overridden
        "question_key": ["question", "choices"], 
        "answer_key": "answer", 
        "module_type": "multi-choice"
    },
    "hotpotqa": {
        "dataset_path": "hotpot_qa",
        "dataset_name": "fullwiki",
        "question_key": "question", 
        "answer_key": "answer", 
        "module_type": "multi-hop"
    },
    "longbench": {
        "dataset_path": "THUDM/longbench",
        "dataset_name": "qasper",  # Default subset, can be overridden
        "question_key": "input", 
        "answer_key": "answers", 
        "module_type": "multi-hop"
    },
}

def format_question(item: Dict[str, Any], question_key: Union[str, List[str]]) -> str:
    """Format question from dataset item according to configuration.
    
    Args:
        item: Dataset item containing question fields
        question_key: Key or list of keys to extract from item
        
    Returns:
        Formatted question string
    """
    try:
        if isinstance(question_key, list):
            parts = []
            for k in question_key:
                if k == "choices" and "choices" in item:
                    # Format multiple-choice options
                    choices = item["choices"]
                    if isinstance(choices, list):
                        for i, choice in enumerate(choices):
                            parts.append(f"{chr(65+i)}: {choice}")
                else:
                    if k in item:
                        parts.append(f"{k}: {item[k]}")
            return "\n".join(parts)
        
        if question_key in item:
            return str(item[question_key])
        
        # If key not found, try to use the first available text field
        for key in ["question", "input", "text"]:
            if key in item:
                return str(item[key])
                
        # Last resort
        return str(list(item.values())[0])
    except Exception as e:
        # In case of errors, return a simplified representation
        return f"Error formatting question: {str(e)}"

def get_answer(item: Dict[str, Any], answer_key: str) -> str:
    """Extract answer safely from dataset item.
    
    Args:
        item: Dataset item containing answer field
        answer_key: Key to extract from item
        
    Returns:
        Answer string or empty string if not found
    """
    try:
        if answer_key in item:
            answer = item[answer_key]
            # Handle answers that might be lists
            if isinstance(answer, list):
                return answer[0] if answer else ""
            return str(answer)
            
        # If key not found, try to use common answer fields
        for key in ["answer", "target", "output"]:
            if key in item:
                return str(item[key])
                
        # Last resort
        return ""
    except Exception as e:
        return f"Error extracting answer: {str(e)}"

def load_data(dataset_name: str, split: str = "test", subset: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load a dataset by name and split, returning a list of dictionaries with question and answer keys.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k')
        split: Dataset split (e.g., 'test', 'train')
        subset: Optional dataset subset name to override default
    
    Returns:
        List of dicts with 'question', 'answer', and 'module_type' keys
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    dataset_path = config["dataset_path"]
    dataset_name = subset or config["dataset_name"]
    
    try:
        # Load dataset with appropriate parameters
        if dataset_name:
            data = load_dataset(dataset_path, dataset_name)[split]
        else:
            data = load_dataset(dataset_path)[split]
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}': {str(e)}")
    
    # Process each item in the dataset
    result = []
    for item in data:
        try:
            formatted_item = {
                "question": format_question(item, config["question_key"]),
                "answer": get_answer(item, config["answer_key"]),
                "module_type": config["module_type"],
                "raw_item": item  # Keep original for reference if needed
            }
            result.append(formatted_item)
        except Exception as e:
            print(f"Error processing item in {dataset_name}: {str(e)}")
            continue
    
    return result

if __name__ == "__main__":
    # Test dataset loading
    for dataset in ["gsm8k"]:
        try:
            print(f"\nLoading {dataset}...")
            data = load_data(dataset, "test")
            print(f"Loaded {len(data)} items")
            if data:
                sample = data[0]
                print(f"Sample question: {sample['question'][:100]}...")
                print(f"Sample answer: {sample['answer'][:100]}...")
                print(f"Module type: {sample['module_type']}")
        except Exception as e:
            print(f"Error with {dataset}: {str(e)}")