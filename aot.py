# Atom of Thoughts (AoT) Implementation
# Performs question decomposition and multi-step reasoning

import asyncio
from functools import wraps
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aot")

# LLM interface with retries and error handling
async def gen(prompt: str, response_format: str = "text") -> Union[str, Dict[str, Any]]:
    """
    Generate response from LLM.
    
    Args:
        prompt: Input prompt for the model
        response_format: Format type ("text" or "json_object")
    
    Returns:
        Response from model in specified format
    """
    # Replace this with actual LLM integration
    logger.debug(f"Generating with prompt: {prompt[:50]}...")
    
    # Mock implementation for testing
    if response_format == "text":
        return "Mock text response"
    elif response_format == "json_object":
        if "decompose" in prompt.lower():
            return {
                "sub-questions": [
                    {"description": "First sub-question", "depend": []},
                    {"description": "Second sub-question", "depend": [0]},
                ]
            }
        return {"answer": "Mock answer"}
    
    raise ValueError(f"Unsupported response format: {response_format}")

class Prompter:
    """Handles prompt construction and response validation for different AoT operations."""
    
    def __init__(self, module_type: str):
        self.module_type = module_type
        self.templates = {
            "math": {
                "direct": "Solve this math problem step by step:\n{question}",
                "multistep": "Break down this problem into smaller sub-problems:\n{question}",
                "label": "Identify dependency relationships between these sub-questions:\n{sub_questions}\nOriginal question: {question}",
                "contract": "Combine these partial answers to solve the original problem:\n{sub_result}\nOriginal question: {question}",
                "ensemble": "Choose the best answer from these candidates:\n{results}\nOriginal question: {question}"
            },
            "multi-choice": {
                "direct": "Answer this multiple-choice question:\n{question}",
                "multistep": "Break down this multiple-choice question into smaller parts:\n{question}",
                "label": "Identify dependencies between sub-questions:\n{sub_questions}\nOriginal question: {question}",
                "contract": "Use these partial answers to select the correct option:\n{sub_result}\nOriginal question: {question}",
                "ensemble": "Select the best answer from these candidates:\n{results}\nOriginal question: {question}"
            },
            "multi-hop": {
                "direct": "Answer this multi-hop question using the provided context:\n{question}\nContext: {contexts}",
                "multistep": "Break down this multi-hop question into sub-questions that need answering:\n{question}\nContext: {contexts}",
                "label": "Identify dependencies between sub-questions:\n{sub_questions}\nOriginal question: {question}",
                "contract": "Combine the answers to these sub-questions to answer the original question:\n{sub_result}\nOriginal question: {question}",
                "ensemble": "Select the best answer from these candidates:\n{results}\nOriginal question: {question}"
            }
        }
    
    def get_template(self, func_name: str) -> str:
        """Get prompt template for given function and module type."""
        if self.module_type not in self.templates:
            raise ValueError(f"Unsupported module type: {self.module_type}")
        
        if func_name not in self.templates[self.module_type]:
            raise ValueError(f"No template for {func_name} in module {self.module_type}")
        
        return self.templates[self.module_type][func_name]
    
    def direct(self, question: str, contexts: Optional[str] = None) -> str:
        """Create prompt for direct answer generation."""
        template = self.get_template("direct")
        return template.format(question=question, contexts=contexts or "")
    
    def multistep(self, question: str, contexts: Optional[str] = None) -> str:
        """Create prompt for decomposing questions into sub-steps."""
        template = self.get_template("multistep")
        return template.format(question=question, contexts=contexts or "")
    
    def label(self, question: str, sub_questions: str, answer: Optional[str] = None) -> str:
        """Create prompt for labeling dependencies between sub-questions."""
        template = self.get_template("label")
        return template.format(question=question, sub_questions=sub_questions, answer=answer or "")
    
    def contract(self, question: str, sub_result: Dict[str, Any], 
                independent_subqs: List[Dict[str, Any]], dependent_subqs: List[Dict[str, Any]], 
                contexts: Optional[str] = None) -> str:
        """Create prompt for combining sub-question results."""
        template = self.get_template("contract")
        sub_result_str = str(sub_result)
        return template.format(question=question, sub_result=sub_result_str, contexts=contexts or "")
    
    def ensemble(self, question: str, results: List[Any], contexts: Optional[str] = None) -> str:
        """Create prompt for ensembling multiple results."""
        template = self.get_template("ensemble")
        results_str = "\n".join([str(r) for r in results])
        return template.format(question=question, results=results_str, contexts=contexts or "")
    
    def check(self, func_name: str, result: Any) -> bool:
        """Validate response format based on function type."""
        if not isinstance(result, dict):
            return False
            
        if func_name == "direct":
            return "answer" in result
        elif func_name == "multistep":
            return "sub-questions" in result and isinstance(result["sub-questions"], list)
        elif func_name == "label":
            return "sub-questions" in result and isinstance(result["sub-questions"], list)
        elif func_name == "contract":
            return "answer" in result
        elif func_name == "ensemble":
            return "answer" in result
            
        return False

# Module state
_prompter: Optional[Prompter] = None

def set_module(module_type: str) -> None:
    """
    Set the module type for AoT operations.
    
    Args:
        module_type: Type of module ('math', 'multi-choice', or 'multi-hop')
    """
    global _prompter
    if module_type not in ["math", "multi-choice", "multi-hop"]:
        raise ValueError(f"Unsupported module type: {module_type}")
    
    _prompter = Prompter(module_type)
    logger.info(f"Module set to: {module_type}")

def get_prompter() -> Prompter:
    """Get current prompter, raising error if not initialized."""
    if _prompter is None:
        raise RuntimeError("Module not set. Call set_module() first.")
    return _prompter

def retry(func_name: str, max_retries: int = 3):
    """
    Retry decorator for LLM operations.
    
    Args:
        func_name: Name of the function in the prompter
        max_retries: Maximum number of retry attempts (default: 3)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            prompter = get_prompter()
            retries = max_retries
            last_result = None
            
            while retries >= 0:
                try:
                    prompt = getattr(prompter, func_name)(*args, **kwargs)
                    response_format = "json_object"
                    response = await gen(prompt, response_format=response_format)
                    
                    # Convert string response to dict if needed
                    if isinstance(response, str):
                        import json
                        try:
                            result = json.loads(response)
                        except:
                            result = {"response": response, "answer": response}
                    else:
                        result = response
                    
                    # Validate the response format
                    if prompter.check(func_name, result):
                        logger.debug(f"{func_name} succeeded after {max_retries - retries} retries")
                        return result
                    
                    logger.warning(f"{func_name} response validation failed, retrying ({retries} left)")
                    last_result = result
                except Exception as e:
                    logger.error(f"Error in {func_name}: {str(e)}, retries left: {retries}")
                    last_result = {"error": str(e)}
                
                retries -= 1
            
            # If we get here, all retries failed
            logger.error(f"{func_name} failed after {max_retries} retries")
            return last_result or {"error": f"All {max_retries} retries failed"}
        return wrapper
    return decorator

@retry("direct")
async def direct(question: str, contexts: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate direct answer to a question.
    
    Args:
        question: Question to answer
        contexts: Optional context information
    
    Returns:
        Dictionary containing the answer
    """
    # Implementation handled by retry decorator
    pass

@retry("multistep")
async def multistep(question: str, contexts: Optional[str] = None) -> Dict[str, Any]:
    """
    Decompose a question into sub-questions.
    
    Args:
        question: Question to decompose
        contexts: Optional context information
    
    Returns:
        Dictionary containing sub-questions and dependencies
    """
    # Implementation handled by retry decorator
    pass

@retry("label")
async def label(question: str, sub_questions: str, answer: Optional[str] = None) -> Dict[str, Any]:
    """
    Label dependencies between sub-questions.
    
    Args:
        question: Original question
        sub_questions: String representation of sub-questions
        answer: Optional answer to include
    
    Returns:
        Dictionary containing labeled sub-questions
    """
    # Implementation handled by retry decorator
    pass

@retry("contract")
async def contract(
    question: str,
    sub_result: Dict[str, Any],
    independent_subqs: List[Dict[str, Any]],
    dependent_subqs: List[Dict[str, Any]],
    contexts: Optional[str] = None
) -> Dict[str, Any]:
    """
    Contract sub-question results into final answer.
    
    Args:
        question: Original question
        sub_result: Results from sub-questions
        independent_subqs: List of independent sub-questions
        dependent_subqs: List of dependent sub-questions
        contexts: Optional context information
    
    Returns:
        Dictionary containing final answer
    """
    # Implementation handled by retry decorator
    pass

@retry("ensemble")
async def ensemble(question: str, results: List[Any], contexts: Optional[str] = None) -> Dict[str, Any]:
    """
    Ensemble multiple results into a final answer.
    
    Args:
        question: Original question
        results: List of results to ensemble
        contexts: Optional context information
    
    Returns:
        Dictionary containing ensembled answer
    """
    # Implementation handled by retry decorator
    pass

async def decompose(question: str, **kwargs) -> Dict[str, Any]:
    """
    Decompose a question into sub-questions.
    
    Args:
        question: Question to decompose
        **kwargs: Additional arguments
    
    Returns:
        Result of multistep decomposition
    """
    result = await multistep(question, kwargs.get("contexts"))
    logger.info(f"Decomposed into {len(result.get('sub-questions', []))} sub-questions")
    return result

async def process_subquestions(
    question: str,
    decompose_result: Dict[str, Any],
    contexts: Optional[str] = None,
    depth: int = 0,
    max_depth: int = 2
) -> Dict[str, Dict[str, Any]]:
    """
    Process all sub-questions and collect their answers.
    
    Args:
        question: Original question
        decompose_result: Result of decomposition
        contexts: Optional context information
        depth: Current recursion depth
        max_depth: Maximum recursion depth
    
    Returns:
        Dictionary mapping sub-question indexes to answers
    """
    if depth > max_depth:
        logger.warning(f"Max recursion depth {max_depth} reached, stopping")
        return {}
    
    sub_questions = decompose_result.get("sub-questions", [])
    if not sub_questions:
        return {}
    
    # Process independent questions first
    independent_subqs = [sub_q for sub_q in sub_questions if not sub_q.get("depend", [])]
    dependent_subqs = [sub_q for sub_q in sub_questions if sub_q.get("depend", [])]
    
    sub_results = {}
    
    # Handle independent questions concurrently
    async def process_independent_question(idx, sub_q):
        try:
            result, _ = await atom(sub_q["description"], contexts=contexts, depth=depth+1)
            return idx, result
        except Exception as e:
            logger.error(f"Error processing sub-question {idx}: {str(e)}")
            return idx, {"error": str(e)}
    
    # Process independent questions concurrently
    independent_tasks = [
        process_independent_question(idx, sub_questions[idx]) 
        for idx in range(len(sub_questions)) 
        if idx < len(sub_questions) and not sub_questions[idx].get("depend", [])
    ]
    
    if independent_tasks:
        independent_results = await asyncio.gather(*independent_tasks)
        for idx, result in independent_results:
            sub_results[idx] = result
    
    # Process dependent questions in order of dependencies
    for idx, sub_q in enumerate(sub_questions):
        if idx in sub_results:  # Already processed as independent
            continue
            
        dependencies = sub_q.get("depend", [])
        # Check if all dependencies are satisfied
        if all(dep in sub_results for dep in dependencies):
            # Collect dependency results
            dep_context = "\n".join([
                f"Sub-question {dep}: {sub_questions[dep]['description']}\n"
                f"Answer: {sub_results[dep].get('answer', 'Unknown')}"
                for dep in dependencies
            ])
            
            # Add dependency context to the question
            enhanced_question = f"{sub_q['description']}\n\nContext from previous questions:\n{dep_context}"
            
            # Process with dependencies
            try:
                result, _ = await atom(enhanced_question, contexts=contexts, depth=depth+1)
                sub_results[idx] = result
            except Exception as e:
                logger.error(f"Error processing dependent sub-question {idx}: {str(e)}")
                sub_results[idx] = {"error": str(e)}
    
    return sub_results

async def merging(
    question: str,
    decompose_result: Dict[str, Any],
    sub_results: Dict[int, Dict[str, Any]],
    contexts: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Merge sub-question results into a final answer.
    
    Args:
        question: Original question
        decompose_result: Result of decomposition
        sub_results: Results from all sub-questions
        contexts: Optional context information
    
    Returns:
        Tuple of (thought type, processed question, contract result)
    """
    sub_questions = decompose_result.get("sub-questions", [])
    
    # Prepare sub-result representation for contraction
    sub_result_repr = {
        "original_question": question,
        "sub_questions": []
    }
    
    for idx, sub_q in enumerate(sub_questions):
        if idx in sub_results:
            sub_result_repr["sub_questions"].append({
                "description": sub_q["description"],
                "answer": sub_results[idx].get("answer", "Unknown"),
                "depend": sub_q.get("depend", [])
            })
    
    # Identify independent and dependent sub-questions
    independent_subqs = [sub_q for sub_q in sub_questions if not sub_q.get("depend", [])]
    dependent_subqs = [sub_q for sub_q in sub_questions if sub_q.get("depend", [])]
    
    # Contract the results
    contract_result = await contract(
        question, 
        sub_result_repr, 
        independent_subqs, 
        dependent_subqs, 
        contexts
    )
    
    return "thought", question, contract_result

async def atom(
    question: str,
    contexts: Optional[str] = None,
    direct_result: Optional[Dict[str, Any]] = None,
    decompose_result: Optional[Dict[str, Any]] = None,
    depth: int = 0,
    log: Optional[Dict[int, Any]] = None
) -> Tuple[Dict[str, Any], Dict[int, Any]]:
    """
    Core AoT operation: decompose, process sub-questions, and merge results.
    
    Args:
        question: Question to process
        contexts: Optional context information
        direct_result: Optional pre-computed direct result
        decompose_result: Optional pre-computed decomposition
        depth: Current recursion depth
        log: Log dictionary for recording operations
    
    Returns:
        Tuple of (final result, operation log)
    """
    # Initialize log if not provided
    log = log or {}
    index = len(log)
    log[index] = {
        "question": question,
        "depth": depth,
        "type": "atom"
    }
    
    try:
        # Get direct result if not provided
        direct_result = direct_result or await direct(question, contexts)
        log[index]["direct_result"] = direct_result
        
        # Short-circuit for simple questions or maximum depth
        if depth > 2:
            logger.info(f"Max depth reached, using direct result for: {question[:50]}...")
            return direct_result, log
        
        # Decompose if not already done
        decompose_result = decompose_result or await decompose(question, contexts=contexts)
        log[index]["decompose_result"] = decompose_result
        
        # If no sub-questions or decomposition failed, return direct result
        sub_questions = decompose_result.get("sub-questions", [])
        if not sub_questions:
            logger.info(f"No sub-questions, using direct result for: {question[:50]}...")
            return direct_result, log
        
        # Process all sub-questions
        sub_results = await process_subquestions(question, decompose_result, contexts, depth)
        log[index]["sub_results"] = sub_results
        
        # Merge results if we have sub-question answers
        if sub_results:
            _, _, contraction_result = await merging(question, decompose_result, sub_results, contexts)
            log[index]["contraction_result"] = contraction_result
            
            # If we have an answer from contraction, use it
            if "answer" in contraction_result:
                return contraction_result, log
        
        # Fallback to direct result if contraction failed or no sub-results
        logger.info(f"Using direct result as fallback for: {question[:50]}...")
        return direct_result, log
        
    except Exception as e:
        logger.error(f"Error in atom: {str(e)}")
        log[index]["error"] = str(e)
        
        # Return direct result or error
        if direct_result:
            return direct_result, log
        else:
            return {"error": str(e), "answer": "Error occurred during processing"}, log

# Main test function
if __name__ == "__main__":
    async def test():
        set_module("math")
        question = "What is 2 + 2?"
        print(f"Testing AoT with question: {question}")
        
        # Try with direct only
        result, log = await atom(question)
        print(f"AoT Result: {result.get('answer', 'Unknown')}")
        
        # Try with decomposition
        decompose_result = await decompose(question)
        print(f"Decomposition: {decompose_result}")
        
        # Full atom with pre-computed decomposition
        result, log = await atom(question, decompose_result=decompose_result)
        print(f"AoT Result with decomposition: {result.get('answer', 'Unknown')}")
    
    asyncio.run(test())