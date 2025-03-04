# AOT-GRPO Integration
# Combines Algorithm of Thoughts with Generative Representational Prompt Optimization

import asyncio
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import local modules
from dataset import load_data
from aot import set_module, atom
from grpo import get_device, optimize_and_generate, batch_optimize_prompts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aot_grpo")

async def process_question_with_aot_grpo(
    question: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    module_type: str = "math",
    max_depth: int = 2,
    optimization_steps: int = 10,
    max_new_tokens: int = 100,
    num_learnable_tokens: int = 5,
    process_dependent: bool = True,
    contexts: Optional[str] = None,
    low_memory_mode: bool = False
) -> Dict[str, Any]:
    """
    Process a question using AoT decomposition followed by GRPO optimization.
    
    Args:
        question: The input question
        model: The pre-trained language model
        tokenizer: The corresponding tokenizer
        device: Device to run on
        module_type: Type of reasoning module ('math', 'multi-choice', or 'multi-hop')
        max_depth: Maximum recursion depth for AoT
        optimization_steps: Number of optimization steps for GRPO
        max_new_tokens: Maximum new tokens to generate
        num_learnable_tokens: Number of trainable tokens for GRPO
        process_dependent: Whether to process dependent sub-questions
        contexts: Optional context information for the question
        
    Returns:
        Dictionary containing the final answer and processing details
    """
    logger.info(f"Processing question with AOT+GRPO: {question[:50]}...")
    
    # Initialize AoT with the appropriate module type
    set_module(module_type)
    
    # Step 1: Use AoT to decompose the question into sub-questions
    decompose_result, aot_log = await atom(question, contexts=contexts)
    logger.info(f"AoT decomposition completed")
    
    # Extract the initial answer from AoT (fallback)
    initial_answer = decompose_result.get("answer", "No answer found")
    
    # Check if we have sub-questions to process
    if "sub-questions" not in decompose_result and "answer" in decompose_result:
        # No sub-questions, just optimize the direct answer
        logger.info("No sub-questions found, optimizing direct answer")
        
        if low_memory_mode:
            # In low memory mode, skip optimization and just generate
            logger.info("Low memory mode: using direct generation without optimization")
            inputs = tokenizer(f"{question}\nAnswer: ", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                optimized_answer = tokenizer.decode(
                    outputs[0, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
        else:
            # Use full GRPO optimization
            optimized_answer = optimize_and_generate(
                model,
                tokenizer,
                f"{question}\nAnswer: ",
                device,
                num_steps=optimization_steps,
                max_new_tokens=max_new_tokens,
                num_learnable_tokens=num_learnable_tokens
            )
        
        return {
            "question": question,
            "aot_answer": initial_answer,
            "optimized_answer": optimized_answer,
            "method": "direct",
            "sub_questions": [],
            "log": aot_log
        }
    
    # Step 2: Process each sub-question with GRPO
    sub_questions = decompose_result.get("sub-questions", [])
    if not sub_questions:
        # Create a default sub-question if none were generated
        sub_questions = [{"description": question, "depend": []}]
    
    logger.info(f"Processing {len(sub_questions)} sub-questions")
    
    # Separate independent and dependent sub-questions
    independent_subqs = [sub_q for sub_q in sub_questions if not sub_q.get("depend", [])]
    dependent_subqs = [sub_q for sub_q in sub_questions if sub_q.get("depend", [])]
    
    # Process independent sub-questions with GRPO
    independent_answers = {}
    independent_prompts = [
        f"{sub_q['description']}\nAnswer: " for sub_q in independent_subqs
    ]
    
    # Optimize and generate answers for independent questions
    if independent_prompts:
        if low_memory_mode:
            # In low memory mode, use direct generation without optimization
            logger.info("Low memory mode: using direct generation for sub-questions")
            opt_answers = []
            for prompt in independent_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    answer = tokenizer.decode(
                        outputs[0, inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    opt_answers.append(answer)
        else:
            # Use full GRPO optimization
            opt_answers = batch_optimize_prompts(
                model,
                tokenizer,
                independent_prompts,
                device,
                num_steps=optimization_steps,
                max_new_tokens=max_new_tokens,
                num_learnable_tokens=num_learnable_tokens
            )
        
        # Store the results
        for i, answer in enumerate(opt_answers):
            sub_q = independent_subqs[i]
            independent_answers[sub_q["description"]] = answer
    
    # Process dependent sub-questions if enabled
    dependent_answers = {}
    if process_dependent and dependent_subqs:
        logger.info(f"Processing {len(dependent_subqs)} dependent sub-questions")
        
        for sub_q in dependent_subqs:
            dependencies = sub_q.get("depend", [])
            
            # Collect context from dependencies
            context_parts = []
            for dep_idx in dependencies:
                if dep_idx < len(sub_questions):
                    dep_q = sub_questions[dep_idx]
                    dep_desc = dep_q["description"]
                    dep_answer = independent_answers.get(dep_desc, "Unknown")
                    context_parts.append(f"Sub-question: {dep_desc}")
                    context_parts.append(f"Answer: {dep_answer}")
            
            # Create enhanced prompt with dependency context
            dep_context = "\n".join(context_parts)
            enhanced_prompt = f"{sub_q['description']}\n\nContext from previous questions:\n{dep_context}\n\nAnswer: "
            
            # Generate answer (with or without optimization)
            if low_memory_mode:
                # Direct generation for low memory
                inputs = tokenizer(enhanced_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    opt_answer = tokenizer.decode(
                        outputs[0, inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
            else:
                # Full optimization
                opt_answer = optimize_and_generate(
                    model,
                    tokenizer,
                    enhanced_prompt,
                    device,
                    num_steps=optimization_steps,
                    max_new_tokens=max_new_tokens,
                    num_learnable_tokens=num_learnable_tokens
                )
            
            dependent_answers[sub_q["description"]] = opt_answer
    
    # Combine all answers
    all_answers = {**independent_answers, **dependent_answers}
    
    # Step 3: Merge the sub-question answers to get the final answer
    if all_answers:
        # Create a prompt for the final answer
        merge_context = []
        for sub_q in sub_questions:
            desc = sub_q["description"]
            if desc in all_answers:
                merge_context.append(f"Sub-question: {desc}")
                merge_context.append(f"Answer: {all_answers[desc]}")
        
        merge_prompt = f"Original question: {question}\n\nBreakdown:\n{chr(10).join(merge_context)}\n\nFinal answer: "
        
        if low_memory_mode:
            # Direct generation for low memory
            inputs = tokenizer(merge_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens * 2,  # Allow more tokens for final answer
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                final_answer = tokenizer.decode(
                    outputs[0, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
        else:
            # Full optimization
            final_answer = optimize_and_generate(
                model,
                tokenizer,
                merge_prompt,
                device,
                num_steps=optimization_steps,
                max_new_tokens=max_new_tokens * 2,  # Allow more tokens for final answer
                num_learnable_tokens=num_learnable_tokens
            )
    else:
        # If no sub-question answers, use the initial answer
        final_answer = initial_answer
    
    # Prepare result
    result = {
        "question": question,
        "aot_answer": initial_answer,
        "optimized_answer": final_answer,
        "method": "decompose" if sub_questions else "direct",
        "sub_questions": [
            {
                "description": sub_q["description"],
                "depend": sub_q.get("depend", []),
                "answer": all_answers.get(sub_q["description"], "Not processed")
            }
            for sub_q in sub_questions
        ],
        "log": aot_log
    }
    
    return result

async def batch_process_questions(
    questions: List[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    module_type: str = "math",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process multiple questions with AOT+GRPO.
    
    Args:
        questions: List of questions to process
        model: The pre-trained model
        tokenizer: The corresponding tokenizer
        device: Device to run on
        module_type: Type of reasoning module
        **kwargs: Additional arguments for process_question_with_aot_grpo
        
    Returns:
        List of results for each question
    """
    results = []
    
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}")
        result = await process_question_with_aot_grpo(
            question, model, tokenizer, device, module_type, **kwargs
        )
        results.append(result)
    
    return results

async def process_dataset(
    dataset_name: str,
    model_name: str,
    split: str = "test",
    num_samples: int = 5,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process examples from a dataset with AOT+GRPO.
    
    Args:
        dataset_name: Name of dataset to load
        model_name: Name of model to use
        split: Dataset split to use
        num_samples: Number of examples to process
        **kwargs: Additional arguments for batch_process_questions
        
    Returns:
        List of results for each example
    """
    # Load the dataset
    data = load_data(dataset_name, split)
    if num_samples > 0:
        data = data[:num_samples]
    
    questions = [item["question"] for item in data]
    ground_truth = [item["answer"] for item in data]
    module_type = data[0].get("module_type", "math")
    
    # Load the model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    
    # Process the questions
    results = await batch_process_questions(
        questions, model, tokenizer, device, module_type, **kwargs
    )
    
    # Add ground truth answers
    for i, result in enumerate(results):
        result["ground_truth"] = ground_truth[i]
    
    return results

if __name__ == "__main__":
    async def test_with_local_model():
        """Test the AOT-GRPO pipeline with a locally downloaded model."""
        from pathlib import Path
        import argparse
        
        # Define and parse command line arguments
        parser = argparse.ArgumentParser(description="Test AOT-GRPO with a local model")
        parser.add_argument("--model_path", type=str, default=None, 
                            help="Path to local model directory")
        parser.add_argument("--dataset", type=str, default="gsm8k", 
                            choices=["gsm8k", "math", "bbh", "mmlu", "hotpotqa", "longbench"],
                            help="Dataset to use for testing")
        parser.add_argument("--samples", type=int, default=1, 
                            help="Number of samples to process")
        parser.add_argument("--opt_steps", type=int, default=5, 
                            help="Number of optimization steps")
        parser.add_argument("--no_dependent", action="store_true", 
                            help="Skip processing dependent questions")
        parser.add_argument("--low_memory", action="store_true",
                            help="Enable low memory mode for large models")
        args = parser.parse_args()
        
        # Default path for Phi Mini on MacOS (can be overridden via command line)
        default_model_path = str(Path.home() / "Downloads" / "phi-mini")
        model_path = args.model_path or default_model_path
        
        print(f"Loading local model from: {model_path}")
        print(f"Testing with {args.samples} example(s) from {args.dataset}")
        
        try:
            # Load the model and tokenizer from local path
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Configure model loading options based on memory constraints
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            if args.low_memory:
                print("Loading model in low memory mode (4-bit quantization)...")
                try:
                    from transformers import BitsAndBytesConfig
                    
                    # 4-bit quantization for memory efficiency
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"  # Automatically decide best device placement
                except ImportError:
                    print("Warning: bitsandbytes not available for quantization")
                    # Fall back to regular 16-bit if quantization not available
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["low_cpu_mem_usage"] = True
            else:
                # Standard loading
                model_kwargs["torch_dtype"] = torch.float16
            
            # Load the model with the configured options
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            # Determine device - use CPU if forced by environment variable
            import os
            if os.environ.get("PYTORCH_MPS_ENABLE", "1") == "0":
                print("Forcing CPU usage as requested...")
                device = torch.device("cpu")
            else:
                device = get_device()
                
            # Move to device if not already placed there by device_map="auto"
            if "device_map" not in model_kwargs:
                model.to(device)
            
            # Get example questions
            data = load_data(args.dataset, "test")
            if args.samples > 0:
                data = data[:args.samples]
            
            questions = [item["question"] for item in data]
            module_type = data[0].get("module_type", "math")
            
            # Process each question
            results = []
            for i, question in enumerate(questions):
                print(f"\nProcessing question {i+1}/{len(questions)}...")
                
                result = await process_question_with_aot_grpo(
                    question,
                    model,
                    tokenizer,
                    device,
                    module_type=module_type,
                    optimization_steps=args.opt_steps,
                    max_new_tokens=50,
                    process_dependent=not args.no_dependent,
                    low_memory_mode=args.low_memory
                )
                
                results.append({**result, "ground_truth": data[i]["answer"]})
            
            # Display results
            for i, result in enumerate(results):
                print(f"\n{'='*50}")
                print(f"Example {i+1}:")
                print(f"Question: {result['question']}")
                print(f"\nAoT Answer: {result['aot_answer']}")
                print(f"\nOptimized Answer: {result['optimized_answer']}")
                print(f"\nGround Truth: {result['ground_truth']}")
                
                # Show sub-questions
                if result["sub_questions"]:
                    print("\nSub-questions:")
                    for j, sub_q in enumerate(result["sub_questions"]):
                        print(f"  {j+1}. {sub_q['description']}")
                        print(f"     Answer: {sub_q['answer']}")
                print(f"{'='*50}")
            
        except Exception as e:
            import traceback
            print(f"Error during test: {str(e)}")
            traceback.print_exc()
    
    async def main():
        # Configuration for online model
        dataset_name = "gsm8k"
        model_name = "gpt2"  # Use a small model for testing
        num_samples = 1
        
        print(f"Processing {num_samples} example(s) from {dataset_name} with {model_name}")
        
        # Process dataset
        results = await process_dataset(
            dataset_name,
            model_name,
            num_samples=num_samples,
            optimization_steps=5,  # Small number of steps for testing
            max_new_tokens=30,
            process_dependent=True
        )
        
        # Display results
        for i, result in enumerate(results):
            print(f"\nExample {i+1}:")
            print(f"Question: {result['question'][:100]}...")
            print(f"AoT Answer: {result['aot_answer']}")
            print(f"Optimized Answer: {result['optimized_answer']}")
            print(f"Ground Truth: {result['ground_truth'][:100]}...")
            
            # Show sub-questions
            if result["sub_questions"]:
                print("\nSub-questions:")
                for j, sub_q in enumerate(result["sub_questions"]):
                    print(f"  {j+1}. {sub_q['description'][:50]}...")
                    print(f"     Answer: {sub_q['answer'][:50]}...")
    
    # Run the local model test
    print("Running AOT-GRPO test with local model...")
    asyncio.run(test_with_local_model())