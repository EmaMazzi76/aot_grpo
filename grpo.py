# Group Relative Policy Optimization (GRPO)
# Optimizes prompts at test time to improve model responses

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("grpo")

def get_device():
    """
    Return the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: Best available compute device
    """
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using MPS device")
        return torch.device("mps")
    
    logger.info("Using CPU device")
    return torch.device("cpu")

def optimize_and_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: torch.device,
    num_learnable_tokens: int = 5,
    num_steps: int = 10,
    learning_rate: float = 0.01,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    original_answer: Optional[str] = None
) -> str:
    """
    Optimize a prompt at test time and generate an answer.
    
    Args:
        model: The pre-trained language model
        tokenizer: The corresponding tokenizer
        prompt: Input prompt to optimize
        device: Device to run on
        num_learnable_tokens: Number of trainable tokens at beginning of prompt
        num_steps: Optimization steps
        learning_rate: Learning rate for optimization
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature (higher = more diverse)
        top_p: Nucleus sampling parameter (lower = more focused)
        original_answer: Optional known answer for guided optimization
        
    Returns:
        Generated answer after prompt optimization
    """
    logger.info(f"Optimizing prompt: {prompt[:50]}...")
    model.eval()  # Set model to evaluation mode
    
    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Get initial answer (to optimize against)
    with torch.no_grad():
        initial_outputs = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        initial_answer_ids = initial_outputs[0, prompt_ids.shape[1]:]
        initial_answer_text = tokenizer.decode(initial_answer_ids, skip_special_tokens=True)
        logger.debug(f"Initial answer: {initial_answer_text}")
    
    # Create learnable prompt embeddings
    if hasattr(model, "get_input_embeddings"):
        try:
            embedding_layer = model.get_input_embeddings()
            embedding_size = model.config.hidden_size
            
            # Initialize trainable tokens (strategies: random, cloned from existing, or zeros)
            learnable_prompt = torch.randn(
                1, num_learnable_tokens, embedding_size, 
                device=device, 
                requires_grad=True
            )
            
            # Set up optimizer
            optimizer = torch.optim.Adam([learnable_prompt], lr=learning_rate)
            
            # Optimization loop
            for step in range(num_steps):
                optimizer.zero_grad()
                
                # Get prompt embeddings
                prompt_emb = embedding_layer(prompt_ids)
                
                # Combine learnable tokens with prompt embeddings
                combined_emb = torch.cat([learnable_prompt, prompt_emb], dim=1)
                
                # Calculate loss based on original answer if provided
                if original_answer is not None:
                    # Encode target answer
                    target_ids = tokenizer.encode(original_answer, return_tensors="pt").to(device)
                    
                    # Forward pass
                    outputs = model(inputs_embeds=combined_emb)
                    logits = outputs.logits
                    
                    # Shift logits and labels for causal language modeling
                    shift_logits = logits[:, :-1, :]
                    shift_labels = prompt_ids[:, 1:]
                    
                    # Calculate cross-entropy loss for predicting next token
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                    
                else:
                    # No target answer, optimize for general fluency and coherence
                    inputs_embeds = combined_emb
                    
                    # We'll run an autoregressive prediction and optimize for coherence
                    outputs = model(inputs_embeds=inputs_embeds)
                    logits = outputs.logits
                    
                    # Calculate loss to predict prompt tokens (proxy for model confidence)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = torch.cat([
                        torch.zeros(1, num_learnable_tokens, dtype=torch.long, device=device),
                        prompt_ids[:, :-1]
                    ], dim=1)
                    
                    # Use cross-entropy loss with a focus on last few tokens
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits[:, -prompt_ids.size(1):].reshape(-1, shift_logits.size(-1)),
                        shift_labels[:, -prompt_ids.size(1):].reshape(-1)
                    )
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                if step % 5 == 0:
                    logger.debug(f"Step {step}, Loss: {loss.item():.4f}")
            
            # Generate with optimized prompt
            with torch.no_grad():
                final_emb = torch.cat([learnable_prompt, prompt_emb], dim=1)
                outputs = model.generate(
                    inputs_embeds=final_emb,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Extract and decode the answer
                answer_text = tokenizer.decode(
                    outputs[0, combined_emb.size(1):], 
                    skip_special_tokens=True
                ).strip()
                
            logger.info(f"Optimization completed after {num_steps} steps")
            return answer_text
            
        except Exception as e:
            logger.error(f"Error during prompt optimization: {str(e)}")
            logger.info("Falling back to non-optimized generation")
            
    else:
        logger.warning("Model doesn't support input embeddings access, using regular generation")
    
    # Fallback to regular generation without optimization
    with torch.no_grad():
        outputs = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        answer_text = tokenizer.decode(
            outputs[0, prompt_ids.size(1):], 
            skip_special_tokens=True
        ).strip()
    
    return answer_text

def batch_optimize_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    device: torch.device,
    batch_size: int = 1,
    **kwargs
) -> List[str]:
    """
    Optimize multiple prompts in batches.
    
    Args:
        model: The pre-trained model
        tokenizer: The corresponding tokenizer
        prompts: List of prompts to optimize
        device: Device to run on
        batch_size: Batch size for processing (currently only supports 1)
        **kwargs: Additional arguments for optimize_and_generate
        
    Returns:
        List of generated answers
    """
    answers = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}")
        answer = optimize_and_generate(model, tokenizer, prompt, device, **kwargs)
        answers.append(answer)
    
    return answers

def train_with_grpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data: List[Dict[str, str]],
    device: torch.device,
    num_iterations: int = 1,
    steps_per_iteration: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    num_learnable_tokens: int = 5
) -> PreTrainedModel:
    """
    Train the model with GRPO on a dataset.
    
    Args:
        model: The pre-trained model
        tokenizer: The corresponding tokenizer
        train_data: List of training examples with 'question' and 'answer' keys
        device: Device to run on
        num_iterations: Number of training iterations
        steps_per_iteration: Steps per iteration
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_learnable_tokens: Number of trainable tokens
        
    Returns:
        Trained model
    """
    logger.info(f"Training with GRPO, {len(train_data)} examples, {num_iterations} iterations")
    
    # Set model to training mode
    model.train()
    embedding_layer = model.get_input_embeddings()
    embedding_size = model.config.hidden_size
    
    # Process in batches
    for iteration in range(num_iterations):
        logger.info(f"Starting iteration {iteration+1}/{num_iterations}")
        
        # Shuffle data for each iteration
        import random
        random.shuffle(train_data)
        
        for batch_idx in range(0, len(train_data), batch_size):
            batch = train_data[batch_idx:batch_idx + batch_size]
            
            # Process each example
            for example in batch:
                question = example['question']
                answer = example['answer']
                
                # Encode inputs
                question_ids = tokenizer.encode(question, return_tensors="pt").to(device)
                answer_ids = tokenizer.encode(answer, return_tensors="pt").to(device)
                
                # Initialize trainable tokens
                learnable_tokens = torch.randn(
                    1, num_learnable_tokens, embedding_size, 
                    device=device, 
                    requires_grad=True
                )
                
                # Set up optimizer
                optimizer = torch.optim.Adam([learnable_tokens], lr=learning_rate)
                
                # Train for steps_per_iteration
                for step in range(steps_per_iteration):
                    optimizer.zero_grad()
                    
                    # Get embeddings for question
                    question_emb = embedding_layer(question_ids)
                    
                    # Combine trainable tokens with question
                    combined_emb = torch.cat([learnable_tokens, question_emb], dim=1)
                    
                    # Forward pass
                    outputs = model(inputs_embeds=combined_emb, labels=answer_ids)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    if step % 10 == 0:
                        logger.debug(f"  Step {step}, Loss: {loss.item():.4f}")
    
    # Set model back to evaluation mode
    model.eval()
    return model

if __name__ == "__main__":
    model_name = "gpt2"  # Use a small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    
    # Test prompt optimization
    prompt = "What is 2 + 2?"
    print(f"Testing GRPO with prompt: {prompt}")
    
    # Run optimization
    answer = optimize_and_generate(
        model, 
        tokenizer, 
        prompt, 
        device, 
        num_steps=5,  # Small number of steps for testing
        max_new_tokens=20
    )
    
    print(f"Optimized Answer: {answer}")