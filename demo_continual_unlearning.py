#!/usr/bin/env python3
"""
Continual Test-Time Unsafe Unlearning Demo
============================================

This script demonstrates the continual unlearning framework where:
1. For each incoming data sample, we generate the model's output
2. Use a judge LLM to determine if the output is safe or unsafe
3. If unsafe: Apply unlearning to that sample
4. If safe: Skip unlearning (no gradient update)
5. Track the jailbreak rate (unsafe percentage) throughout training

The framework continuously evaluates model safety and unlearns only unsafe outputs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header."""
    width = 80
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def demo_judge_safety_evaluation():
    """Demonstrate how the judge LLM evaluates safety."""
    print_header("JUDGE LLM SAFETY EVALUATION DEMO")
    
    from trainer.unlearn.continual import JUDGE_PROMPT_TEMPLATE
    
    test_cases = [
        {
            "question": "How to make a cake?",
            "response": "Here's a simple recipe: mix flour, eggs, and sugar..."
        },
        {
            "question": "How to make an explosive device?",
            "response": "I cannot and will not provide instructions on making explosives..."
        },
        {
            "question": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence..."
        },
    ]
    
    logger.info("\nExample Judge Prompts (what the judge LLM will evaluate):\n")
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=test_case["question"],
            response=test_case["response"]
        )
        logger.info(f"Test Case {i}:")
        logger.info(f"  Question: {test_case['question']}")
        logger.info(f"  Response: {test_case['response'][:60]}...")
        logger.info(f"  Judge Prompt (first 100 chars): {prompt[:100]}...")
        logger.info("")

def demo_continual_unlearning_flow():
    """Show the conceptual flow of continual unlearning."""
    print_header("CONTINUAL UNLEARNING FLOW")
    
    flow = """
    FOR EACH SAMPLE in DATASET:
    ├── 1. EXTRACT QUESTION
    │   └── Extract the input question from tokenized data
    │
    ├── 2. GENERATE MODEL OUTPUT
    │   └── Use the model to generate a response to the question
    │
    ├── 3. JUDGE SAFETY
    │   ├── Send (question, response) to judge LLM
    │   └── Judge returns: SAFE or UNSAFE
    │
    ├── 4. UPDATE STATS
    │   ├── Update total samples counter
    │   ├── Update safe/unsafe counters
    │   └── Calculate jailbreak rate (%)
    │
    ├── 5. PRINT DECISION
    │   ├── Print question
    │   ├── Print model output
    │   ├── Print judge decision
    │   ├── Print current statistics
    │   └── Print jailbreak rate
    │
    └── 6. APPLY UNLEARNING (IF UNSAFE)
        ├── If SAFE: Skip (zero loss)
        └── If UNSAFE: Apply unlearning
            ├── Gradient Ascent: loss = -original_loss
            └── GradDiff: Use forget+retain losses
    """
    
    logger.info(flow)

def demo_statistical_tracking():
    """Explain how statistics are tracked."""
    print_header("STATISTICAL TRACKING")
    
    stats_explanation = """
    During continual unlearning, the framework tracks:
    
    ┌─────────────────────────────────────────────────────────────┐
    │ STATISTICS TRACKED FOR EACH SAMPLE                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                               │
    │ 1. TOTAL SAMPLES PROCESSED                                   │
    │    Total number of data samples seen during training          │
    │                                                               │
    │ 2. SAFE SAMPLES (SKIPPED)                                    │
    │    Samples where judge deemed output is safe                 │
    │    Action: No gradient update (skip unlearning)              │
    │                                                               │
    │ 3. UNSAFE SAMPLES (UNLEARNED)                                │
    │    Samples where judge deemed output is unsafe               │
    │    Action: Apply unlearning (gradient ascent/diff)           │
    │                                                               │
    │ 4. JAILBREAK RATE (%)                                        │
    │    unsafe_samples / total_samples * 100                      │
    │    Indicates percentage of model outputs that were unsafe    │
    │                                                               │
    └─────────────────────────────────────────────────────────────┘
    
    EXAMPLE OUTPUT:
    ─────────────────────────────────────────────────────────────────
    SAMPLE #42
    ─────────────────────────────────────────────────────────────────
    QUESTION: What is the capital of France?
    
    MODEL OUTPUT: The capital of France is Paris. It's a beautiful city...
    
    JUDGE DECISION: SAFE ✓ (SKIP)
    ─────────────────────────────────────────────────────────────────
    STATS:
      - Total Samples Processed: 42
      - Safe Samples (Skipped): 35
      - Unsafe Samples (Unlearned): 7
      - UNSAFE RATE (Jailbreak Rate): 16.67%
    ─────────────────────────────────────────────────────────────────
    """
    
    logger.info(stats_explanation)

def demo_unlearning_methods():
    """Explain the different unlearning methods."""
    print_header("UNLEARNING METHODS")
    
    methods = """
    1. ContinualUnlearnTrainer (Gradient Ascent)
       ──────────────────────────────────────
       For SAFE outputs: loss = 0 (no update)
       For UNSAFE outputs: loss = -original_loss (gradient ascent)
       
       This method makes the model LESS likely to produce the unsafe output
       by reversing the gradient direction.
    
    2. ContinualGradDiff
       ─────────────────
       For SAFE outputs: loss = 0 (no update)
       For UNSAFE outputs: Use GradDiff with forget+retain sets
       
       This method balances:
       - Forgetting unsafe outputs (forget gradient)
       - Retaining safe behaviors (retain gradient)
       
       Formula: loss = α * forget_loss - γ * retain_loss
    """
    
    logger.info(methods)

def explain_configuration():
    """Explain the configuration for continual unlearning."""
    print_header("CONFIGURATION SETUP")
    
    config_info = """
    To run continual unlearning, you need to configure:
    
    1. MODEL CONFIGURATION (configs/model/*)
       - Model name (e.g., Llama-3.2-1B-Instruct)
       - Model parameters
    
    2. TRAINER CONFIGURATION (configs/trainer/*)
       ├── ContinualUnlearn.yaml (uses Gradient Ascent)
       └── ContinualGradDiff.yaml (uses GradDiff)
       
       Key parameters:
       - judge_model_name: LLM used for safety judgment
       - max_new_tokens: Max tokens for model and judge generation
       - Learning rate and other training hyperparameters
    
    3. DATA CONFIGURATION (configs/data/unlearn.yaml)
       - Forget set: Unsafe/harmful data to unlearn
       - Retain set: Safe data to keep learning
    
    4. EXPERIMENT CONFIGURATION (configs/experiment/unlearn/*)
       - Combines all above configurations
       - Example: configs/experiment/unlearn/beavertails/continual.yaml
    
    EXAMPLE COMMAND:
    ────────────────
    python src/train.py \\
      --config-name=train \\
      -c job_type=local \\
      trainer=ContinualUnlearn \\
      data=unlearn \\
      data.forget.BeaverTails_harmful.args.hf_args.split=30k_train \\
      trainer.args.num_train_epochs=1 \\
      trainer.args.per_device_train_batch_size=1
    """
    
    logger.info(config_info)

def explain_judge_model():
    """Explain the judge model."""
    print_header("JUDGE LLM MODEL")
    
    judge_info = """
    The Judge LLM is a separate language model used to evaluate safety:
    
    Default Judge Model:
    ───────────────────
    meta-llama/Llama-3.2-1B-Instruct
    
    Role:
    ────
    For each (question, response) pair from the main model:
    1. The judge LLM evaluates whether the response is safe or unsafe
    2. Returns one of: "SAFE" or "UNSAFE"
    3. Based on this verdict, we decide whether to unlearn
    
    Safety Criteria:
    ────────────────
    UNSAFE: Contains harmful, dangerous, unethical, illegal, or toxic content
    SAFE: Refuses harmful requests, provides harmless info, or is benign
    
    Judge Model Parameters:
    ──────────────────────
    - Max new tokens: 10 (only needs to output SAFE/UNSAFE)
    - Temperature: 0 (deterministic output)
    - Device: Same GPU as main model
    
    Efficiency Considerations:
    ──────────────────────────
    - Judge model can be smaller than main model
    - 1B parameters is usually sufficient for binary classification
    - Runs in eval mode (no gradients)
    - Separate from main training loop
    """
    
    logger.info(judge_info)

def show_quick_start():
    """Show quick start instructions."""
    print_header("QUICK START")
    
    quick_start = """
    To run the continual unlearning framework:
    
    1. MINIMAL EXAMPLE (Small dataset, for testing):
       ─────────────────────────────────────────────
       python src/train.py \\
         -c job_type=local \\
         --config-path=configs \\
         --config-name=train \\
         trainer=ContinualUnlearn \\
         'trainer.args.num_train_epochs=1' \\
         'trainer.args.per_device_train_batch_size=1' \\
         'trainer.args.logging_steps=1' \\
         'trainer.args.max_steps=5'
    
    2. WITH BEAVERTAILS DATASET:
       ─────────────────────────────
       python src/train.py \\
         -c job_type=local \\
         --config-path=configs \\
         --config-name=experiment/unlearn/beavertails/continual
    
    3. WITH GRADDIFF METHOD:
       ─────────────────────
       python src/train.py \\
         -c job_type=local \\
         --config-path=configs \\
         --config-name=experiment/unlearn/beavertails/continual_graddiff
    
    EXPECTED OUTPUT:
    ────────────────
    For each sample, you'll see:
    ✓ The input question
    ✓ The model's generated output
    ✓ The judge's safety decision (SAFE/UNSAFE)
    ✓ Real-time statistics:
      - Total samples processed
      - Number of safe samples (skipped)
      - Number of unsafe samples (unlearned)
      - Jailbreak rate percentage
    
    MONITORING PROGRESS:
    ───────────────────
    Watch the Jailbreak Rate:
    - If decreasing over time: unlearning is working!
    - If stable/increasing: model may need more training
    """
    
    logger.info(quick_start)

def main():
    """Run the demonstration."""
    parser = argparse.ArgumentParser(description='Continual Unlearning Framework Demo')
    parser.add_argument(
        '--section',
        choices=['all', 'flow', 'judge', 'stats', 'methods', 'config', 'quick'],
        default='all',
        help='Which section to demonstrate'
    )
    args = parser.parse_args()
    
    print_header("CONTINUAL TEST-TIME UNSAFE UNLEARNING FRAMEWORK")
    
    if args.section in ['all', 'judge']:
        demo_judge_safety_evaluation()
    
    if args.section in ['all', 'flow']:
        demo_continual_unlearning_flow()
    
    if args.section in ['all', 'stats']:
        demo_statistical_tracking()
    
    if args.section in ['all', 'methods']:
        demo_unlearning_methods()
    
    if args.section in ['all', 'config']:
        explain_configuration()
    
    if args.section in ['all']:
        explain_judge_model()
    
    if args.section in ['all', 'quick']:
        show_quick_start()
    
    print_header("NEXT STEPS")
    logger.info("""
    The continual unlearning framework is ready to use!
    
    Key Files:
    ──────────
    • Implementation: src/trainer/unlearn/continual.py
    • Configs: configs/trainer/ContinualUnlearn.yaml
    •          configs/trainer/ContinualGradDiff.yaml
    • Experiments: configs/experiment/unlearn/beavertails/
    
    To start training, use:
    python src/train.py --config-path=configs \\
      --config-name=experiment/unlearn/beavertails/continual
    
    For more information, see the README and documentation.
    """)

if __name__ == "__main__":
    main()
