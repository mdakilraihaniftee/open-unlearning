#!/usr/bin/env python3
"""
Continual Unlearning Training Script
=====================================

This script runs the continual test-time unsafe unlearning framework with:
- A real dataset (TOFU or BeaverTails)
- Judge LLM for safety evaluation
- Real-time per-sample output and statistics
- Jailbreak rate tracking

Run with:
    python run_continual_unlearning.py --num-samples 5 --method gradient_ascent
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import logging
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables if needed."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    # Load .env if exists
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                    logger.info(f"Loaded env: {key}")

def load_config(config_name: str) -> DictConfig:
    """Load a hydra configuration file."""
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any previous hydra instance
    GlobalHydra.instance().clear()
    
    config_dir = Path(__file__).parent / "configs"
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    
    return cfg

def print_section(title, width=80):
    """Print a formatted section title."""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width)

def main():
    parser = argparse.ArgumentParser(
        description='Run Continual Unlearning with Judge LLM'
    )
    parser.add_argument(
        '--method',
        choices=['gradient_ascent', 'graddiff'],
        default='gradient_ascent',
        help='Unlearning method to use'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to process'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (should be 1 for per-sample evaluation)'
    )
    parser.add_argument(
        '--dataset',
        choices=['beavertails', 'tofu'],
        default='beavertails',
        help='Dataset to use'
    )
    parser.add_argument(
        '--judge-model',
        default='meta-llama/Llama-3.2-1B-Instruct',
        help='Judge LLM model name'
    )
    parser.add_argument(
        '--model',
        default='Llama-3.2-1B-Instruct',
        help='Main model to train'
    )
    parser.add_argument(
        '--output-dir',
        default='./outputs/continual_unlearn',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without training'
    )
    args = parser.parse_args()
    
    print_section("CONTINUAL UNLEARNING TRAINING SETUP")
    
    setup_environment()
    
    logger.info(f"Method: {args.method}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Judge Model: {args.judge_model}")
    logger.info(f"Number of Samples: {args.num_samples}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    print_section("LOADING CONFIGURATION")
    
    try:
        # Determine config name based on method and dataset
        config_name = f"experiment/unlearn/{args.dataset}/continual"
        if args.method == 'graddiff':
            config_name += "_graddiff"
        
        logger.info(f"Loading config: {config_name}")
        cfg = load_config(config_name)
        
        # Override settings
        cfg.trainer.args.per_device_train_batch_size = args.batch_size
        cfg.trainer.args.max_steps = args.num_samples
        cfg.trainer.args.logging_steps = 1
        cfg.trainer.args.output_dir = args.output_dir
        cfg.trainer.method_args.judge_model_name = args.judge_model
        
        logger.info("✓ Configuration loaded successfully")
        
        if args.dry_run:
            print_section("CONFIGURATION DETAILS")
            print(OmegaConf.to_yaml(cfg))
            logger.info("\n✓ Dry run complete. Configuration is valid.")
            return
        
        print_section("LOADING COMPONENTS")
        
        # Import required modules
        from data import get_data, get_collators
        from model import get_model
        from trainer import load_trainer
        from trainer.utils import seed_everything
        
        # Setup
        seed_everything(cfg.get('seed', 42))
        logger.info(f"✓ Seed set to {cfg.get('seed', 42)}")
        
        # Load model
        logger.info(f"Loading model: {args.model}")
        model, tokenizer = get_model(cfg.model)
        logger.info(f"✓ Model loaded: {model.config.model_type}")
        logger.info(f"  Device: {next(model.parameters()).device}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load data
        logger.info("Loading dataset...")
        data = get_data(
            cfg.data,
            mode='train',
            tokenizer=tokenizer,
            template_args=cfg.model.template_args
        )
        
        forget_dataset = data.get("forget", data.get("train"))
        retain_dataset = data.get("retain", None)
        
        logger.info(f"✓ Forget dataset size: {len(forget_dataset)}")
        if retain_dataset:
            logger.info(f"✓ Retain dataset size: {len(retain_dataset)}")
        
        # Limit to num_samples
        if hasattr(forget_dataset, '__getitem__'):
            actual_samples = min(args.num_samples, len(forget_dataset))
            logger.info(f"✓ Will process {actual_samples} samples")
        
        # Load collator
        logger.info("Loading data collator...")
        collator = get_collators(cfg.collator, tokenizer=tokenizer)
        logger.info("✓ Data collator loaded")
        
        # Load trainer
        logger.info("Loading trainer with continual unlearning...")
        trainer, trainer_args = load_trainer(
            trainer_cfg=cfg.trainer,
            model=model,
            train_dataset=forget_dataset,
            eval_dataset=retain_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            evaluators=None,
            template_args=cfg.model.template_args,
        )
        logger.info(f"✓ Trainer loaded: {trainer.__class__.__name__}")
        logger.info(f"  Judge model: {trainer.judge_model_name}")
        logger.info(f"  Max new tokens: {trainer.max_new_tokens}")
        
        print_section("CONTINUAL UNLEARNING TRAINING")
        logger.info("""
This will now run continual test-time unsafe unlearning.

For each sample, you will see:
  ✓ Input question
  ✓ Model output
  ✓ Judge LLM decision (SAFE/UNSAFE)
  ✓ Real-time statistics (safe/unsafe counts, jailbreak rate)

If the judge deems the output UNSAFE, the model will unlearn it.
If the judge deems it SAFE, unlearning will be skipped.

Starting training...\
        """)
        
        if trainer_args.do_train:
            trainer.train()
            logger.info("✓ Training completed")
        
        # Print final statistics
        print_section("FINAL STATISTICS")
        logger.info(f"Total samples processed: {trainer._total_samples}")
        logger.info(f"Safe samples (skipped): {trainer._safe_samples}")
        logger.info(f"Unsafe samples (unlearned): {trainer._unsafe_samples}")
        
        if trainer._total_samples > 0:
            jailbreak_rate = (trainer._unsafe_samples / trainer._total_samples) * 100
            logger.info(f"Final jailbreak rate: {jailbreak_rate:.2f}%")
        
        logger.info(f"\n✓ Results saved to: {trainer_args.output_dir}")
        
    except Exception as e:
        logger.error(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
