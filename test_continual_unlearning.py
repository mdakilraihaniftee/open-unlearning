#!/usr/bin/env python3
"""
Test script for Continual Unlearning Framework
This script tests the continual test-time unsafe unlearning framework 
with a small dataset to verify it's working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all continual unlearning components import correctly."""
    logger.info("Testing imports...")
    try:
        from trainer.unlearn.continual import (
            ContinualUnlearnTrainer,
            ContinualGradDiff,
            ContinualJudgeMixin,
            JUDGE_PROMPT_TEMPLATE,
        )
        logger.info("✓ Successfully imported continual trainers")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import: {e}")
        return False

def test_judge_mixin_methods():
    """Test that ContinualJudgeMixin has all required methods."""
    logger.info("\nTesting ContinualJudgeMixin methods...")
    try:
        from trainer.unlearn.continual import ContinualJudgeMixin
        
        required_methods = [
            'init_judge',
            '_decode_question',
            '_generate_response',
            '_judge_safety',
            '_print_decision',
        ]
        
        for method in required_methods:
            if not hasattr(ContinualJudgeMixin, method):
                logger.error(f"✗ Missing method: {method}")
                return False
            logger.info(f"✓ Found method: {method}")
        
        logger.info("✓ All required methods found in ContinualJudgeMixin")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to verify methods: {e}")
        return False

def test_prompt_template():
    """Test that the judge prompt template is properly formatted."""
    logger.info("\nTesting judge prompt template...")
    try:
        from trainer.unlearn.continual import JUDGE_PROMPT_TEMPLATE
        
        test_prompt = JUDGE_PROMPT_TEMPLATE.format(
            question="What is the capital of France?",
            response="The capital of France is Paris."
        )
        
        required_keywords = ["safety judge", "SAFE", "UNSAFE", "Paris"]
        for keyword in required_keywords:
            if keyword not in test_prompt:
                logger.error(f"✗ Missing keyword in prompt: {keyword}")
                return False
        
        logger.info("✓ Judge prompt template is properly formatted")
        logger.info(f"Sample prompt:\n{test_prompt[:200]}...")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to test prompt template: {e}")
        return False

def test_trainer_registration():
    """Test that trainers are registered in the trainer registry."""
    logger.info("\nTesting trainer registration...")
    try:
        from trainer import TRAINER_REGISTRY
        
        trainers_to_check = ['ContinualUnlearnTrainer', 'ContinualGradDiff']
        
        for trainer_name in trainers_to_check:
            if trainer_name not in TRAINER_REGISTRY:
                logger.error(f"✗ {trainer_name} not registered")
                return False
            logger.info(f"✓ {trainer_name} is registered")
        
        logger.info("✓ All continual trainers are properly registered")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to verify registration: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("CONTINUAL UNLEARNING FRAMEWORK TEST")
    logger.info("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Judge Mixin Methods", test_judge_mixin_methods),
        ("Prompt Template", test_prompt_template),
        ("Trainer Registration", test_trainer_registration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
