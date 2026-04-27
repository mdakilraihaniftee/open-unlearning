import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators
from trainer.qualitative_callback import run_qualitative_generation


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)

        # Run qualitative generation if requested
        num_samples = cfg.get("qualitative_num_samples", 3)
        if num_samples > 0:
            # Try to find a metric that has data populated
            sample_dataset = None
            for metric in evaluator.metrics.values():
                if hasattr(metric, "data") and metric.data is not None:
                    sample_dataset = metric.data
                    break
            
            if sample_dataset:
                run_qualitative_generation(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=sample_dataset,
                    num_samples=num_samples,
                    header=f"Qualitative Samples — {evaluator_name}"
                )


if __name__ == "__main__":
    main()
