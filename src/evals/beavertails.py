from evals.base import Evaluator


class BeaverTailsEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("BEAVERTAILS", eval_cfg, **kwargs)
