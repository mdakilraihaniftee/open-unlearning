from data.qa import QADataset
from data.utils import load_hf_dataset, add_dataset_index


class BeaverTailsDataset(QADataset):
    def __init__(self, is_safe_filter=None, *args, **kwargs):
        # We need to load and filter before calling super if we want to reuse QADataset logic
        # But QADataset calls load_hf_dataset in __init__.
        # So we override __init__ slightly or just let it load then filter.
        super().__init__(*args, **kwargs)
        if is_safe_filter is not None:
            self.data = self.data.filter(lambda x: x["is_safe"] == is_safe_filter)
            # Re-index after filtering to ensure indices are contiguous if needed
            # Actually, add_dataset_index was already called in super().__init__
            # But the indices will now have gaps.
            # If the evaluation relies on contiguous indices, we might need to re-add them.
            # TOFU evaluation doesn't strictly require contiguous indices but uniqueness.
            # However, for clarity, let's keep the original indices for reference if possible,
            # or re-index if it's cleaner.
            # The base evaluator uses the index to store results.
            # So gaps are fine.
