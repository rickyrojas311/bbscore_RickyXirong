"""
PodcastECoG word-level benchmark for language models.
 
Pipeline:
    1. Load podcast transcript with word-level timestamps
    2. For each word, run language model on cumulative context
    3. Load high-gamma ECoG data epoched at word onsets
    4. Ridge regression with KFold CV
    5. Score: per-electrode Pearson correlation
 
Key differences from LeBel2023TRBenchmark:
    - Word-level (not TR-level): each word is a sample
    - No HRF delay: ECoG is a direct neural measurement
    - No story-level GroupKFold: single podcast → use KFold on
      contiguous time blocks to avoid temporal leakage
    - Context window: cumulative text up to each word, truncated
      to model's max context length
"""
 
import os
import pickle
import datetime
import numpy as np
import torch
from typing import Union, List, Optional
 
from sklearn.datasets import get_data_home
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
 
from data.PodcastECoG import (
    PodcastECoGStimulusSet,
    PodcastECoGAssembly,
)
from metrics import METRICS
from metrics.utils import pearson_correlation_scorer
from models import get_model_class_and_id
from benchmarks import BENCHMARK_REGISTRY
 
 
class PodcastECoGBenchmark:
    """
    Word-level encoding model benchmark for the Podcast ECoG dataset.
 
    Evaluates how well a language model's intermediate representations
    predict high-gamma ECoG activity during naturalistic podcast
    listening.
    """
 
    def __init__(
        self,
        model_identifier: str,
        layer_name: Union[str, List[str]],
        subjects: Optional[List[str]] = None,
        n_cv_folds: int = 5,
        epoch_tmin: float = 0.0,
        epoch_tmax: float = 0.5,
        context_words: int = 512,
        batch_size: Union[int, List[int]] = None,
        debug: bool = False,
    ):
        """
        Args:
            model_identifier: Model name (e.g. 'gpt2_small',
                'bert_large').
            layer_name: Which layer(s) to extract features from.
            subjects: Which ECoG subjects to include
                (default: all 9).
            n_cv_folds: Number of cross-validation folds.
            epoch_tmin: ECoG epoch start relative to word onset (s).
            epoch_tmax: ECoG epoch end relative to word onset (s).
            context_words: Max number of preceding words to include
                as context when running the language model.
            batch_size: Not used directly but kept for run.py compat.
            debug: Print extra diagnostics.
        """
        self.debug = debug
        self.subjects = subjects
        self.n_cv_folds = n_cv_folds
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
        self.context_words = context_words
        self.model_identifier = model_identifier
 
        if isinstance(batch_size, list):
            self.batch_size = batch_size[0] if batch_size else 4
        else:
            self.batch_size = batch_size or 4
 
        self.layer_name = layer_name
        self.layer_names = (
            layer_name if isinstance(layer_name, list)
            else [layer_name]
        )
 
        # Initialize model (same pattern as LeBel2023TRBenchmark)
        self.model_class, self.model_id_mapping = (
            get_model_class_and_id(model_identifier)
        )
        self.model_instance = self.model_class()
        self.model = self.model_instance.get_model(
            self.model_id_mapping
        )
        self.device = (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.eval()
        self.model.to(self.device)
 
        # Hook storage and registration
        self.features = {l: [] for l in self.layer_names}
        self._register_hooks()
 
        self.metrics = {}
        self.metric_params = {}
 
        data_home = get_data_home()
        results_base = os.environ.get('RESULTS_PATH', data_home)
        self.results_dir = os.path.join(results_base, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
 
    # ── Hooks (identical to LeBel2023TRBenchmark) ────────────────────
 
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        def hook_fn_factory(layer_id):
            def hook_fn(module, input, output):
                self.features[layer_id].append(output)
            return hook_fn
 
        for l_name in self.layer_names:
            found = False
            for name, module in self.model.named_modules():
                if name == l_name:
                    if isinstance(module, torch.nn.ModuleList):
                        module[-1].register_forward_hook(
                            hook_fn_factory(l_name)
                        )
                    else:
                        module.register_forward_hook(
                            hook_fn_factory(l_name)
                        )
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Layer '{l_name}' not found in model."
                )
 
    # ── Feature extraction ───────────────────────────────────────────
 
    def _extract_word_feature(self, text: str) -> Optional[np.ndarray]:
        """
        Run the language model on a text string and extract the
        feature vector for the last word.
 
        Uses the model's preprocess → forward → postprocess flow,
        identical to LeBel2023TRBenchmark._extract_tr_feature().
 
        Args:
            text: Cumulative context ending with the target word.
 
        Returns:
            Feature vector (D,), or None if text is empty.
        """
        if not text.strip():
            return None
 
        # Clear hook storage
        for l in self.layer_names:
            self.features[l] = []
 
        # Preprocess
        input_ids = self.model_instance.preprocess_fn(text)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
 
        # Forward pass
        with torch.inference_mode():
            _ = self.model(input_ids)
 
        # Collect features from target layers
        layer_features = {}
        for l_name in self.layer_names:
            if not self.features[l_name]:
                raise ValueError(
                    f"No features captured for layer {l_name}"
                )
            feat = self.features[l_name][0]
            if isinstance(feat, tuple):
                feat = feat[0]
 
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            processed = self.model_instance.postprocess_fn(feat)
            if isinstance(processed, torch.Tensor):
                processed = processed.cpu().numpy()
            layer_features[l_name] = processed.squeeze()
 
        if len(self.layer_names) == 1:
            return layer_features[self.layer_names[0]]
        return np.concatenate(
            [layer_features[l] for l in self.layer_names],
            axis=-1,
        )
 
    def _extract_all_word_features(
        self, stimulus_set: PodcastECoGStimulusSet
    ) -> np.ndarray:
        """
        Extract features for every word in the podcast.
 
        For each word, builds cumulative context (up to
        self.context_words preceding words) and runs the model.
 
        This is analogous to LeBel2023TRBenchmark.
        _extract_story_features(), but over words instead of TRs.
 
        Returns:
            (n_words, D) feature matrix
        """
        words = stimulus_set.words
        n_words = len(words)
        features_list = []
 
        for i in range(n_words):
            # Build cumulative context: up to context_words
            # preceding words + current word
            start = max(0, i + 1 - self.context_words)
            context = " ".join(words[start:i + 1])
 
            feat = self._extract_word_feature(context)
            features_list.append(feat)
 
            if self.debug and (i + 1) % 500 == 0:
                print(f"  Extracted features for {i + 1}/{n_words} "
                      f"words")
 
        # Determine feature dim
        feat_dim = None
        for f in features_list:
            if f is not None:
                feat_dim = f.shape[-1]
                break
 
        if feat_dim is None:
            raise ValueError("No features extracted from any word.")
 
        result = np.zeros((n_words, feat_dim), dtype=np.float32)
        for i, f in enumerate(features_list):
            if f is not None:
                result[i] = f
 
        return result
 
    # ── Ridge regression ─────────────────────────────────────────────
 
    def _run_kfold_ridge(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Ridge regression with KFold CV on contiguous time blocks.
 
        Unlike LeBel2023 which uses GroupKFold (stories as groups),
        we use standard KFold on the sequential word samples.
        Contiguous blocks prevent temporal leakage.
 
        Args:
            X: (n_words, D) model features
            y: (n_words, n_electrodes) ECoG data
 
        Returns:
            dict with 'pearson' and 'r2' arrays
        """
        n_splits = min(self.n_cv_folds, X.shape[0])
        kf = KFold(n_splits=n_splits, shuffle=False)
 
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0,
                  100.0, 1e4, 1e6]
 
        pearson_scores = []
        r2_scores = []
 
        for fold_idx, (train_idx, val_idx) in enumerate(
            kf.split(X)
        ):
            print(
                f"  Fold {fold_idx + 1}/{n_splits}: "
                f"train={len(train_idx)}, val={len(val_idx)}"
            )
 
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
 
            model = RidgeCV(
                alphas=alphas, store_cv_results=False
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
 
            # Per-electrode Pearson correlation
            fold_pearson = np.array([
                pearson_correlation_scorer(
                    y_val[:, i], preds[:, i]
                )
                for i in range(y.shape[1])
            ])
            pearson_scores.append(fold_pearson)
 
            # Per-electrode R²
            from sklearn.metrics import r2_score
            fold_r2 = np.array([
                r2_score(y_val[:, i], preds[:, i])
                for i in range(y.shape[1])
            ])
            r2_scores.append(fold_r2)
 
            print(
                f"    Median Pearson: "
                f"{np.median(fold_pearson):.4f}"
            )
 
        return {
            'pearson': np.array(pearson_scores),
            'r2': np.array(r2_scores),
        }
 
    # ── Compatibility with run.py ────────────────────────────────────
 
    def initialize_rp(self, rp):
        if rp is not None:
            print("Warning: Random projection not supported "
                  "for PodcastECoG benchmark. Ignoring.")
 
    def initialize_aggregation(self, mode):
        pass
 
    def add_metric(self, name, metric_params=None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params
 
    # ── Main pipeline ────────────────────────────────────────────────
 
    def run(self):
        """
        Main word-level encoding model pipeline.
 
        Mirrors LeBel2023TRBenchmark.run() structure:
            1. Load stimuli
            2. Load neural data
            3. Extract model features
            4. Ridge regression
            5. Score and save
        """
        # 1. Load stimuli
        print("Loading podcast transcript...")
        stimulus_set = PodcastECoGStimulusSet()
        n_words = len(stimulus_set.words)
        print(f"  {n_words} words, "
              f"{stimulus_set.word_offsets[-1]:.0f}s duration")
 
        # 2. Load ECoG assembly
        print("Loading ECoG assembly...")
        assembly = PodcastECoGAssembly(
            subjects=self.subjects,
            epoch_tmin=self.epoch_tmin,
            epoch_tmax=self.epoch_tmax,
        )
        ecog_data, ncsnr = assembly.get_assembly(
            stimulus_set=stimulus_set
        )
        n_electrodes = ecog_data.shape[1]
        print(f"  {ecog_data.shape[0]} words x "
              f"{n_electrodes} electrodes")
 
        # 3. Extract model features for every word
        print(f"Extracting features from {self.model_identifier} "
              f"(layer: {self.layer_name})...")
        X = self._extract_all_word_features(stimulus_set)
        print(f"  Feature matrix: {X.shape}")
 
        # Verify alignment
        assert X.shape[0] == ecog_data.shape[0], (
            f"Feature/ECoG mismatch: {X.shape[0]} vs "
            f"{ecog_data.shape[0]}"
        )
 
        # 4. Ridge regression
        y = ecog_data
        print(f"Running ridge regression "
              f"({self.n_cv_folds}-fold CV)...")
        fold_scores = self._run_kfold_ridge(X, y)
 
        # 5. Aggregate results
        median_pearson = np.median(
            fold_scores['pearson'], axis=0
        )
        median_r2 = np.median(fold_scores['r2'], axis=0)
 
        results = {
            'median_pearson_per_electrode': median_pearson,
            'median_r2_per_electrode': median_r2,
            'global_median_pearson': float(
                np.median(median_pearson)
            ),
            'global_median_r2': float(np.median(median_r2)),
            'n_words': n_words,
            'n_electrodes': n_electrodes,
            'n_subjects': len(
                self.subjects
                if self.subjects
                else PodcastECoGAssembly().subjects
            ),
            'epoch_window': [self.epoch_tmin, self.epoch_tmax],
            'context_words': self.context_words,
            'timestamp': datetime.datetime.utcnow().isoformat(),
        }
 
        print(f"\nResults:")
        print(f"  Median Pearson (all electrodes): "
              f"{results['global_median_pearson']:.4f}")
        print(f"  Median R² (all electrodes): "
              f"{results['global_median_r2']:.4f}")
 
        # 6. Save results
        layer_str = (
            self.layer_name if isinstance(self.layer_name, str)
            else "_".join(self.layer_names)
        )
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_identifier}_{layer_str}_"
            f"PodcastECoGBenchmark.pkl",
        )
 
        merged = {
            "metrics": results,
            "fold_scores": fold_scores,
        }
        with open(results_file, 'wb') as f:
            pickle.dump(merged, f)
        print(f"Results saved to {results_file}")
 
        return merged

BENCHMARK_REGISTRY["PodcastECoG"] = PodcastECoGBenchmark 


