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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict
 
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
        subjects: Optional[List[str]] = ["sub-01"], # only run one subject to reduce memory load
        n_cv_folds: int = 5,
        epoch_tmin: float = -0.5,
        epoch_tmax: float = 1.0,
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
 
    def _extract_all_word_features_per_layer(
        self, stimulus_set: PodcastECoGStimulusSet
    ) -> Dict[str, np.ndarray]:
        """
        Like _extract_all_word_features but returns a dict
        {layer_name: (n_words, D)} instead of concatenating layers.
        """
        words = stimulus_set.words
        n_words = len(words)
        layer_lists: Dict[str, list] = {l: [] for l in self.layer_names}

        for i in range(n_words):
            start = max(0, i + 1 - self.context_words)
            context = " ".join(words[start:i + 1])

            if not context.strip():
                for l in self.layer_names:
                    layer_lists[l].append(None)
                continue

            for l in self.layer_names:
                self.features[l] = []

            input_ids = self.model_instance.preprocess_fn(context)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)

            with torch.inference_mode():
                _ = self.model(input_ids)

            for l_name in self.layer_names:
                feat = self.features[l_name][0]
                if isinstance(feat, tuple):
                    feat = feat[0]
                if feat.dim() == 2:
                    feat = feat.unsqueeze(0)
                processed = self.model_instance.postprocess_fn(feat)
                if isinstance(processed, torch.Tensor):
                    processed = processed.cpu().numpy()
                layer_lists[l_name].append(processed.squeeze())

            if self.debug and (i + 1) % 500 == 0:
                print(f"  Extracted features for {i + 1}/{n_words} words")

        result: Dict[str, np.ndarray] = {}
        for l_name in self.layer_names:
            feat_dim = next(
                f.shape[-1] for f in layer_lists[l_name] if f is not None
            )
            arr = np.zeros((n_words, feat_dim), dtype=np.float32)
            for i, f in enumerate(layer_lists[l_name]):
                if f is not None:
                    arr[i] = f
            result[l_name] = arr

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
 
    def _run_kfold_ridge_time_resolved(
        self,
        X: np.ndarray,
        ecog_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Time-resolved ridge regression: for each electrode, fit
        X -> ecog_3d[:, e, :] (all time points as multi-output),
        then compute per-time Pearson r.

        Args:
            X:        (n_words, D)
            ecog_3d:  (n_words, n_electrodes, n_times)

        Returns:
            (n_times,) mean Pearson r averaged across electrodes and folds
        """
        n_words, n_electrodes, n_times = ecog_3d.shape
        n_splits = min(self.n_cv_folds, n_words)
        kf = KFold(n_splits=n_splits, shuffle=False)
        splits = list(kf.split(X))
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1e4, 1e6]

        # (n_electrodes, n_times) — mean across folds
        electrode_time_scores = np.zeros((n_electrodes, n_times))

        for e in range(n_electrodes):
            if e % 20 == 0:
                print(f"    Electrode {e}/{n_electrodes}...")
            fold_scores = np.zeros((n_splits, n_times))
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = ecog_3d[train_idx, e, :]  # (n_train, n_times)
                y_val   = ecog_3d[val_idx,   e, :]  # (n_val,   n_times)

                model = RidgeCV(alphas=alphas, store_cv_results=False)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)         # (n_val, n_times)

                for t in range(n_times):
                    fold_scores[fold_idx, t] = pearson_correlation_scorer(
                        y_val[:, t], preds[:, t]
                    )
            electrode_time_scores[e] = fold_scores.mean(axis=0)

        return electrode_time_scores.mean(axis=0)  # (n_times,)

    def _plot_time_resolved(
        self,
        layer_time_scores: Dict[str, np.ndarray],
        times: np.ndarray,
        save_path: str,
    ):
        """
        Plot average Pearson r vs time (in seconds) for each layer.

        Args:
            layer_time_scores: {layer_name: (n_times,)}
            times:             (n_times,) time axis in seconds
            save_path:         path to save the figure (.png)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('viridis')
        n_layers = len(layer_time_scores)

        for i, (layer_name, scores) in enumerate(layer_time_scores.items()):
            color = cmap(i / max(n_layers - 1, 1))
            ax.plot(times, scores, label=layer_name, color=color)

        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Average Pearson r across electrodes')
        ax.set_title(
            f'Time-resolved encoding: {self.model_identifier}'
        )
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                  fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved to {save_path}")

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
 
        # 2. Load ECoG assembly  ->  (n_words, n_electrodes, n_times)
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
        n_times      = ecog_data.shape[2]
        print(f"  {ecog_data.shape[0]} words x "
              f"{n_electrodes} electrodes x {n_times} timepoints")

        # Time axis in seconds
        times = np.linspace(self.epoch_tmin, self.epoch_tmax, n_times)

        # 3. Extract per-layer features
        print(f"Extracting features from {self.model_identifier}...")
        layer_features = self._extract_all_word_features_per_layer(
            stimulus_set
        )
        for l, X in layer_features.items():
            print(f"  {l}: {X.shape}")
            assert X.shape[0] == ecog_data.shape[0], (
                f"Feature/ECoG mismatch for {l}: "
                f"{X.shape[0]} vs {ecog_data.shape[0]}"
            )

        # 4. Time-resolved ridge regression for each layer
        layer_time_scores: Dict[str, np.ndarray] = {}
        for l_name, X in layer_features.items():
            print(f"Running time-resolved ridge for {l_name} "
                  f"({self.n_cv_folds}-fold CV)...")
            layer_time_scores[l_name] = (
                self._run_kfold_ridge_time_resolved(X, ecog_data)
            )
            peak = float(np.max(layer_time_scores[l_name]))
            print(f"  Peak Pearson: {peak:.4f}")

        # 5. Summary over time (mean-over-time Pearson per layer)
        layer_summary = {
            l: float(np.mean(s))
            for l, s in layer_time_scores.items()
        }

        results = {
            'layer_time_scores': layer_time_scores,  # {layer: (n_times,)}
            'times': times,                           # (n_times,) in seconds
            'layer_summary_pearson': layer_summary,   # {layer: float}
            'n_words': n_words,
            'n_electrodes': n_electrodes,
            'n_times': n_times,
            'epoch_window': [self.epoch_tmin, self.epoch_tmax],
            'context_words': self.context_words,
            'timestamp': datetime.datetime.utcnow().isoformat(),
        }

        print(f"\nPer-layer mean Pearson (averaged over time):")
        for l, v in layer_summary.items():
            print(f"  {l}: {v:.4f}")

        # 6. Save results
        layer_str = (
            self.layer_name if isinstance(self.layer_name, str)
            else "_".join(self.layer_names)
        )
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_identifier}_{layer_str}_"
            f"PodcastECoGBenchmark_timeresolved.pkl",
        )
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_file}")

        # 7. Plot
        fig_path = results_file.replace('.pkl', '.png')
        self._plot_time_resolved(layer_time_scores, times, fig_path)

        return results

BENCHMARK_REGISTRY["PodcastECoG"] = PodcastECoGBenchmark 


