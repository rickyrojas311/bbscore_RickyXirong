"""
PodcastECoG.py — BBScore data loader for the Podcast ECoG dataset
(Zada et al., 2025; OpenNeuro ds005574)
 
Drop this file into bbscore_public/data/PodcastECoG.py
 
Cross-platform behavior:
    All data is stored under SCIKIT_LEARN_DATA (set via environment
    variable).  On first run, files are auto-downloaded from
    OpenNeuro's public S3 bucket.  No hardcoded paths.
 
    StimulusSet data → SCIKIT_LEARN_DATA/PodcastECoGStimulusSet/ds005574/
    Assembly data    → SCIKIT_LEARN_DATA/PodcastECoGAssembly/ds005574/
 
Dataset on OpenNeuro (BIDS-iEEG):
    s3://openneuro.org/ds005574/
    ├── sub-01/ ... sub-09/
    │   └── ieeg/
    │       ├── sub-XX_task-podcast_channels.tsv
    │       ├── sub-XX_space-MNI152NLin2009aSym_electrodes.tsv
    │       └── sub-XX_task-podcast_ieeg.edf
    ├── derivatives/
    │   └── ecogprep/
    │       └── sub-XX/
    │           └── ieeg/
    │               ├── sub-XX_task-podcast_desc-highgamma_ieeg.fif
    │               └── sub-XX_task-podcast_ieeg.fif
    └── stimuli/
        ├── podcast_transcript.csv   # columns: word,start,end
        └── podcast.wav
"""
 
import os
import csv
import numpy as np
import warnings
from typing import Optional, List, Callable, Tuple
from data.base import BaseDataset
 
 
# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────
 
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 10)]
 
S3_BASE = "s3://openneuro.org/ds005574"
 
DEFAULT_EPOCH_TMIN = -0.5 # This can be overwritten by benchmark config
DEFAULT_EPOCH_TMAX = 1.0
 
 
# ──────────────────────────────────────────────────────────────────────
#  Stimulus Set
# ──────────────────────────────────────────────────────────────────────
 
class PodcastECoGStimulusSet(BaseDataset):
    """
    Stimulus set for the Podcast ECoG dataset.
 
    Loads stimuli/podcast_transcript.csv (columns: word, start, end).
    Each row is one spoken word with onset/offset time in seconds.
 
    Data location (auto-created):
        SCIKIT_LEARN_DATA/PodcastECoGStimulusSet/ds005574/stimuli/
 
    Analogous to LeBel2023StimulusSet which loads 84 TextGrid files
    into SCIKIT_LEARN_DATA/LeBel2023StimulusSet/ds003020/derivative/.
    """
 
    def __init__(
        self,
        root_dir: Optional[str] = None,
        preprocess: Optional[Callable] = None,
    ):
        super().__init__(root_dir)
        self.preprocess = preprocess
 
        # Mirrors LeBel2023: self.dataset_dir = .../ds003020
        self.dataset_dir = os.path.join(self.root_dir, "ds005574")
        self.stimuli_dir = os.path.join(self.dataset_dir, "stimuli")
 
        self.words: List[str] = []
        self.word_onsets: List[float] = []
        self.word_offsets: List[float] = []
 
        self._prepare_stimuli()
 
    def _prepare_stimuli(self):
        """Download transcript from S3 if not present, then parse."""
        transcript_path = os.path.join(
            self.stimuli_dir, "podcast_transcript.csv"
        )
 
        # Auto-download (same pattern as LeBel2023StimulusSet lines 31-43)
        if not os.path.isfile(transcript_path):
            s3_source = f"{S3_BASE}/stimuli/"
            try:
                print(f"Downloading transcript from {s3_source}...")
                self.fetch(
                    source=s3_source,
                    target_dir=self.dataset_dir,
                    filename="stimuli",
                    method="s3",
                    anonymous=True,
                )
            except Exception as e:
                print(f"Error downloading transcript: {e}")
 
        if not os.path.isfile(transcript_path):
            raise FileNotFoundError(
                f"Transcript not found at {transcript_path}.\n"
                f"Set SCIKIT_LEARN_DATA and re-run, or download "
                f"manually:\n  aws s3 sync --no-sign-request "
                f"{S3_BASE}/stimuli/ {self.stimuli_dir}/"
            )
 
        self._parse_transcript(transcript_path)
 
    def _parse_transcript(self, filepath: str):
        """
        Parse podcast_transcript.csv (CSV with header).
 
        Format:
            word,start,end
            Act,3.71,3.79
            "one,",3.99,4.19
            monkey,4.651,4.931
        """
        print(f"Loading transcript from: {filepath}")
 
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)  # comma-delimited, has header
            for row in reader:
                word = row["word"].strip()
                if not word:
                    continue
                try:
                    onset = float(row["start"])
                    offset = float(row["end"])
                except (ValueError, KeyError):
                    continue
                self.words.append(word)
                self.word_onsets.append(onset)
                self.word_offsets.append(offset)
 
        if not self.words:
            raise ValueError(f"No words parsed from {filepath}")
 
        print(
            f"Loaded {len(self.words)} words, "
            f"span: {self.word_onsets[0]:.1f}s – "
            f"{self.word_offsets[-1]:.1f}s"
        )
 
    # ── Public helpers ───────────────────────────────────────────────
 
    def get_full_text(self) -> str:
        """Return the entire transcript as a single string."""
        return " ".join(self.words)
 
    def get_words_with_times(self) -> List[Tuple[str, float, float]]:
        """Return list of (word, onset_sec, offset_sec)."""
        return list(zip(self.words, self.word_onsets, self.word_offsets))
 
    # ── BaseDataset interface ────────────────────────────────────────
 
    def __len__(self):
        return 1  # single podcast
 
    def __getitem__(self, idx):
        text = self.get_full_text()
        if self.preprocess:
            return self.preprocess(text)
        return text
 
 
# ──────────────────────────────────────────────────────────────────────
#  Assembly
# ──────────────────────────────────────────────────────────────────────
 
class PodcastECoGAssembly(BaseDataset):
    """
    Neural assembly for the Podcast ECoG dataset.
 
    Loads preprocessed high-gamma .fif files, epochs at word onsets,
    returns (n_words, n_electrodes).
 
    Data location (auto-created):
        SCIKIT_LEARN_DATA/PodcastECoGAssembly/ds005574/derivatives/ecogprep/
        SCIKIT_LEARN_DATA/PodcastECoGAssembly/ds005574/sub-XX/ieeg/
 
    Mirrors LeBel2023Assembly which returns (n_stories, n_voxels) from
        SCIKIT_LEARN_DATA/LeBel2023Assembly/ds003020/derivative/
 
    Key mapping:
        LeBel2023:    sample = story,  feature = fMRI voxel
        PodcastECoG:  sample = word,   feature = ECoG electrode
    """
 
    def __init__(
        self,
        root_dir: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        epoch_tmin: float = DEFAULT_EPOCH_TMIN,
        epoch_tmax: float = DEFAULT_EPOCH_TMAX,
    ):
        super().__init__(root_dir)
 
        self.dataset_dir = os.path.join(self.root_dir, "ds005574")
        self.ecogprep_dir = os.path.join(
            self.dataset_dir, "derivatives", "ecogprep"
        )
 
        self.subjects = subjects or ALL_SUBJECTS
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
 
    # ── Channel metadata ─────────────────────────────────────────────
 
    def _ensure_channel_metadata(self, subj: str):
        """Download sub-XX/ieeg/ from S3 if channels.tsv is missing."""
        channels_file = os.path.join(
            self.dataset_dir, subj, "ieeg",
            f"{subj}_task-podcast_channels.tsv",
        )
        if not os.path.isfile(channels_file):
            try:
                print(f"Downloading channel metadata for {subj}...")
                self.fetch(
                    source=f"{S3_BASE}/{subj}/ieeg/",
                    target_dir=os.path.join(self.dataset_dir, subj),
                    filename="ieeg",
                    method="s3",
                    anonymous=True,
                )
            except Exception as e:
                print(f"Error downloading channels for {subj}: {e}")
 
    def _load_channel_status(self, subj: str) -> dict:
        """
        Load sub-XX_task-podcast_channels.tsv (tab-separated).
 
        Columns: name  type  units  low_cutoff  high_cutoff
                 sampling_frequency  status  status_description
 
        Example:
            G1  ECOG  n/a  n/a  n/a  512.0  bad   no localization
            G2  ECOG  n/a  n/a  n/a  512.0  good  n/a
 
        Returns:
            dict mapping channel_name -> bool (True = good)
        """
        self._ensure_channel_metadata(subj)
 
        channels_file = os.path.join(
            self.dataset_dir, subj, "ieeg",
            f"{subj}_task-podcast_channels.tsv",
        )
        if not os.path.isfile(channels_file):
            warnings.warn(f"No channels.tsv found for {subj}")
            return {}
 
        status_map = {}
        with open(channels_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                name = row.get("name", "").strip()
                status = row.get("status", "good").strip().lower()
                status_map[name] = (status == "good")
 
        return status_map
 
    def _load_electrode_coords(self, subj: str) -> dict:
        """
        Load sub-XX_space-MNI152NLin2009aSym_electrodes.tsv.
 
        Columns: name  x  y  z  size  group
 
        Only electrodes with valid localizations appear in this file.
 
        Returns:
            dict mapping channel_name -> (x, y, z) MNI coordinates
        """
        electrodes_file = os.path.join(
            self.dataset_dir, subj, "ieeg",
            f"{subj}_space-MNI152NLin2009aSym_electrodes.tsv",
        )
        if not os.path.isfile(electrodes_file):
            return {}
 
        coords = {}
        with open(electrodes_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                name = row.get("name", "").strip()
                try:
                    coords[name] = (
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    )
                except (ValueError, KeyError):
                    continue
 
        return coords
 
    # ── High-gamma loading ───────────────────────────────────────────
 
    def _ensure_highgamma_downloaded(self, subj: str) -> str:
        """
        Return path to the high-gamma .fif, downloading from S3 if
        needed.
 
        File: derivatives/ecogprep/sub-XX/ieeg/
              sub-XX_task-podcast_desc-highgamma_ieeg.fif
        """
        hg_path = os.path.join(
            self.ecogprep_dir, subj, "ieeg",
            f"{subj}_task-podcast_desc-highgamma_ieeg.fif",
        )
 
        if not os.path.isfile(hg_path):
            s3_source = (
                f"{S3_BASE}/derivatives/ecogprep/{subj}/"
            )
            try:
                print(f"Downloading high-gamma for {subj}...")
                self.fetch(
                    source=s3_source,
                    target_dir=self.ecogprep_dir,
                    filename=subj,
                    method="s3",
                    anonymous=True,
                )
            except Exception as e:
                print(f"Error downloading ecogprep/{subj}: {e}")
 
        if not os.path.isfile(hg_path):
            raise FileNotFoundError(
                f"Not found: {hg_path}\n"
                f"Download manually:\n"
                f"  aws s3 sync --no-sign-request "
                f"{S3_BASE}/derivatives/ecogprep/{subj}/ "
                f"{os.path.join(self.ecogprep_dir, subj)}/"
            )
 
        return hg_path
 
    def _load_high_gamma(self, hg_path: str):
        """
        Load a preprocessed high-gamma .fif file via MNE.
 
        Returns:
            data: (n_channels, n_timepoints) numpy array
            sfreq: sampling frequency in Hz
            ch_names: list of channel names
        """
        try:
            import mne
            mne.set_log_level("WARNING")
        except ImportError:
            raise ImportError(
                "MNE-Python is required: pip install mne"
            )
 
        raw = mne.io.read_raw_fif(
            hg_path, preload=True, verbose=False
        )
        return raw.get_data(), raw.info["sfreq"], raw.ch_names
 
    # ── Epoching ─────────────────────────────────────────────────────
 
    def _epoch_by_words(
        self,
        data: np.ndarray,
        sfreq: float,
        word_onsets: List[float],
    ) -> np.ndarray:
        """
        Epoch continuous high-gamma at word onsets, averaging power
        in each [onset + tmin, onset + tmax] window.
 
        Returns:
            (n_words, n_channels) mean high-gamma power per word
        """
        n_channels = data.shape[0]
        n_words = len(word_onsets)
        n_total = data.shape[1]
        epoched = np.full((n_words, n_channels), np.nan)
 
        tmin_samp = int(round(self.epoch_tmin * sfreq))
        tmax_samp = int(round(self.epoch_tmax * sfreq))
 
        for i, onset in enumerate(word_onsets):
            start = int(round(onset * sfreq)) + tmin_samp
            end = int(round(onset * sfreq)) + tmax_samp
            start = max(0, start)
            end = min(n_total, end)
            if end <= start:
                continue
            epoched[i] = np.nanmean(data[:, start:end], axis=1)
 
        return epoched
 
    # ── Main interface ───────────────────────────────────────────────
 
    def get_assembly(
        self,
        word_onsets: Optional[List[float]] = None,
        stimulus_set: Optional[PodcastECoGStimulusSet] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and epoch ECoG data for all requested subjects.
 
        Mirrors LeBel2023Assembly.get_assembly() → (data, ncsnr).
 
        Args:
            word_onsets: Word onset times (seconds).
            stimulus_set: PodcastECoGStimulusSet instance (used to
                          extract word_onsets if not given directly).
 
        Returns:
            ecog_data: (n_words, n_electrodes) — good electrodes
                       concatenated across all subjects.
            ncsnr: (n_electrodes,) — placeholder (ones).
 
        Noise ceiling note:
            Podcast subjects heard the audio once (no repeats), so
            ncsnr cannot be estimated from repeat trials.  Returns
            ones.  Consider split-half electrode reliability for
            proper normalization.
        """
        if word_onsets is None:
            if stimulus_set is None:
                raise ValueError(
                    "Provide either word_onsets or stimulus_set."
                )
            word_onsets = stimulus_set.word_onsets
 
        all_subject_epochs = []
 
        for subj in self.subjects:
            print(f"Loading {subj}...")
 
            # 1. Channel quality (good/bad from channels.tsv)
            ch_status = self._load_channel_status(subj)
 
            # 2. Electrode MNI coords (for future region filtering)
            ch_coords = self._load_electrode_coords(subj)
 
            # 3. High-gamma data
            try:
                hg_path = self._ensure_highgamma_downloaded(subj)
                data, sfreq, ch_names = self._load_high_gamma(
                    hg_path
                )
            except FileNotFoundError as e:
                warnings.warn(str(e))
                continue
 
            # 4. Good-channel mask
            if ch_status:
                good_mask = np.array([
                    ch_status.get(ch, False) for ch in ch_names
                ])
            elif ch_coords:
                good_mask = np.array([
                    ch in ch_coords for ch in ch_names
                ])
            else:
                good_mask = np.ones(len(ch_names), dtype=bool)
 
            n_good = int(good_mask.sum())
            if n_good == 0:
                warnings.warn(f"No good electrodes for {subj}.")
                continue
 
            # 5. Epoch at word onsets
            epoched = self._epoch_by_words(data, sfreq, word_onsets)
 
            # 6. Keep good channels only
            epoched = epoched[:, good_mask]
 
            print(
                f"  {subj}: {n_good}/{len(ch_names)} good "
                f"electrodes, {epoched.shape[0]} words, "
                f"sfreq={sfreq:.0f}Hz"
            )
 
            all_subject_epochs.append(epoched)
 
        if not all_subject_epochs:
            raise ValueError(
                "No ECoG data loaded. Check that "
                "derivatives/ecogprep/ has .fif files."
            )
 
        # Concatenate across subjects (electrode axis)
        ecog_data = np.concatenate(all_subject_epochs, axis=1)
        ecog_data = np.nan_to_num(ecog_data)
 
        n_electrodes = ecog_data.shape[1]
        ncsnr = np.ones(n_electrodes, dtype=np.float32)
 
        print(
            f"Final assembly: {ecog_data.shape[0]} words x "
            f"{ecog_data.shape[1]} electrodes "
            f"({len(all_subject_epochs)} subjects)"
        )
 
        return ecog_data, ncsnr
 
    # ── BaseDataset interface ────────────────────────────────────────
 
    def __len__(self):
        return len(self.subjects)
 
    def __getitem__(self, idx):
        return None  # accessed via get_assembly()