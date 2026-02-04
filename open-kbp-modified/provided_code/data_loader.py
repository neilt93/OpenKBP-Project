from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
from more_itertools import windowed
from numpy.typing import NDArray
from tqdm import tqdm

from provided_code.batch import DataBatch
from provided_code.data_shapes import DataShapes
from provided_code.utils import get_paths, load_file


class DataLoader:
    """Loads OpenKBP csv data in structured format for dose prediction models."""

    # Normalization constants (per OpenKBP data-description.pdf)
    # Dataset has mix of 12-bit and 16-bit CT; official recommendation: clip to [0, 4095]
    CT_MIN = 0.0       # Minimum CT value (HU, already shifted positive)
    CT_MAX = 4095.0    # 12-bit max value per official docs
    DOSE_PRESCRIPTION = 70.0  # Prescription dose in Gy for normalization

    def __init__(self, patient_paths: List[Path], batch_size: int = 2, cache_data: bool = True, normalize: bool = True, precomputed_path: Optional[Path] = None):
        """
        :param patient_paths: list of the paths where data for each patient is stored
        :param batch_size: the number of data points to lead in a single batch
        :param cache_data: if True, pre-load and pre-shape all data into RAM for faster training
        :param normalize: if True, normalize CT to [0,1] and dose by prescription dose
        :param precomputed_path: if provided, load stacked data from .npz file instead of CSVs
        """
        self.patient_paths = patient_paths
        self.batch_size = batch_size
        self.cache_data = cache_data
        self.normalize = normalize
        self.precomputed_path = precomputed_path
        self._data_cache: Dict[str, dict] = {}  # Cache for raw data
        self._shaped_cache: Dict[str, Dict[str, NDArray]] = {}  # Cache for pre-shaped tensors
        # Pre-stacked arrays for fast batch slicing (eliminates Python loop overhead)
        self._stacked_data: Dict[str, NDArray] = {}
        self._patient_to_idx: Dict[str, int] = {}

        # Light processing of attributes
        self.paths_by_patient_id = {patient_path.stem: patient_path for patient_path in self.patient_paths}
        self.required_files: Optional[Dict] = None
        self.mode_name: Optional[str] = None

        # Parameters that should not be changed unless OpenKBP data is modified
        self.rois = dict(
            oars=["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"],
            targets=["PTV56", "PTV63", "PTV70"],
        )
        self.full_roi_list = sum(map(list, self.rois.values()), [])  # make a list of all rois
        self.num_rois = len(self.full_roi_list)
        self.data_shapes = DataShapes(self.num_rois)

    @property
    def patient_id_list(self) -> List[str]:
        return list(self.paths_by_patient_id.keys())

    def get_batches(self) -> Iterator[DataBatch]:
        batches = windowed(self.patient_paths, n=self.batch_size, step=self.batch_size)
        complete_batches = (batch for batch in batches if None not in batch)
        for batch_paths in tqdm(complete_batches):
            yield self.prepare_data(batch_paths)

    def get_patients(self, patient_list: List[str]) -> DataBatch:
        file_paths_to_load = [self.paths_by_patient_id[patient] for patient in patient_list]
        return self.prepare_data(file_paths_to_load)

    def set_mode(self, mode: str) -> None:
        """Set parameters based on `mode`."""
        self.mode_name = mode
        if mode == "training_model":
            required_data = ["dose", "ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        elif mode == "predicted_dose":
            required_data = [mode]
            self._force_batch_size_one()
        elif mode == "evaluation":
            required_data = ["dose", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
            self._force_batch_size_one()
        elif mode == "dose_prediction":
            required_data = ["ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
            self._force_batch_size_one()
        else:
            raise ValueError(f"Mode `{mode}` does not exist. Mode must be either training_model, prediction, predicted_dose, or evaluation")
        self.required_files = self.data_shapes.from_data_names(required_data)

        # Pre-load and pre-shape all data if caching is enabled
        if self.cache_data and not self._stacked_data:
            if self.precomputed_path and self.precomputed_path.exists():
                self._load_from_precomputed()
            else:
                self._preload_and_shape_all_data()

    def _force_batch_size_one(self) -> None:
        if self.batch_size != 1:
            self.batch_size = 1
            Warning("Batch size has been changed to 1 for dose prediction mode")

    def _load_from_precomputed(self) -> None:
        """Load pre-stacked data from .npz file for instant startup."""
        print(f"Loading precomputed data from {self.precomputed_path}...")
        data = np.load(self.precomputed_path)
        for key in data.files:
            self._stacked_data[key] = data[key]
        # Build patient index from patient_paths order
        for idx, patient_path in enumerate(self.patient_paths):
            self._patient_to_idx[patient_path.stem] = idx
        print(f"Loaded precomputed data. Shape per key: {[(k, v.shape) for k, v in self._stacked_data.items()]}")

    def _preload_and_shape_all_data(self) -> None:
        """Pre-load all patient data into stacked arrays for instant batch slicing."""
        n_patients = len(self.patient_paths)
        print(f"Pre-loading and stacking {n_patients} patients into memory...")

        # First pass: load all data and build patient index
        all_shaped: List[Dict[str, NDArray]] = []
        for idx, patient_path in enumerate(tqdm(self.patient_paths, desc="Loading data")):
            patient_id = patient_path.stem
            self._patient_to_idx[patient_id] = idx
            raw_data = self._load_data_from_disk(patient_path)
            shaped = {key: self.shape_data(key, raw_data) for key in self.required_files}
            all_shaped.append(shaped)

        # Second pass: stack into contiguous arrays (one per data type)
        print("Stacking into contiguous arrays...")
        for key in self.required_files:
            self._stacked_data[key] = np.stack([s[key] for s in all_shaped], axis=0)

        print(f"Data stacked. Shape per key: {[(k, v.shape) for k, v in self._stacked_data.items()]}")

    def shuffle_data(self) -> None:
        np.random.shuffle(self.patient_paths)

    def prepare_data(self, file_paths_to_load: List[Path]) -> DataBatch:
        """Prepares data containing samples in batch so that they are loaded in the proper shape: (n_samples, *dim, n_channels)"""

        patient_ids = [patient_path.stem for patient_path in file_paths_to_load]

        # Fast path: use pre-stacked arrays with direct slicing (no Python loops)
        if self.cache_data and self._stacked_data:
            indices = [self._patient_to_idx[pid] for pid in patient_ids]
            batch_data = DataBatch(
                patient_list=patient_ids,
                patient_path_list=file_paths_to_load,
                structure_mask_names=self.full_roi_list,
                **{key: self._stacked_data[key][indices] for key in self.required_files}
            )
            return batch_data

        # Fallback: compute on the fly (used when caching disabled)
        batch_data = DataBatch.initialize_from_required_data(self.required_files, self.batch_size)
        batch_data.patient_list = patient_ids
        batch_data.patient_path_list = file_paths_to_load
        batch_data.structure_mask_names = self.full_roi_list

        for index, patient_path in enumerate(file_paths_to_load):
            raw_data = self.load_data(patient_path)
            for key in self.required_files:
                batch_data.set_values(key, index, self.shape_data(key, raw_data))

        return batch_data

    def load_data(self, path_to_load: Path) -> Union[NDArray, dict[str, NDArray]]:
        """Load data, using cache if available."""
        patient_id = path_to_load.stem
        if self.cache_data and patient_id in self._data_cache:
            return self._data_cache[patient_id]
        return self._load_data_from_disk(path_to_load)

    def _load_data_from_disk(self, path_to_load: Path) -> Union[NDArray, dict[str, NDArray]]:
        """Load data in its raw form from disk."""
        data = {}
        if path_to_load.is_dir():
            files_to_load = get_paths(path_to_load)
            for file_path in files_to_load:
                # Load all files when caching, or only required files otherwise
                if self.cache_data:
                    data[file_path.stem] = load_file(file_path)
                else:
                    is_required = file_path.stem in self.required_files
                    is_required_roi = file_path.stem in self.full_roi_list
                    if is_required or is_required_roi:
                        data[file_path.stem] = load_file(file_path)
        else:
            data[self.mode_name] = load_file(path_to_load)

        return data

    def shape_data(self, key: str, data: dict) -> NDArray:
        """Shapes into form that is amenable to tensorflow and other deep learning packages."""

        shaped_data = np.zeros(self.required_files[key])

        if key == "structure_masks":
            for roi_idx, roi in enumerate(self.full_roi_list):
                if roi in data.keys():
                    np.put(shaped_data, self.num_rois * data[roi] + roi_idx, int(1))
        elif key == "possible_dose_mask":
            np.put(shaped_data, data[key], int(1))
        elif key == "voxel_dimensions":
            shaped_data = data[key]
        else:
            np.put(shaped_data, data[key]["indices"], data[key]["data"])

        # Apply normalization if enabled
        if self.normalize:
            if key == "ct":
                # CT normalization: clip to [0, 4095] and scale to [0, 1]
                shaped_data = np.clip(shaped_data, self.CT_MIN, self.CT_MAX)
                shaped_data = shaped_data / self.CT_MAX
            elif key in ("dose", "predicted_dose"):
                # Dose normalization: divide by prescription dose (70 Gy)
                shaped_data = shaped_data / self.DOSE_PRESCRIPTION

        return shaped_data
