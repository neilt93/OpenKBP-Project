from .base import BasePerturbation, load_ct_volume, save_ct_volume, create_perturbed_patient, load_structure_mask
from .p1_acquisition_noise import AcquisitionNoise
from .p2_bone_shift import BoneWeightedShift
from .p3_bias_field import BiasField
from .p4_resolution import ResolutionDegradation
from .p5_dental_artifact import DentalArtifact
from .p6_scatter_cupping import ScatterCupping
from .p7_ring import RingArtifact
from .p8_truncation import Truncation

ALL_PERTURBATIONS = [
    AcquisitionNoise,
    BoneWeightedShift,
    BiasField,
    ResolutionDegradation,
    DentalArtifact,
    # CBCT-characteristic families (commissioning-tolerances study for online adaptive RT):
    ScatterCupping,
    RingArtifact,
    Truncation,
]
