from .sampling_dataset import SamplingDataset
from .restricted_dataset import PropertyGuaranteeDataset, SharedSampleTracker
from .inmemory_sequence_shift_dataset import InMemorySequenceShiftDataset

# Also import dataset classes defined in multitask_transformer.py
from cvae.models.multitask_transformer import (
    SequenceShiftDataset,
    RotatingModuloSequenceShiftDataset
)