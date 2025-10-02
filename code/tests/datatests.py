import torch
from cvae.tokenizer import SelfiesPropertyValTokenizer
from cvae.models.multitask_transformer import SequenceShiftDataset
import pathlib

tokenizer = SelfiesPropertyValTokenizer.load("brick/selfies_property_val_tokenizer")

def train_size_test():
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.PAD_IDX

    print(f"Loaded tokenizer with vocab size: {vocab_size}, pad_idx: {pad_idx}")

    dataset = SequenceShiftDataset(
        path="cache/build_tensordataset/multitask_tensors/trn",
        tokenizer=tokenizer,
        nprops=20,
    )

    max_id_seen = -1
    bad_idxs = []

    for idx in range(len(dataset)):
        try:
            selfies, teach, out = dataset[idx]

            for name, seq in [("selfies", selfies), ("teach", teach), ("out", out)]:
                if seq.max() >= vocab_size:
                    print(f"[BAD TOKEN] {name} @ index {idx} → max = {seq.max().item()}, should be < {vocab_size}")
                    bad_idxs.append(idx)
                if (seq < 0).any():
                    print(f"[NEGATIVE TOKEN] {name} @ index {idx} → min = {seq.min().item()}")
                    bad_idxs.append(idx)
                if seq.numel() == 0:
                    print(f"[EMPTY] {name} @ index {idx} has 0 elements")
                    bad_idxs.append(idx)
                if (seq == pad_idx).all():
                    print(f"[ALL PAD] {name} @ index {idx} — all tokens are PAD_IDX ({pad_idx})")
                    bad_idxs.append(idx)

            # Specific test for output
            if out.numel() == 0:
                print(f"[EMPTY OUTPUT] @ index {idx}")
                bad_idxs.append(idx)
            elif out.unique().numel() == 1:
                print(f"[LOW DIVERSITY OUTPUT] @ index {idx} — unique: {out.unique().tolist()}")
                bad_idxs.append(idx)

            max_id_seen = max(max_id_seen, teach.max().item(), out.max().item(), selfies.max().item())

        except Exception as e:
            print(f"[ERROR] Failed at index {idx}: {e}")
            bad_idxs.append(idx)

        if idx % 1000 == 0:
            print(f"Checked {idx} / {len(dataset)} examples...")

    print(f"\n✅ Finished checking {len(dataset)} examples.")
    print(f"Max token index seen: {max_id_seen}")
    if bad_idxs:
        print(f"\n❌ Found {len(set(bad_idxs))} bad examples (indices listed above).")
    else:
        print("🎉 No invalid token indices found.")

def seq_len_test():
    dataset = SequenceShiftDataset(
        path="cache/build_tensordataset/multitask_tensors/trn",
        tokenizer=tokenizer,
        nprops=20,
    )

    expected_shapes = {"selfies": None, "teach": None, "out": None}
    bad_idxs = []

    for idx in range(len(dataset)):
        try:
            selfies, teach, out = dataset[idx]
            sequences = {"selfies": selfies, "teach": teach, "out": out}

            # Set expected shapes from first example
            if idx == 0:
                for name, seq in sequences.items():
                    expected_shapes[name] = seq.shape
                print("Expected sequence shapes:")
                for name, shape in expected_shapes.items():
                    print(f"{name}: {shape}")

            # Check if sequences match their expected shapes
            for name, seq in sequences.items():
                if seq.shape != expected_shapes[name]:
                    print(f"[WRONG SHAPE] {name} @ index {idx} → shape = {seq.shape}, "
                          f"expected {expected_shapes[name]}")
                    bad_idxs.append(idx)

        except Exception as e:
            print(f"[ERROR] Failed at index {idx}: {e}")
            bad_idxs.append(idx)

        if idx % 1000 == 0:
            print(f"Checked {idx} / {len(dataset)} examples...")

    print(f"\n✅ Finished checking {len(dataset)} examples.")
    if bad_idxs:
        print(f"\n❌ Found {len(set(bad_idxs))} examples with wrong sequence shapes (indices listed above).")
    else:
        print("\n🎉 All sequences have consistent shapes:")
        for name, shape in expected_shapes.items():
            print(f"{name}: {shape}")
            
if __name__ == "__main__":
    train_size_test()
    seq_len_test()
