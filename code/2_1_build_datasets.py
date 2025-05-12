from pathlib import Path
import cvae.models.Datasets.SamplingDataset as sd
import cvae.tokenizer

BASE_DIR = Path("cache/build_tensordataset/multitask_tensors")
SPLITS = ["trn", "tst", "hld"]
SAVE_DIR = Path("cache/sampling_dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NPROPS = 5
SAMPLE_POOL = 10000
RECENT_PROP_CAP = 4000

def main():
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    for split in SPLITS:
        input_dir = BASE_DIR / split
        output_path = SAVE_DIR / f"{split}_sampling_dataset.pkl"

        print(f"\n🔧 Building sampling dataset for split: {split}")
        dataset = sd.SamplingDataset(
            path=input_dir,
            tokenizer=tokenizer,
            nprops=NPROPS,
            sample_pool=SAMPLE_POOL,
            recent_prop_cap=RECENT_PROP_CAP
        )

        sd.save_sampling_dataset(dataset, output_path)
        print(f"✅ Saved {len(dataset)} examples to {output_path}")

if __name__ == "__main__":
    main()
