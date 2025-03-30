import datasets
from datasets import load_dataset

dataset_name = "pubmed_qa"
dataset = load_dataset(dataset_name, "pqa_labeled")
dataset = dataset["train"].train_test_split(test_size=0.2,seed=42)

trainset = dataset.get("train")
testset = dataset.get("test")

trainset.save_to_disk(f"Code/Train/SFT/datasets/{dataset_name}_train")
testset.save_to_disk(f"Code/Train/SFT/datasets/{dataset_name}_test")
