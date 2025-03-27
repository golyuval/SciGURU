import datasets
from datasets import load_dataset

dataset = load_dataset("pubmed_qa", "pqa_labeled")


dataset = dataset["train"].train_test_split(
        test_size=0.2,
    	seed=42)

trainset = dataset.get("train")
testset = dataset.get("test")

trainset.save_to_disk("finetuning/datasets/pubmedqa_train")
testset.save_to_disk("finetuning/datasets/pubmedqa_test")
trainset.to_csv("train.csv")
testset.to_csv("test.csv")
