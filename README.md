# Generating South Park Episodes

<center>
	<img src="https://media3.giphy.com/media/l3vR3mTSiTpV082Q0/giphy.gif?cid=790b761153590e114eebb6d2e2b78ab2cada04abecd7af1d&rid=giphy.gif"/>
</center>

---

## Summary

I've fine-tuned the [GPT2](https://huggingface.co/gpt2) model on South Park episodes. The library that was used to train the model was the [**huggingface transformers**](https://huggingface.co/) library and the dataset was gathered from this [Kaggle dataset](https://www.kaggle.com/datasets/mustafacicek/south-park-scripts-dataset).

---

## Modules

The Repo contains 4 modules:

- [Data Preporcessing](./DataPreprocessing/csv_to_dict.ipynb): Contains the code that preprocesses the dataset and creates the `SouthPark_Data_test.pkl` and `SouthPark_Data_train.pkl` files.
- [Train](./AI/train.py): This module contains the code that trains the model.
- [Testing](./AI/test.py): Computes the Rouge-1, Rouge-2, and Rouge-L scores for the test set.
- [Inference](./AI/inference.py): This module is used to generate episodes.

---

## Poster

![Poster](poster.png)
