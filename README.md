# Generating South Park Episodes

<div align="center">
	<img src="https://media3.giphy.com/media/l3vR3mTSiTpV082Q0/giphy.gif?cid=790b761153590e114eebb6d2e2b78ab2cada04abecd7af1d&rid=giphy.gif"/>
</div>

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

## Models

The models can be found [here](https://drive.google.com/drive/folders/1e2taNKJrRZ5_0ae_edxzaltnEQgN2lJd?usp=sharing). Once downloaded, the desired model folders have to be put in the */AI* folder. When calling the *get_model* function, the name should be specified in the *checkpoint* parameter.

---

## Poster

![Poster](poster.png)
