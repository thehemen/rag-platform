# Telegram ML Competition 2023

This project is implemented for the [ML Competition 2023](https://contest.com/docs/ML-Competition-2023) which task is to create a library that detects a programming or markup language of a code snippet.

## How to Create Dataset

The dataset consists of ~8k code samples and ~2k user questions.

The code samples are downloaded from GitHub by using [GitHub REST API](https://docs.github.com/rest). Make sure you have your own access token to use GitHub API.

The user questions are taken from StackOverflow by using [StackAPI](https://pypi.org/project/StackAPI). They contain the source code samples to consider the difference between user messages and code samples.

To have the fair distribution of programming languages, [the GitHub Language Stats](https://madnight.github.io/githut/#/pull_requests/2023/3) are used here. In addition, some programming and markup language numbers are manually increased to use them in the dataset as well.

As the dataset is split into train and test subsets, every language must have at least two samples. So, [the Dirichlet-Rescale algorithm](https://pypi.org/project/drs) is used to generate numbers that meet min-max and sum constraints.

First of all, you need to install all the dependencies required to create a dataset:

```sh
pip install -r requirements.txt
```

Then, you should download GitHub code samples (it takes ~1.5 hours):

```sh
python3 extract_github.py
```

After that, you need to download StackOverflow user questions (it takes ~1 min):

```sh
python3 extract_stack_overflow.py
```

Finally, you can create the dataset:

```sh
python3 create_dataset.py
```

## How to Train Model

The Facebook's [fastText](https://github.com/facebookresearch/fastText) library is used here. To build it, please follow the official instructions.

To train the model, run this command:

```sh
./fasttext supervised -input train.txt -output model
```

To test the model, run this command:

```sh
./fasttext test model.bin test.txt k
```

It outputs the precision and recall at k (P@k and R@k) on a test set.

## How to Build Library

To build the library, run:

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

To build the test script, run:

```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Results

|Name|Value|
|----|-----|
|Code/Other Detection|20.9|
|Language Detection|50.3|
|Average Time|190ms per sample|

The "code/other detection" score is low because the StackOverflow user questions differ from the real Telegram posts by its length and content.

Hope, my project may be useful to you.

Good luck!
