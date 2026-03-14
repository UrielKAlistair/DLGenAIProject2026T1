# Stage 1 

EDA was performed. Not too much to see. I listened to the test audio and it looks like noise randomly gets added in the middle, and the sample itself is 30s of a mix of random songs.
Had to define a lot of utility functions to simply organise a dictionary of all training data and make a synthetic data generator to synthesise data of the shape of the test data. I did not add any noise yet.
Trained a basic catBoost Baseline.  I ran using the following command from the parent directory:
```
python -m DnG.mmfc_trees.catboost_baseline \
    --dataset-root DnG/jan-2026-dl-gen-ai-project/messy_mashup \
    --output-root DnG/mmfc_trees/outputs \
    --run-name mfcc_catboost_v1
```

and inference with

```
python -m DnG.mmfc_trees.infer_test \
    --dataset-root DnG/jan-2026-dl-gen-ai-project/messy_mashup \
    --output-path DnG/mmfc_trees/submission.csv \
    --model-path DnG/mmfc_trees/outputs/mfcc_catboost_v1/model.pkl
```

Kaggle public score ***0.52913***

# Stage 2

Scratch CNN on log-mel spectrograms, still trained on the same synthetic same-genre mashups. Migrated audio resampling to torchaudio to gani GPU speedup. Tried out a very small CNN just as a proof of concept for upcoming models.
***the code snippets here have been removed as they are no longer compatible with the latest version of the repo. See An older commit if you wish to recreate the basic CNN run, but a better base CNN is avaialbe after the refactor in Stage 3.***

# Stage 3

Proper synthetic data generation pipeline courtesy of https://github.com/Photon-08/milestone-4-codebase/tree/main. Now noise is properly factored in. Now all prior pipelines get a huge bump due to significantly better training data.

CRNN code has been merged into the CNN pipeline and common utilites got extracted out and dropped in common. Code is a lot more structured. For all 3 models: CNN, CRNN, EfficientNET, use the following command to start a run:

```
python -m DnG.CNNs.train \
--dataset-root DnG/messy_mashup \
--output-root DnG/CNNs/outputs \
--model cnn \
--run-name cnn_comparison
```

# Stage 4 

EfficientNET uses the vast majority of preexisting code without major changes. Until This point, I had not seriously bothered tuning hyperparameters, as I dont expect a vanilla CNN or CRNN to outperform EfficientNET and I wanted to save my GPU time for tuning the best possible model. 

This commit merely introduces EfficientNET with some params and checks if it runs