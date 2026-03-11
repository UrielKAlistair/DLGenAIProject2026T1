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

```
python -m DnG.cnn.train_cnn \
    --dataset-root DnG/jan-2026-dl-gen-ai-project/messy_mashup \
    --output-root DnG/cnn/outputs \
    --run-name cnn_v1
```

and inference with

```
python -m DnG.cnn.infer_cnn \
    --dataset-root DnG/jan-2026-dl-gen-ai-project/messy_mashup \
    --model-path DnG/cnn/outputs/cnn_v1/model.pt \
    --summary-path DnG/cnn/outputs/cnn_v1/summary.json \
    --output-path DnG/cnn/outputs/cnn_v1/submission.csv
```

# Stage 3

Proper synthetic data generation pipeline courtesy of https://github.com/Photon-08/milestone-4-codebase/tree/main. Now noise is properly factored in. Now all prior pipelines get a huge bump due to significantly better training data.

CRNN code has been merged into the CNN pipeline and common utilites got extracted out and dropped in common. Code is a lot more structured. To recreate my run, use:

```
python -m DnG.train_spectrogram \
    --model crnn \
    --dataset-root DnG/jan-2026-dl-gen-ai-project/messy_mashup \
    --output-root DnG/crnn/outputs \
    --run-name crnn_v1
```
