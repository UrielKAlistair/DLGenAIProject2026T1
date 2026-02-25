#Stage 1 

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