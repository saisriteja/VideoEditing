from glob import glob

import os
import fiftyone as fo
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from logzero import logger


DATASET_DIR = os.path.join(os.getcwd(),"weather_aug/")

try:
    dataset = fo.load_dataset("weather-cell-images")

    logger.debug("Dataset already exists")
except:
    dataset = fo.Dataset.from_dir(
        DATASET_DIR,
        fo.types.ImageClassificationDirectoryTree,
        name="weather-cell-images",
    )
    dataset.persistent = True
    logger.debug("Dataset created")



# vit embeddings
model = foz.load_zoo_model("clip-vit-base32-torch")
# option 1
embeddings = dataset.compute_embeddings(model)
# results = fob.compute_visualization(dataset, embeddings=embeddings)
dataset.save()



# Compute 2D representation
results = fob.compute_visualization(
    dataset,
    embeddings=embeddings,
    num_dims=2,
    method="umap",
    brain_key="weather",
    verbose=True,
    seed=51,
)


session = fo.launch_app(dataset)
# save the embeddings

# compute clip

while True:
    pass



