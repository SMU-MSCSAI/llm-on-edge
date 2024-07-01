import gc
from transformers import AutoModel, AutoTokenizer
from transformers import logging, WEIGHTS_NAME, CONFIG_NAME
from shutil import rmtree
import os

# Suppress unnecessary warnings
logging.set_verbosity_error()

# Load models as an example
model_name = "google/gemma-2-9b"
model = AutoModel.from_pretrained("./models/gemma-2B")
tokenizer = AutoTokenizer.from_pretrained("./models/gemma-2B")

# Your operations with the model...

# Cleanup step
def cleanup_model(model):
    # Delete the model object
    del model
    # Clear cache
    
    # Manually construct the cache directory path
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    for dir_name in [model_name, f"{model_name}/{WEIGHTS_NAME}", f"{model_name}/{CONFIG_NAME}"]:
        dir_path = os.path.join(cache_dir, dir_name)
        if os.path.exists(dir_path):
            rmtree(dir_path)

    # Explicitly call the garbage collector
    gc.collect()

# Cleanup the model
cleanup_model(model)
cleanup_model(tokenizer)
