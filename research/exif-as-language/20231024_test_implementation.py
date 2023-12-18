# %%
import os

from photoholmes.models.exif_as_language.clip import ClipModel

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
model = ClipModel("resnet50", "distilbert", "mean")

# %%
from transformers.models.distilbert import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# %%
test = tokenizer(
    ["This is a test!", "This is another a test!"], return_tensors="pt", padding=True
)
# %%
model.encode_text(test).shape
# %%
model = ClipModel("resnet50", "distilbert", "cls")
# %%
model.encode_text(test).shape
# %%
