# tokenizer, load a pretrained one
from src.transformers.models.bert.tokenization_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from transformers.models.resbert.configuration_resbert import ResbertConfig

config = ResbertConfig(
    vocab_size = tokenizer.vocab_size,
    num_attention_heads=1,
    hidden_size=124,
    num_hidden_layers=6,
    reservoir_scaling_factor=2,
    reservoir_layers=[1, 2],
    eus_input_dim=124,
    type_vocab_size=1,
)

from transformers.models.resbert import ResbertForMaskedLM

model = ResbertForMaskedLM(config=config)

print(model)

print(model.num_parameters())

# usage of a MLM model

import torch

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))