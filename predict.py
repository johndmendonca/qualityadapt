from re import S
import numpy as np
from transformers import AutoModelWithHeads,AutoTokenizer
from transformers.adapters.composition import Fuse,Parallel

import torch

model_str='roberta-large'

special_token_dict = {
    "speaker1_token": "<speaker1>",
    "speaker2_token": "<speaker2>"
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_str)
tokenizer.add_tokens(list(special_token_dict.values()))

model = AutoModelWithHeads.from_pretrained(model_str)
model.resize_token_embeddings(len(tokenizer))

# Overall Quality prediction
#model.load_adapter("QualityAdapt/exp/"+model_str+"/Understandability",load_as="U", with_head=False)
#model.load_adapter("QualityAdapt/exp/"+model_str+"/Sensibleness",load_as="S", with_head=False)
#model.load_adapter_fusion("QualityAdapt/exp/"+model_str+"/Human/U,S")
#model.load_head("QualityAdapt/exp/"+model_str+"/Human/Human")

# Parallel subquality prediction
model.load_adapter("QualityAdapt/exp/"+model_str+"/Understandability",load_as="U", with_head=True)
model.load_adapter("QualityAdapt/exp/"+model_str+"/Sensibleness",load_as="S", with_head=True)
model.set_active_adapters(Parallel("U","S"))

model.to(device)

ctx = "<speaker1>Gosh , you took all the word right out of my mouth . Let's go out and get crazy tonight . <speaker2>Let's go to the new club on West Street ."
res = "<speaker1>I ' m afraid I can ' t ."

args = ((ctx, res))
result = tokenizer(*args, padding=True, max_length=124, truncation=True)

out = model(torch.tensor(result['input_ids']).to(device))

#overall_score = out.logits[0].detach().cpu().numpy()[0]
u_score = out[0].logits[0].detach().cpu().numpy()[0]
s_score = out[1].logits[0].detach().cpu().numpy()[0]

#print(overall_score)
print(u_score,s_score)