from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



class Model(nn.Module):
    def __init__(self, model_name, revision):
        super(Model, self).__init__()
        self.model_name = model_name
        self.revision= revision



    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Make decoder-only generation happy
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            dtype=torch.float16,  # or torch.bfloat16 if your GPU supports it
        )
        self.model.to("cuda")


    def classify_batch(self, messages, ans0s, ans1s, ans2s, max_new_tokens=4):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        results = []
        for i in range(len(messages)):
            # length of non-padding part of input_ids
            input_len_i = (inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum()
            gen_ids = outputs[i][input_len_i:]
            raw_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            first_char = raw_output[0].upper() if raw_output else ""

            results.append(first_char)

        return results

