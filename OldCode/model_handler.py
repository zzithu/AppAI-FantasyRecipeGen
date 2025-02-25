# model_handler.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Handler:
    def __init__(self, model_name="gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate_recipe(self, input_text, max_length=100):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
