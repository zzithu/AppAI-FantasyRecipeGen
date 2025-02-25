from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
import torch

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

# Force GPU mode
device = torch.device("cuda")

# Load tokenizers and models
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-large", torch_dtype=torch.float16)

# Move models to GPU
bert_model.to(device)
t5_model.to(device)

# Encode input with BERT
input_text = "Hello, how are you?"
inputs = bert_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    encoder_outputs = bert_model(**inputs)

# Convert encoder outputs to float32
encoder_outputs = encoder_outputs.last_hidden_state.to(torch.float32)

# Prepare decoder input (e.g., just a starting token)
decoder_input_ids = t5_tokenizer.encode("translate English to French:", return_tensors="pt").to(device)

# Decode with T5
outputs = t5_model.generate(
    input_ids=decoder_input_ids,
    encoder_outputs=encoder_outputs, 
    max_length=50, 
    num_beams=5, 
    early_stopping=True
)

# Decode the generated tokens to text
result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
