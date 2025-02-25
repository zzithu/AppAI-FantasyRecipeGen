from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

# Force GPU mode
device = torch.device("cuda")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoder = AutoModelForCausalLM.from_pretrained(
    "bert-base-uncased",  # Correct the model name here
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Move model to GPU
encoder.to(device)

# Prepare inputs
inputs = tokenizer("Hello, how are you?", return_tensors="pt")  # Change 'tf' to 'pt' for PyTorch
inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU

# Perform the forward pass
with torch.no_grad():
    outputs = encoder(**inputs)


decoder = AutoModelForCausalLM.from_pretrained(
    "t5-large",  # Correct the model name here
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Extract logits and use greedy decoding
logits = outputs.logits
predicted_token_ids = torch.argmax(logits, dim=-1)

# Decode the predicted token IDs
result = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
print(result)
