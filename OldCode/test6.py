from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

# Force GPU mode
device = torch.device("cuda")

# Load the model and tokenizer
model_name = "deepseek-ai/deepseek-r1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Force FP16 instead of FP8
    device_map="auto",
    trust_remote_code=True
)


# Define a test prompt
prompt = "Once upon a time in a mystical land, a brave warrior set out on a journey to"

# Tokenize input and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(
    **inputs, 
    max_length=100, 
    temperature=0.7, 
    top_p=0.9, 
    do_sample=True
)

# Decode and print result
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
