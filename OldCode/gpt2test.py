import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate recipes
def generate_recipe(ingredients):
    prompt = f"""I have {ingredients}. 
Make me a whimsical medieval-style recipe. 

Ignore results that are real. Replace words like 'Bake' with 'Cast Inferno', and other instructions to match medieval and fantasy-styled recipes. 

Draw inspiration from:
- Shakespearian works  
- The Elder Scrolls  
- The Hobbit  
- Lord of the Rings  
- Other similar fantasy works."""

    # Encode prompt (ensure it doesn't exceed GPT-2's token limit)
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=800)

    # Generate text
    output = model.generate(
        inputs, 
        max_length=250,  # Keep output reasonable
        temperature=0.7, 
        repetition_penalty=1.2, 
        top_p=0.9, 
        do_sample=True
    )

    # Decode and return output
    recipe = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Post-process (replace words manually for more control)
    recipe = recipe.replace("Bake", "Cast Inferno").replace("Mix", "Weave Arcane Essence")

    return recipe

# Gradio UI
demo = gr.Interface(
    fn=generate_recipe,  # Function to call
    inputs="text",       # User enters text
    outputs="text"       # Returns generated recipe
)

# Run it
demo.launch()
