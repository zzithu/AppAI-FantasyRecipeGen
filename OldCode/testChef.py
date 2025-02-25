import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
MODEL_ID = "auhide/chef-gpt-en"  # Ensure this model exists or replace with another
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
chef_gpt = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)  # Move model to GPU

def generate_chef_gpt_recipe(ingredients):
    recipe_prompt = (
        "Without a title for this recipe, "
        f"Create a cooking or baking recipe using these ingredients: {ingredients}.\n\n"
        "Follow the format of:\n\n"
        "**INGREDIENTS**\n(List Here)\n\n"
        "**INSTRUCTIONS**\n(Instructions here)\n\n"
        "Ensure instructions follow a step-by-step procedure and can be replicated.\n"
    )

    inputs = tokenizer(recipe_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():  # Disable gradient calculations for inference
        recipe_output = chef_gpt.generate(
            **inputs,
            max_length=800,
            min_length=300,
            temperature=0.65,
            top_p=0.9,
            repetition_penalty=1.3,
            do_sample=True
        )

    recipe = tokenizer.decode(recipe_output[0], skip_special_tokens=True)

    # STEP 2: Generate a title for the recipe
    title_prompt = (
        "Create a medieval fantasy-style name for this dish:\n\n"
        f"{recipe}\n\n"
        "**Title:**"
    )

    title_inputs = tokenizer(title_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        title_output = chef_gpt.generate(
            **title_inputs,
            max_length=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )

    title = tokenizer.decode(title_output[0], skip_special_tokens=True)

    return f"**Title:** {title.strip()}\n\n{recipe.strip()}"

# Test run
print(generate_chef_gpt_recipe("garlic, honey, and mint"))
print("done")
