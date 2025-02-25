import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def generate_t5_recipe(ingredients):
    """Generate a medieval fantasy recipe without a title first."""
    recipe_prompt = (
        f"Create a medieval fantasy-style recipe using these ingredients: {ingredients}.\n\n"
        "**Ingredients:**\n"
        "- List each ingredient once, exactly as provided, no extra repetition.\n\n"
        "**Instructions:**\n"
        "1. Describe the first step clearly and vividly.\n"
        "2. Continue with detailed, fantasy-inspired steps.\n"
        "3. Ensure all ingredients are used logically in the recipe.\n"
        "4. Conclude with a medieval-style presentation suggestion.\n\n"
        "**Do NOT repeat ingredients multiple times. Do NOT skip instructions. Provide a full recipe.**"
    )

    inputs = tokenizer(recipe_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    recipe_output = model.generate(
        **inputs,
        max_length=800,
        min_length=300,
        temperature=0.65,
        top_p=0.9,
        repetition_penalty=1.3,
        do_sample=True
    )

    recipe = tokenizer.decode(recipe_output[0], skip_special_tokens=True)

    # STEP 2: Generate a title based on the recipe
    title_prompt = (
        "Create a medieval fantasy-style name for this dish:\n\n"
        f"{recipe}\n\n"
        "**Title:**"
    )

    title_inputs = tokenizer(title_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    title_output = model.generate(
        **title_inputs,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )

    title = tokenizer.decode(title_output[0], skip_special_tokens=True)

    return f"**Title:** {title}\n\n{recipe}"

# Test run
print(generate_t5_recipe("garlic, honey, and mint"))
print("done")
