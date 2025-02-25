import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google-t5/t5-3b"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def generate_t5_recipe(ingredients):
    """Generate a medieval fantasy recipe without a title first."""
    recipe_prompt =     f"""
    TASK: GENERATE A RECIPE USING ONLY A SELECTION OF THE FOLLOWING INGREDIENTS: {ingredients}. THEY DO NOT ALL HAVE TO BE USED. 
    GENERATE SEVERAL CANDIDATE RECIPES, THEN EVALUATE WHICH MAKES THE MOST SENSE FLAVOR-WISE AND ONLY USE THE BEST.
    PROVIDE ALL CANDIDATE RECIPES AS WELL AS AN EVALUATION OF EACH.
    Output the result in English.

    CONTEXT: YOU ARE A WHIMSICAL RECIPE BOT. YOU ARE IN A HIGH FANTASY WORLD. WHEN CREATING RECIPES, KEEP THIS IN MIND AND GENERATE THEM AS IF THEY ARE FANTASTICAL DISHES. WHEN CREATING INSTRUCTIONS FOR THE USER, MAKE IT FANTASTICAL AND USE MORE UNIQUE WORDING.
    STEP 1: CREATE A RECIPE USING THE PROVIDED INGREDIENTS
    STEP 2: NAME THE RECIPE AND BREAK THE COOKING INSTRUCTIONS INTO DIGESTIBLE STEPS
    STEP 3: PRESENT THE NAME AND STEPS TO THE USER IN A CONVENTIONAL RECIPE FORMAT

    """

    inputs = tokenizer(recipe_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    recipe_output = model.generate(
        **inputs,
        max_length=800,
        min_length=300,
        temperature=0.6,
        top_p=0.8,
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
