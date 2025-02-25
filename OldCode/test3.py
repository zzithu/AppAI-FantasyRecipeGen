import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def generate_step_by_step_recipe(ingredients):
    """Generate a structured medieval fantasy recipe with a whimsical touch."""

    # Generate Title First
    title_prompt = f"Create a medieval fantasy-style name for a dish made with these ingredients: {ingredients}.\n\n**Title:**"
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

    # Initial Recipe Steps (coherent, non-whimsical)
    num_steps = random.randint(5, 8)  # Random number of recipe steps
    recipe_steps = []

    # Basic Recipe Generation
    for i in range(1, num_steps + 1):
        step_prompt = (
            f"Create a clear, coherent cooking step using these ingredients: {ingredients}. "
            "The step should make sense in the context of a recipe, taking inspiration from common cooking techniques and practices. "
            "Ensure all ingredients are used and referenced logically, while maintaining culinary coherence. "
            "The step should be concise, and describe a common action (e.g., mixing, heating, pouring) while avoiding excessive repetition. "
            "If some ingredients seem unusual together, incorporate them using familiar techniques or flavor combinations that could make sense in a real recipe. "
            "The step should also have a clear sense of progression, fitting naturally into the flow of a recipe."
        )


        step_inputs = tokenizer(step_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        step_output = model.generate(
            **step_inputs,
            max_length=512,
            temperature=0.5,
            top_p=0.7,
            repetition_penalty=1.5,
            do_sample=True
        )
        new_step = tokenizer.decode(step_output[0], skip_special_tokens=True)
        recipe_steps.append(f"{i}. {new_step}")

    # Random whimsy depth (1 to 3 recursions)
    whimsy_depth = random.randint(1, 3)
    
    # Add whimsy recursively
    for i in range(whimsy_depth):
        new_recipe_steps = []
        for j, step in enumerate(recipe_steps):
            whimsical_prompt = (
                f"Enhance this cooking step with whimsical, magical details. "
                "Imagine this as a mystical recipe, part of an ancient tome or magical ritual. Keep the core instruction intact, "
                "but add vivid imagery: describe magical elements, enchanted surroundings, or fantastical beings that make the recipe come alive. "
                f"Here is the instruction to enhance: {step}"
            )


            whimsical_inputs = tokenizer(whimsical_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            whimsical_output = model.generate(
                **whimsical_inputs,
                max_length=512,
                temperature=0.75,
                top_p=0.85,
                repetition_penalty=1.2,
                do_sample=True
            )
            whimsical_step = tokenizer.decode(whimsical_output[0], skip_special_tokens=True)
            new_recipe_steps.append(f"{j+1}. {whimsical_step}")
        recipe_steps = new_recipe_steps

    # Combine everything into the final recipe
    final_recipe = f"**Title:** {title}\n\n**Ingredients:** {ingredients}\n\n**Instructions:**\n" + "\n".join(recipe_steps)
    return final_recipe

# Test run
print(generate_step_by_step_recipe("sorghum legume corn flour coconut oil sugar basil barley sugar"))
