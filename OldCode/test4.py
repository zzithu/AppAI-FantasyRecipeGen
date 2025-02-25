import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def determine_flavor_profile(ingredients):
    """Determine the flavor profile based on the ingredients."""
    flavors = []
    if 'sugar' in ingredients or 'coconut oil' in ingredients:
        flavors.append('sweet')
    if 'garlic' in ingredients or 'basil' in ingredients:
        flavors.append('savory')
    if 'spice' in ingredients:
        flavors.append('spicy')
    return ", ".join(flavors) if flavors else 'neutral'

def generate_step_prompt(ingredients, difficulty="medium", dish_type="savory"):
    """Generate the step prompt with an adaptive structure."""
    flavor_profile = determine_flavor_profile(ingredients)
    
    # Base step structure
    base_prompt = (
        f"Create a clear, coherent cooking step using these ingredients: {ingredients}. "
        "The step should make sense in the context of a recipe, inspired by common cooking techniques and practices. "
        "Instead of relying solely on the dish's title, derive the overall flavor profile based on the ingredients. "
        f"The dish will have a {flavor_profile} flavor. Consider how these flavors should combine and influence the final dish. "
        "The recipe should flow logically, building on each step to enhance or balance the derived flavors. "
        "Ensure that the ingredients work in harmony and that each step brings out complementary flavors."
    )
    
    # Add difficulty level to the prompt
    if difficulty == "easy":
        base_prompt += " Keep the recipe simple, focusing on a few basic techniques and short steps."
    elif difficulty == "hard":
        base_prompt += " Incorporate advanced techniques and more complex flavors, making sure to balance them effectively."
    
    # Add dish type considerations
    if dish_type == "savory":
        base_prompt += " Focus on savory flavors and consider techniques like browning or reducing liquids."
    elif dish_type == "sweet":
        base_prompt += " Focus on balancing sweetness and texture, incorporating techniques like folding or caramelizing."
    
    return base_prompt

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

    # Generate Recipe Steps (coherent and flavor-driven)
    num_steps = random.randint(5, 8)  # Random number of recipe steps
    recipe_steps = []

    # Create recipe steps based on derived flavor and structure
    for i in range(1, num_steps + 1):
        step_prompt = generate_step_prompt(ingredients, difficulty="medium", dish_type="savory")

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
ingredients = ["sorghum", "legume", "corn flour", "coconut oil", "sugar", "basil", "barley"]
print(generate_step_by_step_recipe(ingredients))
