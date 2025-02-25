import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def generate_step_by_step_recipe(ingredients):
    """Generate a structured medieval fantasy recipe, then enhance it with whimsical details."""
    
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
    
    # Start the Recipe Instructions
    recipe_steps = []
    num_steps = random.randint(3, 6)  # Randomly choose number of steps to keep variation

    # Initial context
    current_recipe = f"**Title:** {title}\n\n**Ingredients:** {ingredients}\n\n**Instructions:**\n"

    # Adjusting generation parameters and improving prompts

    # Step generation: ensure clear and logically structured steps
    for i in range(1, num_steps + 1):
        step_prompt = (
            f"Create a unique, whimsical cooking step using these ingredients: {ingredients}. "
            "Ensure this step is different from the others, avoiding repetition or redundancy, and adding a touch of medieval fantasy. "
            "The language should feel like part of a magical potion-making process, and each step should use these ingredients in a creative, distinct way."
        )

        step_inputs = tokenizer(step_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        step_output = model.generate(
            **step_inputs,
            max_length=512,
            temperature=0.5,  # Lowered temperature for more coherent results
            top_p=0.7,  # Reduced randomness
            repetition_penalty=1.5,  # Increased penalty to reduce repetition
            do_sample=True
        )
        new_step = tokenizer.decode(step_output[0], skip_special_tokens=True)

        # Ensure the generated step is clear and relevant
        if "draw a circle" in new_step or new_step in recipe_steps:
            continue  # Ignore abstract or repetitive steps
        
        recipe_steps.append(f"{i}. {new_step}")
        current_recipe = f"**Ingredients:** {ingredients}\n\n**Instructions:**\n" + "\n".join(recipe_steps)

    # After this, you can apply whimsical enhancements as previously planned

    # Second round: Enhance with whimsical details
    for i in range(len(recipe_steps)):
        whimsical_prompt = (
            f"{current_recipe}\nStep {i+1}: Enhance this step with whimsical, magical details, using medieval fantasy-inspired language. "
            "Make the step feel like part of a magical ritual or an ancient potion-making process, while keeping the core actions intact."
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
        
        # Update the recipe with the whimsical step
        recipe_steps[i] = f"{i+1}. {whimsical_step}"
        current_recipe = f"**Title:** {title}\n\n**Ingredients:**\n\n**Instructions:**\n" + "\n".join(recipe_steps)

    # Add final medieval-style presentation step
    final_prompt = (
        f"{current_recipe}\nStep {num_steps+1}: In the grand tradition of medieval feasts, describe how this dish should be presented. "
        "Draw upon the imagery of ancient banquets, enchanted gatherings, or rustic taverns. "
        "Describe the setting—whether it's placed upon a sturdy oak table under a canopy of twinkling lights, or served in the dimly lit chambers of a sorcerer’s lair. "
        "Invoke vivid imagery to make the dish come alive as if it were part of an epic tale."
    )

    final_inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    final_output = model.generate(
        **final_inputs,
        max_length=1024,
        temperature=0.6,
        top_p=0.8,
        repetition_penalty=1.2,
        do_sample=True
    )
    
    final_step = tokenizer.decode(final_output[0], skip_special_tokens=True)
    recipe_steps.append(f"{num_steps+1}. {final_step}")
    
    return f"**Title:** {title}\n\n**Ingredients:** {ingredients}\n\n**Instructions:**\n" + "\n".join(recipe_steps)

# Test run
print(generate_step_by_step_recipe("garlic, honey, and mint"))
print("done")
