import random
from prompt_generation import generate_step_prompt

def generate_recipe(ingredients, difficulty="medium", dish_type="savory"):
    print(f"Generating recipe with ingredients: {ingredients}")
    
    # Generate steps using the step prompt logic
    steps = []
    for _ in range(5):  # Generate 5 steps for the recipe
        step = generate_step_prompt(ingredients, difficulty, dish_type)
        steps.append(step)
    
    # Generate a random recipe title
    title = f"Recipe: {random.choice(['The Secret', 'A Dash of', 'Mystic', 'Grandmothers'])} {random.choice(['Soup', 'Stew', 'Concoction', 'Delight'])}"
    
    print(f"Title: {title}")
    print("\nIngredients:", ingredients)
    print("\nInstructions:")
    for idx, step in enumerate(steps, 1):
        print(f"{idx}. {step}")

# Example of running the recipe generator
ingredients = ['sorghum', 'legume', 'corn flour', 'coconut oil', 'sugar', 'basil', 'barley']
generate_recipe(ingredients)
