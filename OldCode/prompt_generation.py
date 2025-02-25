import random

# Helper function to determine flavor profile based on ingredients
def determine_flavor_profile(ingredients):
    flavors = []
    if 'sugar' in ingredients or 'coconut oil' in ingredients:
        flavors.append('sweet')
    if 'garlic' in ingredients or 'basil' in ingredients:
        flavors.append('savory')
    if 'spice' in ingredients:
        flavors.append('spicy')
    return ", ".join(flavors) if flavors else 'neutral'

def generate_step_prompt(ingredients, difficulty="medium", dish_type="savory"):
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
