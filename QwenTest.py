#Relating to model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

# Check device availability >> Want GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#I am going to time the whole operation till just before it prints
start_time = time.time()

# Model preparation
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
mistral_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

def transform_recipe_to_fantasy(output_json):
    # Create fantasy-style prompt
    fantasy_model_prompt = f"""
        Convert the following recipe into a human-readable medieval fantasy recipe. Do not include this in your output, instead focus on creating the recipe based on this prompt:

        Context: You are a medieval recipe writer in a fantasy world. Use elaborate descriptions, poetic phrasing, and a rich medieval tone, akin to an ancient grimoire or a bard’s tale. The ingredients should be described in whimsical detail, but the focus should be on creating vivid and actionable cooking instructions, written in a step-by-step manner that can be easily followed by a cook in this world.

        Here is the recipe in a pseudo-JSON-like format (use it for reference, but don’t include it in your output):
        {output_json}

        Now, create the recipe in the following format:

        **TITLE**  
        (Title Here, Be creative)  

        **INGREDIENTS**  
        (List Here with concise whimsical descriptions)  

        **INSTRUCTIONS**  
        (Instructions here, step-by-step, ensuring clarity and vivid detail to guide the cook.)
        """



    # Tokenize input >> Qwen heavily favors 512 tokens it seems
    inputs = tokenizer(fantasy_model_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate fantasy-styled recipe
    with torch.no_grad():
        fantasy_recipe_output = mistral_model.generate(
            **inputs,
            max_length=800,  # Adjust based on response length
            temperature=0.7,  # Balances randomness and coherence
            top_p=0.8,  # Focus on higher probability tokens
            repetition_penalty=1.2,  # Reduce repetitive outputs
            do_sample=True  # Enable sampling
        )

    # Decode the output
    fantasy_recipe = tokenizer.decode(fantasy_recipe_output[0], skip_special_tokens=True)

    return fantasy_recipe.strip()

# Example usage
output_json = {  # Replace with the actual output from the previous model
    "title": "veggie and nut mushrooms",
    "ingredients": ["1 1/2 cups water", "1/2 cup mackerel", "1/2 cup brown rice", "1 cup cashew nuts", "1 cup oats", "1/2 cup tomato sauce", "1 teaspoon cayenne pepper"],
    "method": ["bring the water to a boil in a medium saucepan...", "soak the cashews overnight...", "preheat the oven to 180°C..."]
}

fantasy_recipe = transform_recipe_to_fantasy(output_json)

elapsed_time = time.time() - start_time
print(f"{fantasy_recipe}\nElapsed Time: {elapsed_time}")

