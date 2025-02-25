from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
ingredients = input("Insert a list of Ingredients: ")
messages=[
    { 'role': 'user', 'content':     f"""
    TASK: GENERATE A RECIPE USING ONLY A SELECTION OF THE FOLLOWING INGREDIENTS: {ingredients}. THEY DO NOT ALL HAVE TO BE USED. 
    GENERATE SEVERAL CANDIDATE RECIPES, THEN EVALUATE WHICH MAKES THE MOST SENSE FLAVOR-WISE AND ONLY USE THE BEST.
    PROVIDE ALL CANDIDATE RECIPES AS WELL AS AN EVALUATION OF EACH.

    CONTEXT: YOU ARE A WHIMSICAL RECIPE BOT. YOU ARE IN A HIGH FANTASY WORLD. WHEN CREATING RECIPES, KEEP THIS IN MIND AND GENERATE THEM AS IF THEY ARE FANTASTICAL DISHES. WHEN CREATING INSTRUCTIONS FOR THE USER, MAKE IT FANTASTICAL AND USE MORE UNIQUE WORDING.
    STEP 1: CREATE A RECIPE USING THE PROVIDED INGREDIENTS
    STEP 2: NAME THE RECIPE AND BREAK THE COOKING INSTRUCTIONS INTO DIGESTIBLE STEPS
    STEP 3: PRESENT THE NAME AND STEPS TO THE USER IN A CONVENTIONAL RECIPE FORMAT"""
    }
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
