import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("Ashikan/dut-recipe-generator")
model = AutoModelForCausalLM.from_pretrained("Ashikan/dut-recipe-generator")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

input_ingredients = ["cashew nut, apple, couscous, apple, bok choy, soy milk, baking soda, navy beans, turmeric, wasabi, pinto beans, coconut oil, portabello mushroom, orange, celery, sweet pepper, bean thread, black olive, tomato sauce, portabello mushroom, poppy seed, cayenne pepper, avocado, rhubarb, jam, coriander, plum tomato, cider, barbecue, alfalfa sprout, tomato, vegemite, chicory, soy sauce, couscous, sprout, pastry, tart, Goji berry, date sugar, dried leek, chipotle pepper, pecans, oats, tamale, mint, onion powder, wasabi, sauerkraut, citron"]

input_text = '{"prompt": ' + json.dumps(input_ingredients)

output = pipe(input_text, max_length=1024, temperature=0.2, do_sample=True, truncation=True)[0]["generated_text"]

#JSON formatted output with "title", "ingredients" and "method" nodes available
print(output)

#This works pretty well as a base! we're going to fantitize it

fantasy_model_prompt = f"""Without a title
                        Using this Json formatted recipe: {output} Convert it into human readable format.
                            Context: You are a medival recipe writer in a fantasy world. Use a broad assortment of old and fantasy language
                            while enhancing the imagery felt by the text.
                            "Enusre you follow the format of:\n\n"
                            "**INGREDIENTS**\n(List Here)\n\n"
                            "**INSTRUCTIONS**\n(Instructions here)\n\n"
                            "Ensure instructions follow a step-by-step procedure and can be replicated.\n"

                        """