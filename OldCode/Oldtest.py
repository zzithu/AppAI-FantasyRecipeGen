# __main__.py
from model_handler import GPT2Handler

def main():
    gpt2 = GPT2Handler()
    input_text = "I have garlic, honey, and mint. Make me a recipe."
    recipe = gpt2.generate_recipe(input_text)
    print(recipe)

if __name__ == "__main__":
    main()
