# LangChain-Chatbot-with-Streamlit

This is a python script to implement a Chatbot which provide answers based on your own information.

## Installation

Install [Langchain](https://github.com/hwchase17/langchain) and other required packages.
```
pip install langchain openai chromadb tiktoken unstructured
```
Modify `Your OpenAI API key` inside `constants.py` to use your own [OpenAI API key](https://platform.openai.com/account/api-keys).

Place your own data into `data/data.txt`.

## Usage
Modify the `system_template` inside `chatgpt.py` based on your own need.

Run the script using command `streamlit run chatgpt.py`.

Enter local server `Running on http://127.0.0.1:5000` to access.

Enter your username before question for chat history tracking.

Test reading
```
> python chatgpt.py "Any easy recipes recommend?"
any easy recipes recommend?

Yes, here are a few easy recipes that you can try:

Chili con carne:
- Ingredients: Ground/minced beef, onion, garlic, tomatoes, tomato puree, chili powder, cumin, Worcestershire sauce, red pepper, kidney beans.
- Instructions: Brown the beef with onion and garlic, then add the remaining ingredients and simmer.

Simple Spaghetti Bolognese:
- Ingredients: Beef mince, onion, beef stock, garlic, carrots, mushrooms, dried herbs, tomato paste, passata or chopped tomatoes.
- Instructions: Cook the onion, then add the beef and remaining ingredients. Simmer until cooked through.
```
