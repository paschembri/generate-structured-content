# Generate dynamic structured content for your apps

Sample code from the article [How to generate structured content with OpenAI API ?](https://blog.sodium.cl/tech-tutorials/generating-structured-content-with-openai-api.html)

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create a vars.env file containing 'export OPENAI_API_KEY="..."'
source vars.env

python3 todo-app-categories.py

```

`todo-app-categories.py` generates and print a combination of task categories according to user profiles.

