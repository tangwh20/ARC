import os
import re
import json

from openai import AzureOpenAI
with open(os.path.join(os.path.dirname(__file__), "openai.json")) as f:
    openai_config = json.load(f)
os.environ["OPENAI_API_VERSION"] = openai_config["OPENAI_API_VERSION"]
os.environ["AZURE_OPENAI_API_KEY"] = openai_config["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = openai_config["AZURE_OPENAI_ENDPOINT"]


def get_seeds_contents():
    # get all files in seeds directory
    # get current directory path
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    seeds = os.listdir(os.path.join(current_file_dir, "seeds"))
    # filter files with .py extension and 8 hex value characters in the file name
    pattern = r"[0-9a-f]{8}(_[a-zA-Z]+)?\.py"
    # get all files and its content
    seeds = [seed for seed in seeds if re.match(pattern, seed)]
    seeds_contents = []
    for seed in seeds:
        with open(os.path.join(current_file_dir, "seeds", seed)) as f:
            content = f.read()
            content = content.split("# ============= remove below this point for prompting =============")[0].strip()
            seeds_contents.append((seed, content))

    # print all files
    print(f"Using the following {len(seeds)} seeds:", ", ".join(seeds).replace(".py", ""))

    return seeds_contents


def get_concepts_from_content(content: str):
    concepts = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "# concepts:" in line:
            in_line_concepts = lines[i][12:]
            if in_line_concepts.strip() != "":
                concepts.extend(lines[i][12:].split(","))
            while i+1 < len(lines) and lines[i+1].startswith("# ") and not lines[i+1].startswith("# description:"):
                concepts.extend(lines[i+1][2:].split(","))
                i += 1
            concepts = [c.strip() for c in concepts]
            break
    return concepts


def get_description_from_content(content: str):
    description = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "# description:" in line:
            while i+1 < len(lines) and lines[i+1].startswith("# "):
                description.append(lines[i+1][2:])
                i += 1
            description = " ".join(description)
            break
    return description if description else "No description found."


def get_prompt(num_generations: int = 5):
    seeds_contents = get_seeds_contents()
    examples = []
    for seed, content in seeds_contents:
        concepts = get_concepts_from_content(content)
        description = get_description_from_content(content)
        example = f"```python\n# concepts:\n# {', '.join(concepts)}\n\n# description:\n# {description}\n```"
        examples.append(example)
    
    # read the prompt template from prompts/description_prompt.md
    template_path = os.path.join(os.path.dirname(__file__), "templates", "description_prompt.md")
    with open(template_path) as f:
        prompt_template = f.read()

    prompt = prompt_template.format(examples=examples, num_generations=num_generations)
    return prompt



if __name__ == "__main__":
    import time

    model_name = "gpt-4.1-mini"
    price = (0.4, 1.6)  # (prompt, completion) in USD per 1M tokens
    chat = AzureOpenAI()

    prompt = get_prompt(num_generations=5)
    messages = [
        {"role": "system", "content": "You are a puzzle maker designing geometric, physical, and topological puzzles for curious middle-schoolers. You are creative, playful, and love to explore new ideas."},
        {"role": "user", "content": prompt},
    ]
    completion = chat.chat.completions.create(model=model_name, messages=messages)
    response = completion.choices[0].message.content
    print(response)

    # Save the response to a file
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"text_{timestamp}.md")
    with open(output_file, "w") as f:
        f.write(response)
    print(f"Response saved to {output_file}")

    # Log the cost
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    total_tokens = prompt_tokens + completion_tokens
    cost = (prompt_tokens / 1_000_000) * price[0] + (completion_tokens / 1_000_000) * price[1]
    print(f"Total tokens: {total_tokens}, Cost: ${cost:.6f} (Prompt: {prompt_tokens}, Completion: {completion_tokens})")
    print(f"Model: {model_name}, Price per 1M tokens: ${price[0]:.6f} (prompt), ${price[1]:.6f} (completion)")
    
    breakpoint()
    print("Done!")