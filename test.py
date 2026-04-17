from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./tinyLLM",
    tokenizer="./tinyLLM",
)

prompt = "User: What is yorr name?\nAssistant:"
out = pipe(prompt, max_new_tokens=30, do_sample=False)

print(out[0]["generated_text"])