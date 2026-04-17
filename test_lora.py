from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./tinyLLM_lora",
    tokenizer="./tinyLLM_lora",
)

prompt = "User: Who is the current president of the Philippines?\nAssistant:"
out = pipe(prompt, max_new_tokens=30, do_sample=True)

print(out[0]["generated_text"])