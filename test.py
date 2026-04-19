from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./tinyLLM",
    tokenizer="./tinyLLM",
)

prompt = "User: Hello\nAssistant:"
out = pipe(prompt, max_new_tokens=30, do_sample=False)
print("\n\n\n", out[0]["generated_text"], "\n\n\n")

prompt = "Hi"
out = pipe(prompt, max_new_tokens=30, do_sample=False)
print("\n\n\n", out[0]["generated_text"], "\n\n\n")

prompt = "Who are you"
out = pipe(prompt, max_new_tokens=30, do_sample=False)
print("\n\n\n", out[0]["generated_text"], "\n\n\n")
