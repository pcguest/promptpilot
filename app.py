import dspy
from modules.smart_answer import SmartAnswerModule

# Configure LLM (OpenAI's GPT-4 or GPT-3.5)
from dspy.llms import ChatGPT
dspy.settings.configure(lm=ChatGPT(model="gpt-4"))  # or "gpt-3.5-turbo"

smart_prompt = SmartAnswerModule()

print("🧠 PromptPilot is ready. Ask a question!\n(Type 'exit' to quit.)")

while True:
    user_input = input("\nAsk: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    result = smart_prompt.forward(user_input)
    print(f"\n💡 {result}\n")

