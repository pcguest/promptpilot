import dspy

class SmartAnswerSig(dspy.Signature):
    """Answer clearly, concisely, and with evidence if needed."""
    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="A helpful, fact-based, natural response")

class SmartAnswerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SmartAnswerSig)

    def forward(self, question):
        result = self.predictor(question=question)
        return result.answer

