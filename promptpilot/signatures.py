import dspy

class BasicQASignature(dspy.Signature):
    """Answers questions concisely."""
    # The __doc__ attribute can be used by DSPy optimizers.
    # Alternatively, a class-level docstring like the one above also works.
    # __doc__ = "Answers questions concisely."

    question = dspy.InputField(desc="The question to be answered.")
    answer = dspy.OutputField(desc="A concise and direct answer to the question.")

# Future signatures can be added here, for example:
#
# class SummarizationSignature(dspy.Signature):
#     """Summarises a given text."""
#     __doc__ = "Summarises a given text."
#     text_to_summarize = dspy.InputField(desc="The input text that needs summarisation.")
#     summary = dspy.OutputField(desc="A brief summary of the input text.")
#
# class SentimentAnalysisSignature(dspy.Signature):
#     """Analyses the sentiment of a piece of text."""
#     __doc__ = "Analyses the sentiment of a piece of text."
#     text_input = dspy.InputField(desc="The text to analyse.")
#     sentiment = dspy.OutputField(desc="The detected sentiment (e.g., positive, negative, neutral).")
#     confidence_score = dspy.OutputField(desc="A score indicating the confidence of the sentiment analysis.")

if __name__ == '__main__':
    # Example of how signatures might be inspected or used directly (though typically used within Modules)
    print("Available Signatures in this module:")

    print(f"\n--- {BasicQASignature.__name__} ---")
    print(f"Description: {BasicQASignature.__doc__}")
    print("Inputs:")
    for name, field in BasicQASignature.inputs().items():
        print(f"  - {name}: {field.desc}")
    print("Outputs:")
    for name, field in BasicQASignature.outputs().items():
        print(f"  - {name}: {field.desc}")

    # You could instantiate a signature if needed, though it's rare to do so outside a Predictor.
    # sig = BasicQASignature()
    # print(f"\nInstantiated signature: {sig}")
