import dspy

class BasicQASignature(dspy.Signature):
    """Answers questions concisely."""
    # The docstring of the class itself (like the one above) is used by DSPy
    # for understanding the purpose of the signature, especially for optimisers.
    # Setting __doc__ explicitly is an alternative but not necessary if a class docstring is present.
    # __doc__ = "Answers questions concisely."

    question = dspy.InputField(desc="The question that needs to be answered.")
    answer = dspy.OutputField(desc="A concise and direct answer to the provided question.")

# Additional signatures can be defined below for other tasks. For example:
#
# class TextSummarisationSignature(dspy.Signature):
#     """Summarises a given piece of text."""
#     # __doc__ = "Summarises a given piece of text." # Alternative way to set description
#
#     text_to_summarise = dspy.InputField(desc="The input text requiring summarisation.")
#     summary = dspy.OutputField(desc="A brief and coherent summary of the input text.")
#
# class SentimentAnalysisSignature(dspy.Signature):
#     """Analyses the sentiment of a given piece of text."""
#     # __doc__ = "Analyses the sentiment of a given piece of text."
#
#     text_input = dspy.InputField(desc="The text for which sentiment is to be analysed.")
#     sentiment = dspy.OutputField(desc="The detected sentiment (e.g., positive, negative, neutral).")
#     confidence_score = dspy.OutputField(desc="A numerical score indicating the confidence of the sentiment analysis.")

if __name__ == '__main__':
    # This block provides an example of how signatures can be inspected.
    # Signatures are typically used within DSPy Modules (via dspy.Predict or dspy.ChainOfThought, etc.)
    # rather than being instantiated or called directly in this manner.
    print("Demonstrating inspection of defined Signatures in this module:")

    print(f"\n--- {BasicQASignature.__name__} ---")
    # The effective description used by DSPy is derived from the class docstring or __doc__.
    print(f"Effective Description for DSPy: {BasicQASignature.__doc__}")
    print("Input Fields:")
    for name, field in BasicQASignature.inputs().items():
        print(f"  - Name: '{name}', Description: '{field.desc}'")
    print("Output Fields:")
    for name, field in BasicQASignature.outputs().items():
        print(f"  - Name: '{name}', Description: '{field.desc}'")

    # It's generally not common to instantiate a signature directly like this for typical use,
    # as DSPy's Predictors and other components handle their instantiation and usage.
    # sig_instance = BasicQASignature()
    # print(f"\nExample of an instantiated signature object: {sig_instance}")
    # print(f"Instantiated signature's question field: {sig_instance.question}")
    # print(f"Instantiated signature's answer field: {sig_instance.answer}")

    print("\nNote: Signatures define the structure of inputs and outputs for LM interactions within DSPy.")
