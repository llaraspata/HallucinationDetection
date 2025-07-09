import os
from src.model.HallucinationDetection import HallucinationDetection

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
USE_LOCAL = False
KC_LAYERS = list(range(14, 17))         # Upper bound excluded
PROJECT_ROOT = os.getcwd()


def main():
    for layer in KC_LAYERS:
        print("=="*50)
        print(f"Hallucination Detection Evaluation for layer {layer}")
        print("=="*50)
        hallucination_detector = HallucinationDetection(
            project_dir=PROJECT_ROOT,
            llm_name=MODEL_NAME,
            kc_layer=layer,
            use_local=USE_LOCAL
        )

        print("\n\n")

        hallucination_detector.eval()
        print("=="*50)


if __name__ == "__main__":
    main()
