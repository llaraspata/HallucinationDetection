import os
from src.model.HallucinationDetection import HallucinationDetection
import argparse


PROJECT_ROOT = os.getcwd()


def main(args):
    model_name = args.model_name
    use_local = args.use_local
    hallucination_detector = HallucinationDetection(project_dir=PROJECT_ROOT)

    hallucination_detector.predict_llm(llm_name=model_name, use_local=use_local)
    
    for activation in HallucinationDetection.ACTIVATION_TARGET:
        for layer in HallucinationDetection.TARGET_LAYERS:
            hallucination_detector.predict_kc(target=activation, layer=layer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hallucination Detection predictions.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the LLM model to use.")
    parser.add_argument("--use_local", action="store_true", help="Use local model instead of remote.")

    args = parser.parse_args()
    main(args)
