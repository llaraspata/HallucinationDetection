import os
import argparse
from src.model.HallucinationDetection import HallucinationDetection


PROJECT_ROOT = os.getcwd()


def main(args):
    model_name = args.model_name
    data_name = args.data_name
    use_local = args.use_local
    hallucination_detector = HallucinationDetection(project_dir=PROJECT_ROOT)

    llm_name = model_name.split("/")[-1]

    for label, desc in HallucinationDetection.LABELS.items():
        print("=="*50)
        print(f"Predicting {desc} instances")
        print("=="*50)
        hallucination_detector.predict_llm(llm_name=model_name, use_local=use_local, data_name=data_name, label=label)
        
        for activation in HallucinationDetection.ACTIVATION_TARGET:
            for layer in HallucinationDetection.TARGET_LAYERS:
                hallucination_detector.predict_kc(target=activation, layer=layer, data_name=data_name, use_local=use_local, label=label, llm_name=llm_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hallucination Detection predictions.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the LLM model to use.")
    parser.add_argument("--data_name", type=str, default=HallucinationDetection.DEFAULT_DATASET, help="Name of the dataset to use.")
    parser.add_argument("--use_local", action="store_true", help="Use local model instead of remote.")

    args = parser.parse_args()
    main(args)
