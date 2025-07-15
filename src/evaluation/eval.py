import os
import wandb
import argparse
import src.evaluation.utilis as ut
from src.model.HallucinationDetection import HallucinationDetection


PROJECT_ROOT = os.getcwd()

WANDB_ENTITY = "llaraspata-cilab"
WANDB_PROJECT = "Hallucination"
DATASET_NAME = "mushroom"
WANDB_RUN_NAME = "eval-{dataset_name}"


def main(args):
    model_name = args.model_name
    data_name = args.data_name

    llm_name = model_name.split("/")[-1]
    run_name = WANDB_RUN_NAME.format(dataset_name=data_name)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=run_name,
        name=run_name,
        config={
            "layers": HallucinationDetection.TARGET_LAYERS,
            "activation_types": HallucinationDetection.ACTIVATION_TARGET,
        },
    )

    hallucination_detector = HallucinationDetection(project_dir=PROJECT_ROOT)

    for target in HallucinationDetection.ACTIVATION_TARGET:
        for kc_layer in HallucinationDetection.TARGET_LAYERS:
            hallucination_detector.load_kc_probing(activation=target, layer=kc_layer)
            hallucination_detector.eval(target, data_name=data_name, llm_name=llm_name)

    metrics = ut.read_metrics(os.path.join(PROJECT_ROOT, HallucinationDetection.RESULTS_DIR, llm_name, data_name))
    run.log({"metrics": wandb.Table(dataframe=metrics)})

    run.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hallucination Detection predictions.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the LLM model to use.")
    parser.add_argument("--data_name", type=str, default=HallucinationDetection.DEFAULT_DATASET, help="Name of the dataset to use.")

    args = parser.parse_args()
    main(args)

