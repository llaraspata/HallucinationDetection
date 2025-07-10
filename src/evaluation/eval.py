import os
import wandb
import src.evaluation.utilis as ut
from src.model.HallucinationDetection import HallucinationDetection


PROJECT_ROOT = os.getcwd()

WANDB_ENTITY = "llaraspata-cilab"
WANDB_PROJECT = "Hallucination"
DATASET_NAME = "mushroom"
WANDB_RUN_NAME = f"eval-{DATASET_NAME}"


def main():
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=WANDB_RUN_NAME,
        name=WANDB_RUN_NAME,
        config={
            "layers": HallucinationDetection.TARGET_LAYERS,
            "activation_types": HallucinationDetection.ACTIVATION_TARGET,
        },
    )

    hallucination_detector = HallucinationDetection(project_dir=PROJECT_ROOT)

    for target in HallucinationDetection.ACTIVATION_TARGET:
        for kc_layer in HallucinationDetection.TARGET_LAYERS:
            hallucination_detector.load_kc_probing(activation=target, layer=kc_layer)
            hallucination_detector.eval(target)

    metrics = ut.read_metrics(os.path.join(PROJECT_ROOT, HallucinationDetection.RESULTS_DIR))
    run.log({"metrics": wandb.Table(dataframe=metrics)})

    run.finish()
    

if __name__ == "__main__":
    main()
