import os
import wandb

WANDB_ENTITY = "llaraspata_cilab"
WANDB_PROJECT = "Hallucination"

PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
KC_MODEL_NAME = "llaraspata_cilab/Hallucination/kc_{activation_type}_layer{layer}:latest"
BEST_LAYERS = list(range(0, 32))   # Upper bound excluded
ACTIVATION_TYPES = ["hidden", "mlp", "attn"]


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    for activation_type in ACTIVATION_TYPES:
        print("--"*50)
        print(f"{activation_type.capitalize()} Activations")
        print("--"*50)
        for layer in BEST_LAYERS:
            model_name = KC_MODEL_NAME.format(activation_type=activation_type, layer=layer)
            print(f"Downloading model for layer {layer} from {model_name}")
            
            artifact = run.use_artifact(model_name, type='model')
            artifact_dir = artifact.download()
            print(f"Model for layer {layer} downloaded to {artifact_dir}")

            # Move the model to the models directory
            new_model_path = os.path.join(MODELS_DIR, f"kc_{activation_type}_layer{layer}.pt")
            os.rename(artifact_dir, new_model_path)
            print(f"Model for layer {layer} moved to {new_model_path}")
        print("--"*50)

    run.finish()


if __name__ == "__main__":
    main()
