import os
import wandb


PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
KC_MODEL_NAME = "llaraspata-cilab/Hallucination/kc_hidden_layer{layer}:v0"
BEST_LAYERS = list(range(14, 17))   # Upper bound excluded


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    run = wandb.init()

    for layer in BEST_LAYERS:
        model_name = KC_MODEL_NAME.replace("{layer}", str(layer))
        print(f"Downloading model for layer {layer} from {KC_MODEL_NAME}")
        
        artifact = run.use_artifact(model_name, type='model')
        artifact_dir = artifact.download()
        print(f"Model for layer {layer} downloaded to {artifact_dir}")

        # Move the model to the models directory
        new_model_path = os.path.join(MODELS_DIR, f"kc_hidden_layer{layer}.pt")
        os.rename(artifact_dir, new_model_path)
        print(f"Model for layer {layer} moved to {new_model_path}")

    run.finish()


if __name__ == "__main__":
    main()
