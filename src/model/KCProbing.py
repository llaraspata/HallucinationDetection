import os
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, use_bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(0)

    @torch.inference_mode()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x.to(self.linear.weight.device)
        return torch.sigmoid(self.linear(x)).squeeze(0)
    

class KCProbing:
    # -------------
    # Constants
    # -------------
    MODEL_DIR = "kc_{activation}_layer{layer}.pt"
    MODEL_NAME = "prob_model_list_{layer}_L1factor3.pt"

    LABEL = {
        0: "no_conflict",
        1: "conflict",
    }


    # -------------
    # Constructor
    # -------------
    def __init__(self, project_dir, activation="hidden", layer=16):
        self.layer = layer
        self.activation = activation
        self.model_name = self.MODEL_NAME.format(layer=layer)
        self.model_path = os.path.join(project_dir, "models", self.MODEL_DIR.format(activation=activation, layer=layer), self.model_name)
        self.model = self.load_probing_model()


    # -------------
    # Public Methods
    # -------------
    def load_probing_model(self):
        saved_data = torch.load(self.model_path, weights_only=True)
        
        model = LogisticRegression(input_dim=4096, use_bias=True)
        
        if isinstance(saved_data, list):
            if len(saved_data) > 0:
                if hasattr(saved_data[0], 'state_dict'):
                    model.load_state_dict(saved_data[0].state_dict())
                elif isinstance(saved_data[0], dict):
                    model.load_state_dict(saved_data[0])
                else:
                    print(f"Unexpected format in list: {type(saved_data[0])}")
                    return
            else:
                print("Empty list found in saved file")
                return
        elif isinstance(saved_data, dict):
            model.load_state_dict(saved_data)
        else:
            print(f"Unexpected save format: {type(saved_data)}")
            return
        
        model.eval()
        print("Model loaded successfully. Ready for inference.")

        return model
    

    @torch.no_grad()
    def predict(self, activation):
        self.model.to("cuda").eval()

        output = self.model(activation.cuda())
        predicted = (output > 0.5).float()

        return predicted
