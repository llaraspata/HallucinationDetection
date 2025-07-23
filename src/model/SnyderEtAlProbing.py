import os
import torch

DROPOUT_MLP = 0.5
DROPOUT_GRU = 0.25


class FFHallucinationClassifier(torch.nn.Module):
    def __init__(self, input_shape, dropout = DROPOUT_MLP):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 2)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

class RNNHallucinationClassifier(torch.nn.Module):
    def __init__(self, dropout=DROPOUT_GRU):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, seq):
        gru_out, _ = self.gru(seq)
        return self.linear(gru_out)[-1, -1, :]


class SnyderEtAlProbing:
    # -------------
    # Constants
    # -------------
    MODEL_DIR = "snyder_et_al"
    ACTIVATIONS = ["logits", "fully", "attn", "attribution"]
    LAYERS = list(range(0, 32))     # Upper bound excluded
    RNN_MODEL_ATTRIBUTION_NAME = "rnn_hallucination_detection_attribution.pt"
    FFN_MODEL_LOGITS_NAME = "ffn_hallucination_detection_logits.pt"
    FFN_MODEL_LAYER_NAME = "ffn_hallucination_detection_{activation}_layer{layer}.pt"

    LABEL = {
        0: "no_hallucination",
        1: "hallucination",
    }


    # -------------
    # Constructor
    # -------------
    def __init__(self, project_dir, activation="logits", layer=0):
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Invalid activation type: {activation}. Must be one of {self.ACTIVATIONS}.")
        if layer not in self.LAYERS:
            raise ValueError(f"Invalid layer: {layer}. Must be one of {self.LAYERS}.")
        
        self.layer = layer
        self.activation = activation

        match activation:
            case "logits":
                self.model_name = self.FFN_MODEL_LOGITS_NAME
            case "attribution":
                self.model_name = self.RNN_MODEL_ATTRIBUTION_NAME
            case _:
                self.model_name = self.FFN_MODEL_LAYER_NAME.format(activation=activation, layer=layer)

        self.model_path = os.path.join(project_dir, "models", self.MODEL_DIR)
        self.model = self.load_probing_model(self.model_name)


    # -------------
    # Public Methods
    # -------------
    def load_probing_model(self, model_name):
        model_path = os.path.join(self.model_path, model_name)
        saved_data = torch.load(model_path, weights_only=True)

        if self.activation == "attribution":
            model = RNNHallucinationClassifier()
        else:
            model = FFHallucinationClassifier()
        
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

        output = self.model(torch.tensor(activation).view(1, -1, 1).to(torch.float))
        predicted = (output > 0.5).float()

        return predicted
    