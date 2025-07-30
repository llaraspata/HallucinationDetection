# Analysing the correlation between Hallucinations and Knowledge Conflicts in Large Language Models


This project investigates whether hallucinations correlate to knowledge conflicts in LLMs. It provides tools and scripts to collect, analyze, and probe model outputs for factual inconsistencies, supporting research into model reliability and interpretability. 

To assess if hallucinations can be detected by using knowledge conflict probing models, we implemented the pipeline illustrated in the figure below.
![Hallucination by Knowledge Conflicts Schema](images/schema/Hallucination_by_KC.svg)

Vice versa, to check if knowledge conflicts can be detected by using hallucination probing models, we implemented what shown in the next figure.
![Knowledge Conflicts by Hallucinations Schema](images/schema/KC_by_Hallucination.svg)


## 🛠️ Setup

> [!NOTE]
> This project "imports" code from several reference studies by including their repositories. As a result, you need to install dependencies for each referenced repository separately by following the setup instructions for each project below.

### Root project
1. Clone the repository:
```bash
git clone github_repo_url
cd HallucinationDetection
```

2. Create and activate a virtual environment using uv
```bash
uv venv --python 3.11.5
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Hallucination probing project
1. Move to the project folder:
```bash
cd llm-hallucinations-factual-qa
```

2. Create and activate a virtual environment using uv
```bash
uv venv --python 3.11.5
source .venv/bin/activate
```

3. Install dependencies:
```bash
bash setup.sh
```

### Knowledge Conflict probing project
1. Move to the project folder:
```bash
cd SAE-based-representation-engineering
```

2. Create and activate a virtual environment using uv
```bash
uv venv --python 3.9
source .venv/bin/activate
```

3. Install dependencies:
```bash
bash ./scripts/install.sh
```


## 📊 Datasets

Our analysis on hallucination detection involved the following datasets:

- **Mu-SHROOM (SemEval 2025)**, which collects pairs of questions and hallucinated answer. Its instances cover 14 different languages. The adopted dataset is  `data/raw/labeled.json`

- **HaluEval**, available on [🤗HuggingFace](https://huggingface.co/datasets/pminervini/HaluEval), which collects human-annotated pairs of (question, answer). For our purposes, we used the `dialog` subset.

- **HaluBench**, available on [🤗HuggingFace](https://huggingface.co/datasets/PatronusAI/HaluBench), which collects instances sourced from real-world domains, spanning from finance to medicine for hallucination detection in Question-Answering tasks.

Our analysis on knowledge conflict detection involved the **NQ-Swap** dataset (available on [🤗HuggingFace](https://huggingface.co/datasets/pminervini/NQ-Swap)), collects artificially constructed conflicting data pairs designed to test and evaluate LLMs' ability to handle knowledge conflicts in question-answering tasks.

## 🧪 Experiments

> [!NOTE]
> If you have Internet access during computations, then remove the option `use_local` from the commands below, otherwise you have to download both models and datasets running the following commands:
> ```bash
> huggingface-cli download --repo-type dataset <dataset_repo_id>
> huggingface-cli download <model_repo_id>
> ```


### 1. Detect Hallucination through Knowledge Conflicts
First of all, you have to train knowledge conflict probing models. So run the following commands:

```bash
cd SAE-based-representation-engineering
source .venv/bin/activate

python -W ignore -m hallucination.probing_model.save_activations
python -W ignore -m hallucination.probing_model.activation_patterns
python -W ignore -m hallucination.probing_model.prepare_eval
python -W ignore -m hallucination.probing_model.train_probing_model
```

The last command will save all the trained probing models. You can run the cells in the notebook `SAE-based-representation-engineering/hallucination/notebook/plot_accuracy.ipynb` from Section 3, to push them in a WandB workspace. This notebook plots performance metrics for knowledge conflicts detection (in this setting only), also.

Then, you should move to the root project and run the following command to pull the model artifacts from the previous WandB workspace.
```bash
cd ../HallucinationDetection
source .venv/bin/activate
python -W ignore -m src.model.download_kc_probing_model
```

Lastly, you can run the following commands to predict and evaluate the performances of knowledge conflicts probing models on all hallucination datasets.
```bash
python -W ignore -m src.model.predict --model_name "meta-llama/Meta-Llama-3-8B" --data_name "mushroom" --use_local
python -W ignore -m src.model.predict --model_name "meta-llama/Meta-Llama-3-8B" --data_name "halu_eval" --use_local
python -W ignore -m src.model.predict --model_name "meta-llama/Meta-Llama-3-8B" --data_name "halu_bench" --use_local

python -W ignore -m src.evaluation.eval --model_name "meta-llama/Meta-Llama-3-8B" --data_name "mushroom"
python -W ignore -m src.evaluation.eval --model_name "meta-llama/Meta-Llama-3-8B" --data_name "halu_eval"
python -W ignore -m src.evaluation.eval --model_name "meta-llama/Meta-Llama-3-8B" --data_name "halu_bench"
```

The notebook `2.0-ll-results-analysis-kc.ipynb` plots the results of this last task.


### 2. Detect Knowledge Conflicts through Hallucination
First of all, you have to train collect artifacts and train hallucination probing models. So run the following commands:

```bash
cd llm-hallucinations-factual-qa
source .venv/bin/activate

python -m result_collector_kc
python -W ignore -m classifier_model
```

Then, you can run the following commands to predict and evaluate the performances of hallucinations probing models on NQ-Swap.
```bash
python -m result_collector_kc
python -m predict_kc_by_hall
```

The notebook `llm-hallucinations-factual-qa/plot_accuracy.ipynb` plots the results for both tasks.


## 📁 Project Structure

```
HallucinationDetection/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.py
├── 📁 data/                                  # Mu-SHROOM dataset
├── 📁 src/                                   # Main source code for detecting hallucinations through knowledge conflicts
│   ├── 📁 data/                              # Dataset loaders and processors
│   ├── 📁 model/                             # Core detection models and utilities
│   ├── 📁 evaluation/                        # Evaluation metrics and scripts
│   └── 📁 visualization/                     # Plotting and analysis tools
├── 📁 models/                                # Trained probing models
├── 📁 notebooks/                             # Analysis notebooks
├── 📁 results/                               # Evaluation results
├── 📁 predictions/                           # Model predictions
├── 📁 scripts/                               # Utility scripts
├── 📁 artifacts/                             # Generated artifacts and cache
├── 📁 images/                                # Documentation images and schemas
│   ├── 📁 schema/                            # Architecture diagrams (SVG)
│   └── 📁 hallucination_detection/           # Result visualizations
├── 📁 llm-hallucinations-factual-qa/         # Original hallucination detection research (with further implementation for our research)
├── 📁 SAE-based-representation-engineering/  # Original Knowledge conflict probing research (with further implementation for our research)
└── 📁 wandb/                                 # Weights & Biases experiment logs
```

