import os
import re
import json
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import src.evaluation.utilis as ut
from src.model.HallucinationDetection import HallucinationDetection

METRICS_FILE_NAME = "metrics_"


def plot_model_accuracy(metrics_dir, images_dir, model_name, dataset_name):
    """
    Plots the model accuracy from the metrics file and saves the plot as an image.
    
    Args:
        metrics_dir (str): Directory containing the metrics files.
        images_dir (str): Directory to save the plot images.
        model_name (str): Name of the LLM.
        dataset_name (str): Name of the dataset.
    """
    records = ut.read_metrics(metrics_dir)

    model_name = model_name.split("/")[-1]
    palette = {"hidden": "red", "mlp": "blue", "attn": "green"}

    image_save_dir = os.path.join(images_dir, "hallucination_detection")
    os.makedirs(image_save_dir, exist_ok=True)

    rcParams['axes.labelsize'] = 21
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['legend.fontsize'] = 22
    rcParams['legend.title_fontsize'] = 20
    rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'mathtext.default': 'regular',
        'font.weight': 'bold',
    })

    plt.figure(figsize=(20, 12), dpi=150)
    sns.lineplot(data=records, x="layer", y="ACC", hue="activation", palette=palette)
    # plt.title(f"Probing model for conflict classification. Accuracy\n{model_name} {data_name}")
    plt.ylabel("Accuracy")
    plt.xlabel("Layer")
    plt.grid(True)
    plt.savefig(os.path.join(image_save_dir, f"{model_name} {dataset_name} Accuracy.pdf"), format='pdf', bbox_inches='tight')
    plt.show()


def plot_lang_accuracy(metrics_dir, images_dir, model_name, dataset_name, lang):
    """
    Plots the model accuracy for the specified languages from the metrics file and saves the plot as an image.
    
    Args:
        metrics_dir (str): Directory containing the metrics files.
        images_dir (str): Directory to save the plot images.
        model_name (str): Name of the LLM.
        dataset_name (str): Name of the dataset.
        lang (str): Language for which to plot the accuracy.
    """
    records = ut.read_metrics(metrics_dir)

    model_name = model_name.split("/")[-1]
    palette = {"hidden": "red", "mlp": "blue", "attn": "green"}

    image_save_dir = os.path.join(images_dir, "hallucination_detection")
    os.makedirs(image_save_dir, exist_ok=True)

    rcParams['axes.labelsize'] = 21
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['legend.fontsize'] = 22
    rcParams['legend.title_fontsize'] = 20
    rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'mathtext.default': 'regular',
        'font.weight': 'bold',
    })

    plt.figure(figsize=(20, 12), dpi=150)
    sns.lineplot(data=records, x="layer", y=lang, hue="activation", palette=palette)
    # plt.title(f"Accuracy for {lang} language")
    plt.ylabel("Accuracy")
    plt.xlabel("Layer")
    plt.grid(True)
    plt.savefig(os.path.join(image_save_dir, f"{model_name} {dataset_name} {lang} Accuracy.pdf"), format='pdf', bbox_inches='tight')
    plt.show()


def plot_all_langs_accuracy(metrics_dir, images_dir, model_name, dataset_name):
    """
    Plots the model accuracy from the metrics file and saves the plot as an image.
    
    Args:
        metrics_dir (str): Directory containing the metrics files.
        images_dir (str): Directory to save the plot images.
        model_name (str): Name of the LLM.
        dataset_name (str): Name of the dataset.
    """
    records = ut.read_metrics(metrics_dir)

    model_name = model_name.split("/")[-1]

    image_save_dir = os.path.join(images_dir, "hallucination_detection")
    os.makedirs(image_save_dir, exist_ok=True)

    rcParams['axes.labelsize'] = 21
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['legend.fontsize'] = 22
    rcParams['legend.title_fontsize'] = 20
    rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'mathtext.default': 'regular',
        'font.weight': 'bold',
    })

    all_languages = ut.get_languages(records)

    for activation in HallucinationDetection.ACTIVATION_TARGET:
        plt.figure(figsize=(20, 12), dpi=150)
        act_records = records[records["activation"] == activation]
        for lang in all_languages:
            if lang in act_records.columns:
                sns.lineplot(data=act_records, x="layer", y=lang, label=lang)
        # plt.title(f"Accuracy for languages with {activation} activation")
        plt.ylabel("Accuracy")
        plt.xlabel("Layer")
        plt.grid(True)
        plt.savefig(os.path.join(image_save_dir, f"{model_name} {dataset_name} {activation} All Languages Accuracy.pdf"), format='pdf', bbox_inches='tight')
        plt.show()
