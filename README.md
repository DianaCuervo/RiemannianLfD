# Riemannian Learning from Demonstration (LfD)

A modular, configuration-driven PyTorch framework for training Neural Ordinary Differential Equations (NODE) on Riemannian manifolds using Variational Autoencoders (VAEs).

## 🚀 Features
* **Configuration-Driven Architecture:** Control model architecture, data paths, and hyperparameters entirely via `node_config.yaml`.
* **Automated Physics Scaling:** Automatically calculates `safe_baseline` and `metric_scale_energy` from raw human demonstrations to perfectly balance Imitation and Energy losses.
* **Smart Data Pipeline:** Automatically slices, interpolates, and caches datasets to prevent redundant processing.
* **Vector Graphics Logging:** Automatically generates paper-ready SVG plots of the training tug-of-war.

## 📁 Project Structure

## 🛠️ Installation
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## 🏃‍♂️ Usage
To train a model, simply run `main.py` specifying the dataset and shape. The pipeline will automatically handle preprocessing, hyperparameter calculation, and training: