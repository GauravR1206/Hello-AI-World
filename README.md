# 👋 Hello AI World – Setting Up AI Environments with `uv`

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![uv](https://img.shields.io/badge/uv-fast%20env%20manager-orange)](https://github.com/astral-sh/uv)

A **hands-on AI Summer School session** showing you how to quickly set up Python environments for AI development using [`uv`](https://github.com/astral-sh/uv). We'll explore two modes: **fun instant tools** with `uvx` and **serious project environments** for training deep learning models.

## 🌟 What You'll Learn

* 🛡️ **Why Virtual Environments Matter** – Keep projects isolated and conflict-free
* ⚡ **`uv` Basics** – Fast installs, clean env management
* 🐄 **Fun with Tools** – Try out terminal toys instantly with `uvx`
* 🧠 **Real ML Workflows** – Train a Variational Autoencoder (VAE) in an isolated environment

---

## 1️⃣ Why Virtual Environments?

Virtual environments let you:

* Avoid dependency conflicts between projects
* Share code with reproducible installs
* Experiment without breaking your system Python

**tl;dr:** They keep your projects clean, reproducible, and conflict‑free.

---

## 2️⃣ Installing `uv` & Playing with Tools

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Tool Mode Fun 🐄

With `uvx`, you can run CLI tools instantly without installing them permanently:

```bash
uvx fortune | uvx cowsay
uvx cowsay "Hello AI World!"
```

* **fortune** prints a random quote
* **cowsay** makes a cow say it 🐮

Chain them for instant smiles.

---

## 3️⃣ Create an Environment & Train a VAE

### Step 1 – Add Project Files

`git clone `

`train.py`: Your training script for a VAE (provided in the session).

### Step 2 – Install Dependencies

```bash
uv sync
```

This creates & populates your virtual environment from `pyproject.toml`.

### Step 3 – Train the Model

```bash
uv run python train.py
```

`uv run` ensures the right environment is used—no manual activation needed.

---

**🎉 That's it!** You’ve learned how to:

* Use `uv` for quick, fun CLI tools
* Create a clean, reproducible environment for ML
* Train a real deep learning model in minutes

