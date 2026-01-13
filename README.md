# 🧠 DiffCoT-MoE: Diffusion of Thought on Mixture-of-Experts

> **A PyTorch implementation of Latent Diffusion Reasoning attached to the Qwen-1.5-MoE-A2.7B model.**

## 📖 Overview
This project explores a novel architecture that combines **Sparse Mixture-of-Experts (MoE)** with **Diffusion Models**. Instead of generating text one token at a time (autoregressive), this model attempts to "diffuse" a complete thought vector from random noise, allowing for global planning and reasoning before outputting text.

This implementation was built to run on **free-tier GPU constraints (T4 16GB)** using 4-bit Quantization and custom device-agnostic memory management.

## 🚀 Key Features
* **Base Model:** `Qwen/Qwen1.5-MoE-A2.7B` (Sparse MoE).
* **Architecture:** Custom `DiffusionMoE` head attached to the embedding latent space.
* **Training Objective:** Denoising Score Matching (predicting noise added to reasoning traces).
* **Optimization:**
    * 4-bit Quantization (`bitsandbytes` NF4).
    * Automatic CPU/GPU offloading for embedding layers.
    * Stabilized Loss Calculation (NaN prevention via clamping).
* **Dataset:** GSM8K (Grade School Math) for reasoning tasks.
''' mermaid
graph LR
    %% Data Input Phase
    In[Input Text] --&gt; Tokenizer
    Tokenizer --&gt; Embed[Embedding Layer<br>(Continuous Space)]
    
    %% The Noise Injection
    Noise[Gaussian Noise] -.-&gt; Mix
    Embed --&gt; Mix((Mix))
    Mix --&gt; Zt[Noisy Latent Z_t]

    %% The Reasoning Loop (The Core Innovation)
    subgraph Diffusion_Process [DiffCoT Reasoning Loop]
        direction TB
        Zt --&gt; MoE[Qwen MoE Layers<br>(Sparse Router)]
        MoE -- Select Experts --&gt; Pred[Predict Noise]
        Pred --&gt; Update[Subtract Noise]
        Update -- Refine Step --&gt; Zt
    end

    %% The Output Phase
    Update -- Final Clean Vector --&gt; Z0[Clean Thought Z_0]
    Z0 --&gt; LM[LM Head Decoder]
    LM --&gt; Out[Output Text]

    %% Styling
    style MoE fill:#ffecb3,stroke:#ff9800,stroke-width:2px
    style Diffusion_Process fill:#e1f5fe,stroke:#039be5,stroke-dasharray: 5 5
    style Zt fill:#ffcdd2,stroke:#f44336
    style Z0 fill:#c8e6c9,stroke:#4caf50
## 🛠️ Tech Stack
* **Python 3.10+**
* **PyTorch** (Custom Module Logic)
* **HuggingFace Transformers** (Model Loading)
* **BitsAndBytes** (QLoRA/Quantization)
* **Accelerate** (Device Management)

## 💻 Installation

```bash
pip install -r requirements.txt
