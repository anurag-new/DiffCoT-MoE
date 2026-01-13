import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import your custom class from the other file
from diffusion_model import DiffusionMoE

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
SAVE_PATH = "diffusion_moe_v1.pt"
BATCH_SIZE = 2
LR = 1e-4
NUM_STEPS = 50  # Set to higher (e.g., 1000) for real training

def main():
    print("⏳ Loading Data (GSM8K)...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    def format_instruction(sample):
        return f"Question: {sample['question']}\nAnswer: {sample['answer']}"
    
    def collate_fn(batch):
        return [format_instruction(item) for item in batch]
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print("⏳ Loading Model (4-bit Quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Force Embeddings to GPU (Critical Fix)
    print("🚚 Moving Embeddings to GPU...")
    if hasattr(model, "model"):
        model.model.embed_tokens = model.model.embed_tokens.to("cuda")
    else:
        model.get_input_embeddings().to("cuda")
        
    # Initialize Diffusion Head
    diff_model = DiffusionMoE(model, tokenizer)
    print("✅ Model Initialized.")
    
    # Optimizer
    optimizer = optim.AdamW(diff_model.parameters(), lr=LR)
    diff_model.train()
    
    print("🚀 Starting Training Loop...")
    step = 0
    epoch_loss = 0
    progress_bar = tqdm(train_loader, total=NUM_STEPS)
    
    for batch_texts in progress_bar:
        optimizer.zero_grad()
        loss = diff_model.compute_loss(batch_texts)
        loss.backward()
        optimizer.step()
        
        step += 1
        epoch_loss += loss.item()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
        if step >= NUM_STEPS:
            print("✅ Training Target Reached.")
            break
            
    print(f"\n🎉 SUCCESS! Final Avg Loss: {epoch_loss / step:.4f}")
    
    # Save Model
    torch.save(diff_model.state_dict(), SAVE_PATH)
    print(f"💾 Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()