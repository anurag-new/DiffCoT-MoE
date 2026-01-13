import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionMoE(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = model.config.hidden_size
        
        # Noise Schedule (1000 steps)
        self.n_steps = 1000
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, self.n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Device Agnostic Logic (Handles CPU/GPU split safely)
        buffer_device = self.alphas_cumprod.device
        t_local = t.to(buffer_device)
        
        sqrt_alpha = self.alphas_cumprod[t_local] ** 0.5
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t_local]) ** 0.5
        
        target_device = x_start.device
        sqrt_alpha = sqrt_alpha.to(target_device)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.to(target_device)
        
        return (
            sqrt_alpha.view(-1, 1, 1) * x_start +
            sqrt_one_minus_alpha.view(-1, 1, 1) * noise
        )

    def compute_loss(self, text_input):
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        # Force inputs to CUDA (Assuming model is on GPU)
        input_ids = inputs.input_ids.to("cuda")
        
        with torch.no_grad():
            # Handle potential wrapper (Accelerate/BitsAndBytes)
            if hasattr(self.model, "model"):
                clean_latents = self.model.model.embed_tokens(input_ids)
            else:
                clean_latents = self.model.get_input_embeddings()(input_ids)

            # STABILIZATION: Clamp huge values to prevent NaNs in 16-bit mode
            clean_latents = torch.clamp(clean_latents, -1.0, 1.0)

        batch_size = clean_latents.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device="cuda").long()
        noise = torch.randn_like(clean_latents)
        
        # Add noise (Forward Diffusion)
        noisy_latents = self.q_sample(clean_latents, t, noise)

        # TYPE CASTING: Ensure input matches model dtype (BFloat16/Float16)
        target_dtype = self.model.dtype 
        noisy_latents = noisy_latents.to(target_dtype)

        # Predict Noise (Reverse Diffusion Step)
        model_output = self.model(inputs_embeds=noisy_latents).logits
        
        # STABILIZATION: Force output to Float32 for Loss Calculation
        model_output = model_output.float()
        
        # Safety Net: Replace any NaNs with zeros
        if torch.isnan(model_output).any():
            model_output = torch.nan_to_num(model_output, nan=0.0)

        # Handle Dimension Mismatch (if Vocab > Hidden)
        vocab_size = model_output.shape[-1]
        if vocab_size > self.hidden_dim:
             predicted_noise = model_output[..., :self.hidden_dim]
        else:
             predicted_noise = F.pad(model_output, (0, self.hidden_dim - vocab_size))

        # Calculate MSE Loss
        loss = F.mse_loss(predicted_noise, noise.float()) 
        return loss