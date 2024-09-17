import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

class RefinedLatentDiffusion(nn.Module):
    def __init__(self, esm_model, num_steps, device):
        super().__init__()
        self.esm_model = esm_model
        self.num_steps = num_steps
        self.device = device
        self.latent_dim = esm_model.embed_dim

        self.clip_model = CLIPModel(embed_dim=self.latent_dim, projection_dim=self.latent_dim)
        self.refined_representation = RefinedRepresentation(seq_len=1000)
        self.esm_attention_layers = nn.ModuleList([self.esm_model.layers[i] for i in range(-4, 0)])
        self.self_attention = SelfAttentionModule(self.esm_attention_layers)
        self.cross_attention = CrossAttentionModule(self.esm_attention_layers)

        self.beta = torch.linspace(1e-4, 0.02, num_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        self.epitope_proj = nn.Sequential(
            nn.LazyLinear(256),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1280),
        )

        self.noise_prediction_network = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 1

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        x_noisy = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        log_snr = torch.log(self.alpha_bar[t] / (1 - self.alpha_bar[t]))
        return x_noisy, log_snr

    def p_losses(self, ab_latent, ag_latent, epitope_latent, t, target_seq):
        noise = torch.randn_like(ab_latent).to(self.device)
        x_noisy, log_snr = self.q_sample(x_start=ab_latent, t=t, noise=noise)
        epitope_latent = self.epitope_proj(epitope_latent)
        ag_latent = self.self_attention(ag_latent)
        x_noisy = self.self_attention(x_noisy)
        x_noisy, epitope_latent = self.cross_attention(x_noisy, epitope_latent)
        predicted_noise = self.noise_prediction_network(x_noisy)
        diff_losses = F.mse_loss(predicted_noise, noise, reduction='none')
        diff_losses = reduce(diff_losses, 'b ... -> b', 'mean')
        if self.p2_loss_weight_gamma >= 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -self.p2_loss_weight_gamma
            diff_losses = diff_losses * loss_weight
        diff_loss = diff_losses.mean()
        diff_loss = torch.clamp(diff_loss, max=10.0)
        ab_clip, ag_clip, clip_loss = self.clip_model(ab_latent, ag_latent)
        logits = self.esm_model.lm_head(x_noisy)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))
        total_loss = diff_loss + clip_loss + ce_loss
        return total_loss, clip_loss, ce_loss, diff_loss

    @torch.no_grad()
    def p_sample(self, x, ag_latent, epitope_latent, t):
        betas_t = self.beta[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alpha[t])[:, None, None]
        epitope_latent = self.epitope_proj(epitope_latent)
        ag_latent = self.self_attention(ag_latent)
        x = self.self_attention(x)
        x, epitope_latent = self.cross_attention(x, epitope_latent)
        noise_pred = x - ag_latent
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        if t[0] > 0:
            noise = torch.randn_like(x).to(self.device)
            x0_t = model_mean + torch.sqrt(betas_t) * noise
        else:
            x0_t = model_mean
        p = 0.995
        s = torch.quantile(torch.abs(x0_t), p, dim=(1, 2), keepdim=True)
        s = torch.maximum(s, torch.ones_like(s))
        x0_t = torch.clip(x0_t, -s, s) / s
        return x0_t

    @torch.no_grad()
    def sample(self, ag_seq, epitope_seq, num_samples=1, guidance_scale=2.0):
        device = next(self.parameters()).device
        batch_converter = self.esm_model.alphabet.get_batch_converter()
        _, _, ag_tokens = batch_converter([(i, seq) for i, seq in enumerate(ag_seq)])
        _, _, epitope_tokens = batch_converter([(i, seq) for i, seq in enumerate(epitope_seq)])
        ag_tokens = ag_tokens.to(device)
        epitope_tokens = epitope_tokens.to(device)
        max_seq_length = max(ag_tokens.size(1), epitope_tokens.size(1))
        ag_tokens = pad_or_truncate(ag_tokens, max_seq_length, pad_value=self.esm_model.alphabet.padding_idx)
        epitope_tokens = pad_or_truncate(epitope_tokens, max_seq_length, pad_value=self.esm_model.alphabet.padding_idx)
        ag_latent = self.esm_model(ag_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        epitope_latent = self.esm_model(epitope_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        shape = (num_samples, ag_latent.shape[1], self.latent_dim)
        x = torch.randn(shape, device=device)
        for t in reversed(range(0, self.num_steps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x_cond = self.p_sample(x, ag_latent, epitope_latent, t_batch)
            x_uncond = self.p_sample(x, torch.zeros_like(ag_latent), torch.zeros_like(epitope_latent), t_batch)
            x = x_uncond + guidance_scale * (x_cond - x_uncond)
        logits = self.esm_model.lm_head(x)
        sequences = logits.argmax(dim=-1)
        return self.esm_model.decode(sequences)

class SelfAttentionModule(nn.Module):
    def __init__(self, esm_layers):
        super().__init__()
        self.esm_attention_layers = esm_layers

    def forward(self, x):
        attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        for esm_layer in self.esm_attention_layers:
            x = x.transpose(0, 1)
            residual = x
            x = esm_layer.self_attn_layer_norm(x)
            x, _ = esm_layer.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=~attention_mask,
                need_weights=False
            )
            x = residual + x
            residual = x
            x = esm_layer.final_layer_norm(x)
            x = esm_layer.fc1(x)
            x = F.gelu(x)
            x = esm_layer.fc2(x)
            x = residual + x
            x = x.transpose(0, 1)
        return x

class CrossAttentionModule(nn.Module):
    def __init__(self, esm_layers):
        super().__init__()
        self.esm_attention_layers = esm_layers

    def forward(self, ab_latent, epitope_latent):
        ab_latent = ab_latent.transpose(0, 1)
        epitope_latent = epitope_latent.transpose(0, 1)
        for esm_layer in self.esm_attention_layers:
            ab_residual = ab_latent
            ab_latent = esm_layer.self_attn_layer_norm(ab_latent)
            ab_latent, _ = esm_layer.self_attn(
                query=ab_latent,
                key=epitope_latent,
                value=epitope_latent,
                need_weights=False
            )
            ab_latent = ab_residual + ab_latent
            epitope_residual = epitope_latent
            epitope_latent = esm_layer.self_attn_layer_norm(epitope_latent)
            epitope_latent, _ = esm_layer.self_attn(
                query=epitope_latent,
                key=ab_latent,
                value=ab_latent,
                need_weights=False
            )
            epitope_latent = epitope_residual + epitope_latent
            ab_latent = self._apply_feed_forward(esm_layer, ab_latent)
            epitope_latent = self._apply_feed_forward(esm_layer, epitope_latent)
        ab_latent = ab_latent.transpose(0, 1)
        epitope_latent = epitope_latent.transpose(0, 1)
        return ab_latent, epitope_latent

    def _apply_feed_forward(self, esm_layer, x):
        residual = x
        x = esm_layer.final_layer_norm(x)
        x = esm_layer.fc1(x)
        x = F.gelu(x)
        x = esm_layer.fc2(x)
        x = residual + x
        return x

class CLIPModel(nn.Module):
    def __init__(self, embed_dim, projection_dim):
        super().__init__()
        self.ab_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=3
        )
        self.ag_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=3
        )
        self.project_ab = nn.Linear(embed_dim, projection_dim)
        self.project_ag = nn.Linear(embed_dim, projection_dim)

    def forward(self, ab_emb, ag_emb):
        ab_vec = self.ab_encoder(ab_emb)
        ag_vec = self.ag_encoder(ag_emb)
        ab_embed = F.normalize(self.project_ab(ab_vec[:, 0]), dim=-1)
        ag_embed = F.normalize(self.project_ag(ag_vec[:, 0]), dim=-1)
        similarity = torch.matmul(ab_embed, ag_embed.t())
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        loss_i = F.cross_entropy(similarity, labels)
        loss_t = F.cross_entropy(similarity.t(), labels)
        clip_loss = (loss_i + loss_t) / 2
        return ab_embed, ag_embed, clip_loss

class RefinedRepresentation(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, tokens, energy_scores):
        one_hot = F.one_hot(tokens, num_classes=len(esm_model.alphabet))
        motif_channel = (energy_scores <= -1).float().unsqueeze(-1)
        combined = torch.cat([one_hot, motif_channel], dim=-1)
        return combined

# Note: The esm_model is assumed to be imported or passed to this module
