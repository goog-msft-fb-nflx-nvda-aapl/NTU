import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import timm
import loralib as lora
from decoder import Decoder, Config


class VisualProjector(nn.Module):
    """MLP to project visual tokens to language space."""
    def __init__(self, visual_dim, language_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class LLaVACaptioner(nn.Module):
    def __init__(self, decoder_path=None, lora_r=8, lora_alpha=16):
        super().__init__()

        # --- Vision Encoder (CLIP ViT-Base/16) ---
        self.vision_encoder = timm.create_model(
            'vit_base_patch16_clip_224.openai',
            pretrained=True,
            num_classes=0,  # remove classification head
        )
        visual_dim = self.vision_encoder.embed_dim  # 768

        # --- Projector ---
        config = Config()
        language_dim = config.hidden_size  # 1024
        self.projector = VisualProjector(visual_dim, language_dim)

        # --- Decoder ---
        self.decoder = Decoder(config)
        if decoder_path is not None:
            state = torch.load(decoder_path, map_location='cpu')
            self.decoder.load_state_dict(state, strict=False)

        # --- Apply LoRA to decoder attention layers ---
        self._apply_lora(lora_r, lora_alpha)

        # Freeze everything except LoRA params and projector
        self._freeze_params()

    def _apply_lora(self, r, lora_alpha):
        """Replace q_proj and v_proj in each attention layer with LoRA versions."""
        config = self.decoder.config
        for layer in self.decoder.layers:
            attn = layer.self_attn
            in_dim = config.hidden_size
            q_out = config.num_attention_heads * config.head_dim
            v_out = config.num_key_value_heads * config.head_dim

            # Replace q_proj
            new_q = lora.Linear(in_dim, q_out, r=r, lora_alpha=lora_alpha, bias=False)
            new_q.weight = attn.q_proj.weight
            attn.q_proj = new_q

            # Replace v_proj
            new_v = lora.Linear(in_dim, v_out, r=r, lora_alpha=lora_alpha, bias=False)
            new_v.weight = attn.v_proj.weight
            attn.v_proj = new_v

        # Mark only LoRA params as trainable in decoder
        lora.mark_only_lora_as_trainable(self.decoder)

    def _freeze_params(self):
        # Freeze vision encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        # Projector is fully trainable
        for p in self.projector.parameters():
            p.requires_grad = True
        # Decoder: only LoRA params trainable (set by mark_only_lora_as_trainable)

    def get_visual_tokens(self, images):
        """Extract patch tokens from ViT (exclude CLS token)."""
        features = self.vision_encoder.forward_features(images)
        # features: (B, num_patches+1, 768) — drop CLS
        patch_tokens = features[:, 1:, :]  # (B, 196, 768)
        visual_embeds = self.projector(patch_tokens)  # (B, 196, 1024)
        return visual_embeds

    def forward(self, images, input_ids):
        """
        images: (B, 3, 224, 224)
        input_ids: (B, seq_len)  — tokenized captions with im_start ... im_end + padding
        """
        B = images.size(0)
        pad_id = 151643

        # Visual tokens
        visual_embeds = self.get_visual_tokens(images)  # (B, 196, 1024)
        num_vis = visual_embeds.size(1)

        # Text embeddings
        text_embeds = self.decoder.embed_tokens(input_ids)  # (B, seq_len, 1024)

        # Concatenate: [visual | text]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, 196+seq_len, 1024)

        total_len = inputs_embeds.size(1)

        # Build causal mask
        causal_mask = torch.triu(
            torch.ones((total_len, total_len), device=images.device, dtype=torch.bool),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(B, 1, total_len, total_len)

        # Position ids
        position_ids = torch.arange(total_len, device=images.device).unsqueeze(0).expand(B, -1)

        logits = self.decoder(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=causal_mask,
        )  # (B, 196+seq_len, vocab_size)

        # Compute loss only on text tokens (shift by 1 for next-token prediction)
        # text logits start at position num_vis
        text_logits = logits[:, num_vis:-1, :]       # (B, seq_len-1, vocab_size)
        text_targets = input_ids[:, 1:]              # (B, seq_len-1)

        # Ignore padding
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss = loss_fn(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_targets.reshape(-1)
        )
        return loss

    @torch.no_grad()
    def generate(self, images, tokenizer, max_new_tokens=50):
        """Greedy decode captions for a batch of images."""
        self.eval()
        B = images.size(0)
        device = images.device

        visual_embeds = self.get_visual_tokens(images)  # (B, 196, 1024)

        im_start_id = tokenizer.encode('<|im_start|>')[0]
        im_end_id = tokenizer.encode('<|im_end|>')[0]

        # Start tokens
        cur_ids = torch.tensor([[im_start_id]] * B, dtype=torch.long, device=device)
        generated = [[] for _ in range(B)]
        done = [False] * B

        for _ in range(max_new_tokens):
            text_embeds = self.decoder.embed_tokens(cur_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            total_len = inputs_embeds.size(1)

            causal_mask = torch.triu(
                torch.ones((total_len, total_len), device=device, dtype=torch.bool),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0).expand(B, 1, total_len, total_len)

            position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

            logits = self.decoder(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=causal_mask,
            )

            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # (B,)

            for i in range(B):
                if not done[i]:
                    tok = next_tokens[i].item()
                    if tok == im_end_id:
                        done[i] = True
                    else:
                        generated[i].append(tok)

            if all(done):
                break

            cur_ids = torch.cat([cur_ids, next_tokens.unsqueeze(1)], dim=1)

        captions = []
        for ids in generated:
            captions.append(tokenizer.decode(ids))
        return captions

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
