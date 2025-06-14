import soundfile as sf
import os
import json
import random
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import jiwer
from scipy.spatial.distance import cosine
import soundfile as sf
import soundfile as sf

# Import Chatterbox components
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.t3.modules.cond_enc import T3Cond

# Add matplotlib imports for metrics tracking
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime
import threading
import time
from collections import deque

# Hardcoded configuration
AUDIO_DATA_DIR = "./audio_data"
BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 1e-5  # Lower for GRPO
WARMUP_STEPS = 500
MAX_AUDIO_LENGTH = 400.0
MIN_AUDIO_LENGTH = 1.0
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
GRADIENT_ACCUMULATION_STEPS = 4  # Lower for GRPO
SAVE_EVERY_N_STEPS = 200
CHECKPOINT_DIR = "checkpoints_grpo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
MAX_TEXT_LENGTH = 1000
VALIDATION_SPLIT = 0.1

# GRPO specific parameters
NUM_SAMPLES_PER_INPUT = 4  # Number of samples to generate for each input
KL_COEFF = 0.01  # KL divergence coefficient
REWARD_BASELINE_MOMENTUM = 0.9  # Momentum for baseline reward estimation
TEMPERATURE = 1.0  # Temperature for sampling
TOP_K = 50  # Top-k sampling
TOP_P = 0.95  # Top-p sampling

# Reward weights
WER_WEIGHT = -1.0  # Negative because lower WER is better
SPEAKER_SIM_WEIGHT = 1.0  # Positive because higher similarity is better
LENGTH_PENALTY_WEIGHT = -0.5  # Negative for length mismatch penalty


class GRPOMetricsTracker:
    def __init__(self, save_path="grpo_training_metrics.png", update_interval=2.0):
        self.save_path = save_path
        self.update_interval = update_interval
        self.metrics = {
            'train_loss': deque(maxlen=1000),
            'val_loss': deque(maxlen=100),
            'learning_rate': deque(maxlen=1000),
            'steps': deque(maxlen=1000),
            'epochs': deque(maxlen=1000),
            'batch_loss': deque(maxlen=100),
            'gradient_norm': deque(maxlen=1000),
            'avg_reward': deque(maxlen=1000),
            'wer_score': deque(maxlen=1000),
            'speaker_sim': deque(maxlen=1000),
            'length_penalty': deque(maxlen=1000),
            'kl_divergence': deque(maxlen=1000),
            'baseline_reward': deque(maxlen=1000),
        }
        self.start_time = time.time()
        self.last_update = 0
        self.running = True
        self.lock = threading.Lock()
        
        # Initialize plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(24, 14))
        self.fig.suptitle('Chatterbox TTS GRPO Training Metrics', fontsize=16, fontweight='bold')
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Create initial plot
        self._create_initial_plot()
    
    def _create_initial_plot(self):
        """Create the initial plot layout"""
        self.fig.clf()
        
        # Create subplots
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_reward = self.fig.add_subplot(gs[0, 2:])
        self.ax_wer = self.fig.add_subplot(gs[1, 0])
        self.ax_speaker = self.fig.add_subplot(gs[1, 1])
        self.ax_length = self.fig.add_subplot(gs[1, 2])
        self.ax_kl = self.fig.add_subplot(gs[1, 3])
        self.ax_lr = self.fig.add_subplot(gs[2, 0])
        self.ax_grad = self.fig.add_subplot(gs[2, 1])
        self.ax_baseline = self.fig.add_subplot(gs[2, 2:])
        self.ax_info = self.fig.add_subplot(gs[3, :2])
        self.ax_epoch = self.fig.add_subplot(gs[3, 2:])
        
        # Configure info panel
        self.ax_info.axis('off')
        
        # Set titles
        self.ax_loss.set_title('Training Loss', fontweight='bold')
        self.ax_reward.set_title('Average Reward', fontweight='bold')
        self.ax_wer.set_title('WER Score', fontweight='bold')
        self.ax_speaker.set_title('Speaker Similarity', fontweight='bold')
        self.ax_length.set_title('Length Penalty', fontweight='bold')
        self.ax_kl.set_title('KL Divergence', fontweight='bold')
        self.ax_lr.set_title('Learning Rate', fontweight='bold')
        self.ax_grad.set_title('Gradient Norm', fontweight='bold')
        self.ax_baseline.set_title('Baseline Reward', fontweight='bold')
        self.ax_epoch.set_title('Rewards by Epoch', fontweight='bold')
        
        # Enable grids
        for ax in [self.ax_loss, self.ax_reward, self.ax_wer, self.ax_speaker,
                   self.ax_length, self.ax_kl, self.ax_lr, self.ax_grad,
                   self.ax_baseline, self.ax_epoch]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
    
    def add_metrics(self, **kwargs):
        """Add metrics to the tracker"""
        with self.lock:
            for key, value in kwargs.items():
                if key in self.metrics and value is not None:
                    self.metrics[key].append(value)
            self.last_update = time.time()
    
    def _update_loop(self):
        """Background thread to update plots"""
        while self.running:
            time.sleep(self.update_interval)
            if time.time() - self.last_update < self.update_interval * 2:
                self._update_plot()
    
    def _update_plot(self):
        """Update the plot with current metrics"""
        with self.lock:
            try:
                # Clear all axes
                for ax in [self.ax_loss, self.ax_reward, self.ax_wer, self.ax_speaker,
                          self.ax_length, self.ax_kl, self.ax_lr, self.ax_grad,
                          self.ax_baseline, self.ax_epoch]:
                    ax.clear()
                
                # Plot training loss
                if len(self.metrics['train_loss']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['train_loss']):]
                    self.ax_loss.plot(steps, list(self.metrics['train_loss']), 
                                     'b-', label='Train Loss', linewidth=2)
                    self.ax_loss.set_ylim(bottom=0)
                    self.ax_loss.legend()
                    self.ax_loss.set_title('Training Loss', fontweight='bold')
                    self.ax_loss.set_xlabel('Steps')
                    self.ax_loss.set_ylabel('Loss')
                    self.ax_loss.grid(True, alpha=0.3)
                
                # Plot average reward
                if len(self.metrics['avg_reward']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['avg_reward']):]
                    self.ax_reward.plot(steps, list(self.metrics['avg_reward']), 
                                       'g-', linewidth=2)
                    self.ax_reward.set_title('Average Reward', fontweight='bold')
                    self.ax_reward.set_xlabel('Steps')
                    self.ax_reward.set_ylabel('Reward')
                    self.ax_reward.grid(True, alpha=0.3)
                
                # Plot WER score
                if len(self.metrics['wer_score']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['wer_score']):]
                    self.ax_wer.plot(steps, list(self.metrics['wer_score']), 
                                    'r-', linewidth=2)
                    self.ax_wer.set_title('WER Score', fontweight='bold')
                    self.ax_wer.set_xlabel('Steps')
                    self.ax_wer.set_ylabel('WER')
                    self.ax_wer.grid(True, alpha=0.3)
                
                # Plot speaker similarity
                if len(self.metrics['speaker_sim']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['speaker_sim']):]
                    self.ax_speaker.plot(steps, list(self.metrics['speaker_sim']), 
                                        'c-', linewidth=2)
                    self.ax_speaker.set_title('Speaker Similarity', fontweight='bold')
                    self.ax_speaker.set_xlabel('Steps')
                    self.ax_speaker.set_ylabel('Similarity')
                    self.ax_speaker.grid(True, alpha=0.3)
                
                # Plot length penalty
                if len(self.metrics['length_penalty']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['length_penalty']):]
                    self.ax_length.plot(steps, list(self.metrics['length_penalty']), 
                                       'm-', linewidth=2)
                    self.ax_length.set_title('Length Penalty', fontweight='bold')
                    self.ax_length.set_xlabel('Steps')
                    self.ax_length.set_ylabel('Penalty')
                    self.ax_length.grid(True, alpha=0.3)
                
                # Plot KL divergence
                if len(self.metrics['kl_divergence']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['kl_divergence']):]
                    self.ax_kl.plot(steps, list(self.metrics['kl_divergence']), 
                                   'orange', linewidth=2)
                    self.ax_kl.set_title('KL Divergence', fontweight='bold')
                    self.ax_kl.set_xlabel('Steps')
                    self.ax_kl.set_ylabel('KL')
                    self.ax_kl.grid(True, alpha=0.3)
                
                # Plot learning rate
                if len(self.metrics['learning_rate']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['learning_rate']):]
                    self.ax_lr.plot(steps, list(self.metrics['learning_rate']), 
                                   'g-', linewidth=2)
                    self.ax_lr.set_title('Learning Rate', fontweight='bold')
                    self.ax_lr.set_xlabel('Steps')
                    self.ax_lr.set_ylabel('LR')
                    self.ax_lr.grid(True, alpha=0.3)
                    self.ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
                
                # Plot gradient norm
                if len(self.metrics['gradient_norm']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['gradient_norm']):]
                    self.ax_grad.plot(steps, list(self.metrics['gradient_norm']), 
                                     'lime', linewidth=2)
                    self.ax_grad.set_title('Gradient Norm', fontweight='bold')
                    self.ax_grad.set_xlabel('Steps')
                    self.ax_grad.set_ylabel('Norm')
                    self.ax_grad.grid(True, alpha=0.3)
                
                # Plot baseline reward
                if len(self.metrics['baseline_reward']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['baseline_reward']):]
                    self.ax_baseline.plot(steps, list(self.metrics['baseline_reward']), 
                                         'yellow', linewidth=2)
                    self.ax_baseline.set_title('Baseline Reward', fontweight='bold')
                    self.ax_baseline.set_xlabel('Steps')
                    self.ax_baseline.set_ylabel('Baseline')
                    self.ax_baseline.grid(True, alpha=0.3)
                
                # Update info panel
                self.ax_info.clear()
                self.ax_info.axis('off')
                
                info_text = [
                    f"GRPO Training Information",
                    f"{'='*30}",
                    f"Device: {DEVICE}",
                    f"Batch Size: {BATCH_SIZE}",
                    f"Samples per Input: {NUM_SAMPLES_PER_INPUT}",
                    f"KL Coefficient: {KL_COEFF}",
                    f"Temperature: {TEMPERATURE}",
                    f"",
                    f"Reward Weights:",
                    f"  WER: {WER_WEIGHT}",
                    f"  Speaker Sim: {SPEAKER_SIM_WEIGHT}",
                    f"  Length: {LENGTH_PENALTY_WEIGHT}",
                    f"",
                    f"Current Stats",
                    f"{'='*30}",
                ]
                
                if len(self.metrics['steps']) > 0:
                    current_step = self.metrics['steps'][-1]
                    info_text.append(f"Step: {current_step}")
                
                if len(self.metrics['epochs']) > 0:
                    current_epoch = self.metrics['epochs'][-1]
                    info_text.append(f"Epoch: {current_epoch}/{EPOCHS}")
                
                if len(self.metrics['avg_reward']) > 0:
                    current_reward = self.metrics['avg_reward'][-1]
                    info_text.append(f"Current Reward: {current_reward:.4f}")
                
                elapsed_time = time.time() - self.start_time
                info_text.append(f"")
                info_text.append(f"Time Elapsed: {elapsed_time/3600:.2f}h")
                
                self.ax_info.text(0.05, 0.95, '\n'.join(info_text), 
                                 transform=self.ax_info.transAxes,
                                 fontsize=10, verticalalignment='top',
                                 fontfamily='monospace',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.fig.text(0.99, 0.01, f"Last updated: {timestamp}", 
                             ha='right', va='bottom', fontsize=8, color='gray')
                
                # Save figure
                self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
                
            except Exception as e:
                print(f"Error updating plot: {e}")
    
    def stop(self):
        """Stop the metrics tracker"""
        self.running = False
        self.update_thread.join()
        plt.close(self.fig)


class LoRALayer(nn.Module):
    """LoRA adapter layer"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Proper initialization
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/np.sqrt(rank))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.lora_dropout(x)
        result = result @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


def inject_lora_layers(model: nn.Module, target_modules: List[str], rank: int, alpha: float, dropout: float):
    lora_layers = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                if min(module.in_features, module.out_features) < rank:
                    continue
                    
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                lora_layer = lora_layer.to(device)
                lora_layers[name] = lora_layer
                
                original_forward = module.forward
                def make_new_forward(orig_forward, lora):
                    def new_forward(x):
                        return orig_forward(x) + lora(x)
                    return new_forward
                
                module.forward = make_new_forward(original_forward, lora_layer)
    
    return lora_layers


@dataclass
class AudioSample:
    """Container for audio sample data"""
    audio_path: Path
    transcript: str
    duration: float
    sample_rate: int


class TTSDataset(Dataset):
    """Dataset handling"""
    def __init__(
        self,
        samples: List[AudioSample],
        tokenizer: EnTokenizer,
        s3_sr: int = S3_SR,
        s3gen_sr: int = S3GEN_SR,
        max_audio_length: float = MAX_AUDIO_LENGTH,
        max_text_length: int = MAX_TEXT_LENGTH,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.s3_sr = s3_sr
        self.s3gen_sr = s3gen_sr
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process audio
        audio, sr = librosa.load(sample.audio_path, sr=self.s3gen_sr)
        audio = librosa.util.normalize(audio)
        
        # Keep original padding/trimming logic
        max_samples = int(self.max_audio_length * self.s3gen_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            pad_amount = max_samples - len(audio)
            audio = np.pad(audio, (0, pad_amount), mode='constant', constant_values=0)
        
        # Resample for S3 tokenizer
        audio_16k = librosa.resample(audio, orig_sr=self.s3gen_sr, target_sr=self.s3_sr)
        
        # Process text
        text = punc_norm(sample.transcript)
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return {
            'audio': torch.FloatTensor(audio),
            'audio_16k': torch.FloatTensor(audio_16k),
            'text': text,
            'transcript': sample.transcript,  # Keep original for WER calculation
            'audio_path': str(sample.audio_path),
            'duration': sample.duration,
        }


def prepare_batch_conditionals(
    batch: Dict[str, torch.Tensor],
    model: ChatterboxTTS,
    ve: VoiceEncoder,
    s3gen: S3Gen,
) -> Tuple[T3Cond, List[dict]]:
    B = batch['audio'].size(0)
    device = model.device

    ve_embeds = []
    for i in range(B):
        try:
            wav_16k = batch['audio_16k'][i].numpy()
            
            if len(wav_16k) < S3_SR:
                wav_16k = np.pad(wav_16k, (0, S3_SR - len(wav_16k)), mode='reflect')
            
            utt_embeds = ve.embeds_from_wavs([wav_16k],
                                             sample_rate=S3_SR,
                                             as_spk=False,
                                             batch_size=8,
                                             rate=1.3,
                                             overlap=0.5)

            parts = torch.from_numpy(utt_embeds)
            ref = parts[0].unsqueeze(0)
            sims = F.cosine_similarity(parts, ref, dim=-1)
            voiced = parts[sims > 0.6]
            ve_embed = voiced.mean(0, keepdim=True) if len(voiced) else parts.mean(0, keepdim=True)
            ve_embeds.append(ve_embed)
        except Exception as e:
            print(f"Error in voice embedding {i}: {e}")
            if ve_embeds:
                ve_embed = ve_embeds[-1].clone()
            else:
                ve_embed = torch.zeros(1, 256)
            ve_embeds.append(ve_embed)

    ve_embeds = torch.cat(ve_embeds, dim=0).to(device)

    s3gen_refs = []
    for i in range(B):
        try:
            audio = batch['audio'][i].numpy()
            ref_audio = audio[:model.DEC_COND_LEN]
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))
        except Exception as e:
            print(f"Error in S3Gen ref {i}: {e}")
            ref_audio = np.zeros(model.DEC_COND_LEN)
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))

    t3_tokzr = s3gen.tokenizer
    plen = model.t3.hp.speech_cond_prompt_len
    tok_list = []
    if plen:
        for i in range(B):
            try:
                wav_16k = batch['audio_16k'][i].numpy()
                ref_16k = wav_16k[:model.ENC_COND_LEN]
                
                if len(ref_16k) < S3_SR // 2:
                    ref_16k = np.pad(ref_16k, (0, S3_SR // 2 - len(ref_16k)), mode='reflect')
                
                tokens, _ = t3_tokzr.forward([ref_16k], max_len=plen)
                tok_list.append(torch.atleast_2d(tokens))
            except Exception as e:
                print(f"Error tokenizing speech {i}: {e}")
                dummy_tokens = torch.zeros(1, plen, dtype=torch.long)
                tok_list.append(dummy_tokens)
        t3_cond_tokens = torch.cat(tok_list, dim=0).to(device)
    else:
        t3_cond_tokens = torch.empty(B, 0, dtype=torch.long, device=device)

    t3_cond = T3Cond(
        speaker_emb=ve_embeds,
        cond_prompt_speech_tokens=t3_cond_tokens,
        emotion_adv=0.5 * torch.ones(B, 1, 1, device=device),
    )

    return t3_cond, s3gen_refs


def generate_samples(
    model: ChatterboxTTS,
    batch: Dict[str, torch.Tensor],
    t3_cond: T3Cond,
    s3gen_refs: List[dict],
    num_samples: int = NUM_SAMPLES_PER_INPUT,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate multiple samples for each input in the batch"""
    batch_size = batch['audio'].size(0)
    device = model.device
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Text to tokens
            text_tokens_list = []
            for i in range(batch_size):
                text = batch['text'][i]
                tokens = model.tokenizer.text_to_tokens(text).to(device)
                text_tokens_list.append(tokens)
            
            # Pad text tokens
            max_text_len = max(t.size(-1) for t in text_tokens_list)
            text_tokens_padded = []
            for t in text_tokens_list:
                pad_amount = max_text_len - t.size(-1)
                if pad_amount > 0:
                    padded = F.pad(t, (0, pad_amount), value=model.tokenizer.pad_token_id or 0)
                else:
                    padded = t
                text_tokens_padded.append(padded)
            
            text_tokens = torch.cat(text_tokens_padded, dim=0)
            
            # Add start/stop tokens
            sot = model.t3.hp.start_text_token
            eot = model.t3.hp.stop_text_token
            
            if text_tokens.size(1) == 0 or text_tokens[0, 0] != sot:
                text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            if text_tokens.size(1) == 0 or text_tokens[0, -1] != eot:
                text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            
            # Generate speech tokens
            speech_tokens = []
            max_speech_len = int(MAX_AUDIO_LENGTH * S3_SR / 320)  # Approximate max tokens
            
            # Double batch for CFG
            text_tokens_doubled = torch.cat([text_tokens, text_tokens], dim=0)
            t3_cond_doubled = T3Cond(
                speaker_emb=torch.cat([t3_cond.speaker_emb, t3_cond.speaker_emb], dim=0),
                cond_prompt_speech_tokens=torch.cat(
                    [t3_cond.cond_prompt_speech_tokens, t3_cond.cond_prompt_speech_tokens], dim=0
                ) if t3_cond.cond_prompt_speech_tokens.numel() > 0 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
                emotion_adv=torch.cat(
                    [t3_cond.emotion_adv, t3_cond.emotion_adv], dim=0
                ) if t3_cond.emotion_adv is not None else None,
            )
            
            # Prepare initial embeddings
            embeds, len_cond = model.t3.prepare_input_embeds(
                t3_cond=t3_cond_doubled,
                text_tokens=text_tokens_doubled,
                speech_tokens=torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
            )
            
            generated_tokens = []
            for _ in range(max_speech_len):
                # Forward pass
                with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                    hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]
                
                # Get logits for next token
                speech_logits = model.t3.speech_head(hidden_states[:, -1:])
                
                # Apply temperature
                speech_logits = speech_logits / temperature
                
                # Top-k and top-p filtering
                filtered_logits = top_k_top_p_filtering(speech_logits[0, 0], top_k=top_k, top_p=top_p)
                
                # Sample
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for stop token
                if next_token.item() == model.t3.hp.stop_speech_token:
                    break
                
                generated_tokens.append(next_token)
                
                # Update embeddings for next iteration
                # Try to get the speech embedding layer
                if hasattr(model.t3, 'speech_embed'):
                    next_embed = model.t3.speech_embed(next_token.unsqueeze(0).expand(batch_size * 2, -1))
                elif hasattr(model.t3, 'speech_emb'):
                    next_embed = model.t3.speech_emb(next_token.unsqueeze(0).expand(batch_size * 2, -1))
                else:
                    # If neither exists, we need to handle this differently
                    print("Warning: Could not find speech embedding layer")
                    break
                embeds = torch.cat([embeds, next_embed], dim=1)
            
            if generated_tokens:
                speech_tokens = torch.cat(generated_tokens, dim=0).unsqueeze(0)
            else:
                speech_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
            
            # Store generated tokens and compute log probabilities
            samples.append((speech_tokens, text_tokens[0]))
    
    return samples

def compute_rewards(
    model: "ChatterboxTTS",
    samples: List[Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]],
    batch: Dict[str, torch.Tensor],
    t3_cond: T3Cond,
    s3gen_refs: List[dict],
    whisper_model,
    *,
    wer_weight: float = 1.0,
    speaker_sim_weight: float = 1.0,
    length_penalty_weight: float = 1.0,
    min_tok_for_synth: int = 10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reward = -WER * wer_weight
             +SpeakerSim * speaker_sim_weight
             -LengthPenalty * length_penalty_weight
    """

    import os, tempfile, jiwer, soundfile as sf, numpy as np, librosa, torch
    from scipy.spatial.distance import cosine
    import torch.nn.functional as F

    device, sr_gen = model.device, model.sr
    rew, wer_vals, sim_vals, lp_vals = [], [], [], []

    def safe_flatten_to_numpy(tensor_or_array):
        """Safely convert tensor/array to 1D numpy array"""
        if isinstance(tensor_or_array, torch.Tensor):
            arr = tensor_or_array.detach().cpu().numpy()
        elif isinstance(tensor_or_array, np.ndarray):
            arr = tensor_or_array
        else:
            arr = np.array(tensor_or_array)
        
        # Flatten to 1D regardless of original shape
        return arr.flatten()

    # Get reference speaker embedding (take first item in batch)
    try:
        ref_speaker_emb = t3_cond.speaker_emb
        if isinstance(ref_speaker_emb, (list, tuple)):
            ref_speaker_emb = ref_speaker_emb[0]
        elif ref_speaker_emb.dim() > 1 and ref_speaker_emb.size(0) > 1:
            ref_speaker_emb = ref_speaker_emb[0]
        
        ref_speaker_emb = safe_flatten_to_numpy(ref_speaker_emb)
        
    except Exception as e:
        print(f"Error extracting reference speaker embedding: {e}")
        # Create a dummy reference embedding
        ref_speaker_emb = np.zeros(256)

    for i, (speech_tok, _) in enumerate(samples):
        try:
            # Convert speech tokens to tensor
            if isinstance(speech_tok, np.ndarray):
                speech_tok = torch.as_tensor(speech_tok, dtype=torch.long)
            if not torch.is_tensor(speech_tok):
                raise TypeError("speech_tokens must be tensor or ndarray")

            # Check minimum token count
            if speech_tok.numel() < min_tok_for_synth:
                raise ValueError("too few speech tokens for reliable synthesis")

            speech_tok = speech_tok.to(device)

            # Synthesize audio
            with torch.no_grad():
                try:
                    # Ensure speech_tok has batch dimension
                    if speech_tok.dim() == 1:
                        speech_tok_batch = speech_tok.unsqueeze(0)
                    else:
                        speech_tok_batch = speech_tok

                    # Flow -> mel
                    mel = model.s3gen.flow_inference(
                        speech_tokens=speech_tok_batch,
                        ref_dict=s3gen_refs[0],
                        finalize=True,
                    )
                    
                    # Pad mel time dimension if <3 frames (HiFi-T conv kernel = 3)
                    if mel.size(-1) < 3:
                        mel = F.pad(mel, (0, 3 - mel.size(-1)), mode="replicate")

                    # Generate waveform
                    wav, _ = model.s3gen.hift_inference(
                        mel, torch.zeros(1, 1, 0, device=device)
                    )

                    audio = wav.squeeze().cpu().numpy()
                    
                except Exception as e:
                    print(f"Audio synthesis error for sample {i}: {e}")
                    raise ValueError("synthesis failed")

            # Compute WER
            try:
                if audio.size == 0:
                    wer = 1.0
                else:
                    # Write temporary audio file
                    fd, tmp = tempfile.mkstemp(suffix=f"_cmp_{i}.wav")
                    os.close(fd)
                    
                    # Ensure audio is 1D
                    audio_1d = audio.flatten()
                    sf.write(tmp, audio_1d, sr_gen)

                    # Move whisper to GPU if available
                    moved = False
                    if next(whisper_model.model.parameters()).device.type == "cpu" and torch.cuda.is_available():
                        whisper_model.model.to("cuda")
                        moved = True

                    try:
                        result = whisper_model(tmp, return_timestamps=False)
                        hyp = result["text"].strip() if result and "text" in result else ""
                        
                        # Get reference transcript
                        ref_transcript = batch["transcript"][0] if batch["transcript"] else ""
                        
                        if hyp and ref_transcript:
                            wer = jiwer.wer(ref_transcript, hyp)
                        else:
                            wer = 1.0
                            
                    except Exception as e:
                        print(f"Whisper transcription error for sample {i}: {e}")
                        wer = 1.0
                    
                    # Move whisper back to CPU
                    if moved: 
                        whisper_model.model.cpu()
                    
                    # Clean up temp file
                    try:
                        os.remove(tmp)
                    except:
                        pass
                        
            except Exception as e:
                print(f"WER computation error for sample {i}: {e}")
                wer = 1.0

            wer_vals.append(float(wer))

            # Compute speaker similarity
            try:
                # Check if audio is long enough
                if audio.size <= S3_SR:  # <1 s @16 kHz
                    raise ValueError("clip <1 s, embedding invalid")

                # Resample to 16kHz for voice encoder
                audio_16k = librosa.resample(
                    y=audio.astype(np.float32),
                    orig_sr=sr_gen,
                    target_sr=S3_SR,
                )

                # Generate speaker embedding for synthesized audio
                try:
                    gen_emb_raw = model.ve.embeds_from_wavs([audio_16k], sample_rate=S3_SR)
                    gen_emb_np = safe_flatten_to_numpy(gen_emb_raw)
                    
                except Exception as e:
                    print(f"Generated embedding error for sample {i}: {e}")
                    raise ValueError("generated embedding failed")
                
                # Ensure embeddings have compatible lengths
                min_len = min(len(gen_emb_np), len(ref_speaker_emb))
                if min_len == 0:
                    sim = 0.0
                else:
                    gen_emb_trimmed = gen_emb_np[:min_len]
                    ref_emb_trimmed = ref_speaker_emb[:min_len]
                    
                    # Check for zero embeddings
                    if np.allclose(gen_emb_trimmed, 0) or np.allclose(ref_emb_trimmed, 0):
                        sim = 0.0
                    else:
                        # Compute cosine similarity
                        try:
                            sim = 1.0 - cosine(gen_emb_trimmed, ref_emb_trimmed)
                            sim = float(max(0.0, min(1.0, sim)))  # Clamp to [0, 1]
                        except Exception as e:
                            print(f"Cosine similarity error for sample {i}: {e}")
                            sim = 0.0
                
            except Exception as e:
                print(f"Speaker similarity error for sample {i}: {e}")
                sim = 0.0
                
            sim_vals.append(float(sim))

            # Compute length penalty
            try:
                tgt_sec = float(batch["duration"][0]) if batch["duration"] else 1.0
                gen_sec = audio.size / sr_gen
                
                if tgt_sec <= 0:
                    tgt_sec = 1.0
                    
                r = gen_sec / tgt_sec
                
                # Length penalty: penalize if too short (<80%) or too long (>120%)
                if r < 0.8:
                    lp = (0.8 - r) ** 2
                elif r > 1.2:
                    lp = (r - 1.2) ** 2
                else:
                    lp = 0.0
                    
            except Exception as e:
                print(f"Length penalty error for sample {i}: {e}")
                lp = 1.0
                
            lp_vals.append(float(lp))

            # Compute total reward
            reward = (
                -wer_weight * wer +
                speaker_sim_weight * sim -
                length_penalty_weight * lp
            )
            rew.append(float(reward))

        except Exception as e:
            # Assign worst possible reward for failed samples
            print(f"[compute_rewards] sample {i} -> {e}")
            rew.append(-1.0)
            wer_vals.append(1.0)
            sim_vals.append(0.0)
            lp_vals.append(1.0)

    # Convert to tensor
    rewards_tensor = torch.tensor(rew, device=device, dtype=torch.float32)
    
    # Compute metrics
    metrics = {
        "wer": float(np.mean(wer_vals)) if wer_vals else 1.0,
        "speaker_sim": float(np.mean(sim_vals)) if sim_vals else 0.0,
        "length_penalty": float(np.mean(lp_vals)) if lp_vals else 1.0,
    }
    
    return rewards_tensor, metrics

def compute_grpo_loss(
    model: ChatterboxTTS,
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    rewards: torch.Tensor,
    baseline_reward: float,
    t3_cond: T3Cond,
    kl_coeff: float = KL_COEFF,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO loss"""
    device = model.device
    batch_size = 1  # Single input for now
    
    # Normalize rewards using baseline
    advantages = rewards - baseline_reward
    
    # Rank samples by reward
    ranked_indices = torch.argsort(advantages, descending=True)
    
    total_loss = 0.0
    total_kl = 0.0
    
    for rank, idx in enumerate(ranked_indices):
        speech_tokens, text_tokens = samples[idx]
        
        if speech_tokens.numel() == 0:
            continue
        
        # Prepare for forward pass
        text_tokens = text_tokens.unsqueeze(0).to(device)
        speech_tokens = speech_tokens.to(device)
        
        # Double for CFG
        text_tokens_doubled = torch.cat([text_tokens, text_tokens], dim=0)
        speech_tokens_doubled = torch.cat([speech_tokens, speech_tokens], dim=0)
        
        t3_cond_doubled = T3Cond(
            speaker_emb=torch.cat([t3_cond.speaker_emb, t3_cond.speaker_emb], dim=0),
            cond_prompt_speech_tokens=torch.cat(
                [t3_cond.cond_prompt_speech_tokens, t3_cond.cond_prompt_speech_tokens], dim=0
            ) if t3_cond.cond_prompt_speech_tokens.numel() > 0 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
            emotion_adv=torch.cat(
                [t3_cond.emotion_adv, t3_cond.emotion_adv], dim=0
            ) if t3_cond.emotion_adv is not None else None,
        )
        
        # Forward pass
        input_speech_tokens = speech_tokens_doubled[:, :-1] if speech_tokens_doubled.size(1) > 1 else torch.empty(2, 0, dtype=torch.long, device=device)
        
        embeds, len_cond = model.t3.prepare_input_embeds(
            t3_cond=t3_cond_doubled,
            text_tokens=text_tokens_doubled,
            speech_tokens=input_speech_tokens,
        )
        
        with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
            hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]
        
        # Get speech logits
        speech_start = len_cond + text_tokens.size(1)
        speech_end = min(speech_start + speech_tokens.size(1) - 1, hidden_states.size(1))
        
        if speech_start < speech_end:
            speech_hidden = hidden_states[:, speech_start:speech_end]
            speech_logits = model.t3.speech_head(speech_hidden)
            
            # Compute log probabilities
            target_tokens = speech_tokens_doubled[:, 1:speech_end-speech_start+1]
            log_probs = F.log_softmax(speech_logits, dim=-1)
            
            # Gather log probs for generated tokens
            gathered_log_probs = torch.gather(
                log_probs[0],
                dim=-1,
                index=target_tokens[0].unsqueeze(-1)
            ).squeeze(-1)
            
            # GRPO loss: weight by rank
            rank_weight = 1.0 / (rank + 1)  # Higher ranked samples get more weight
            sample_loss = -gathered_log_probs.mean() * rank_weight * advantages[idx]
            
            total_loss += sample_loss
            
            # KL divergence penalty (simplified - compare with uniform distribution)
            kl_div = (log_probs[0].exp() * log_probs[0]).sum(-1).mean()
            total_kl += kl_div
    
    # Average over samples
    num_samples = len(samples)
    total_loss = total_loss / num_samples + kl_coeff * total_kl / num_samples
    
    return total_loss, total_kl / num_samples


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits


def main():
    """Main GRPO training function"""
    print(f"Starting Chatterbox TTS GRPO fine-tuning")
    print(f"Device: {DEVICE}")
    
    # Initialize metrics tracker
    metrics_tracker = GRPOMetricsTracker(save_path="grpo_training_metrics.png", update_interval=2.0)
    
    # Load Whisper model
    print("Loading Whisper model...")
    whisper_model = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device="cuda")
    
    # Load audio samples
    samples = load_audio_samples(AUDIO_DATA_DIR, whisper_model)
    if len(samples) == 0:
        raise ValueError(f"No valid audio samples found in {AUDIO_DATA_DIR}")
    
    # Split into train/val
    random.shuffle(samples)
    val_size = int(len(samples) * VALIDATION_SPLIT)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")
    
    # Free up GPU memory by moving whisper to CPU
    whisper_model.model.cpu()
    
    # Load Chatterbox model
    print("Loading Chatterbox TTS model...")
    model = ChatterboxTTS.from_pretrained(DEVICE)
    
    # Keep reference model for KL divergence
    ref_model = ChatterboxTTS.from_pretrained(DEVICE)
    # Set the actual models to eval mode
    ref_model.t3.eval()
    ref_model.ve.eval()
    ref_model.s3gen.eval()
    # Freeze all parameters
    for param in ref_model.t3.parameters():
        param.requires_grad = False
    for param in ref_model.ve.parameters():
        param.requires_grad = False
    for param in ref_model.s3gen.parameters():
        param.requires_grad = False
    
    # Inject LoRA layers
    print("Injecting LoRA layers...")
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_layers = inject_lora_layers(
        model.t3.tfmr,
        target_modules,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )
    print(f"Injected {len(lora_layers)} LoRA layers")
    
    # Create datasets
    train_dataset = TTSDataset(train_samples, model.tokenizer)
    val_dataset = TTSDataset(val_samples, model.tokenizer)
    
    # Set num_workers to 0 on Windows
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Setup optimizer (only LoRA parameters)
    lora_params = []
    for layer in lora_layers.values():
        lora_params.extend([layer.lora_A, layer.lora_B])
    
    optimizer = AdamW(lora_params, lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * EPOCHS,
        eta_min=LEARNING_RATE * 0.1
    )
    
    # Initialize baseline reward
    baseline_reward = 0.0
    
    # Training loop
    print("Starting GRPO training...")
    global_step = 0
    scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None
    
    for epoch in range(EPOCHS):
        # Training
        model.t3.train()
        model.ve.eval()  # Keep voice encoder in eval mode
        model.s3gen.eval()  # Keep S3Gen in eval mode
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare conditionals
            t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, model.ve, model.s3gen)
            
            # Generate samples
            samples = generate_samples(
                model, batch, t3_cond, s3gen_refs,
                num_samples=NUM_SAMPLES_PER_INPUT,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P
            )
            
            # Compute rewards
            rewards, reward_metrics = compute_rewards(
                model, samples, batch, t3_cond, s3gen_refs, whisper_model
            )
            
            # Update baseline reward
            avg_reward = rewards.mean().item()
            baseline_reward = (
                REWARD_BASELINE_MOMENTUM * baseline_reward +
                (1 - REWARD_BASELINE_MOMENTUM) * avg_reward
            )
            
            # Compute GRPO loss
            loss, kl_div = compute_grpo_loss(
                model, samples, rewards, baseline_reward, t3_cond, kl_coeff=KL_COEFF
            )
            
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Calculate gradient norm
                grad_norm = 0.0
                for p in lora_params:
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                train_steps += 1
                
                # Update metrics
                current_lr = scheduler.get_last_lr()[0]
                
                metrics_tracker.add_metrics(
                    train_loss=train_loss / train_steps,
                    learning_rate=current_lr,
                    steps=global_step,
                    epochs=epoch,
                    batch_loss=loss.item() * GRADIENT_ACCUMULATION_STEPS,
                    gradient_norm=grad_norm,
                    avg_reward=avg_reward,
                    wer_score=reward_metrics['wer'],
                    speaker_sim=reward_metrics['speaker_sim'],
                    length_penalty=reward_metrics['length_penalty'],
                    kl_divergence=kl_div.item(),
                    baseline_reward=baseline_reward,
                )
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{train_loss/train_steps:.4f}',
                    'reward': f'{avg_reward:.4f}',
                    'wer': f'{reward_metrics["wer"]:.3f}'
                })
                
                # Save checkpoint
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    save_checkpoint(model, lora_layers, optimizer, epoch, global_step, 
                                   train_loss/train_steps, CHECKPOINT_DIR)
        
        # Validation
        model.t3.eval()
        model.ve.eval()
        model.s3gen.eval()
        val_rewards = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, model.ve, model.s3gen)
                
                # Generate single sample for validation
                samples = generate_samples(
                    model, batch, t3_cond, s3gen_refs,
                    num_samples=1,
                    temperature=1.0,  # No temperature for validation
                    top_k=0,  # No filtering for validation
                    top_p=0.0
                )
                
                # Compute rewards
                rewards, _ = compute_rewards(
                    model, samples, batch, t3_cond, s3gen_refs, whisper_model
                )
                
                val_rewards.append(rewards.mean().item())
        
        avg_val_reward = np.mean(val_rewards) if val_rewards else 0.0
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/train_steps:.4f}, Val Reward: {avg_val_reward:.4f}")
        
        # Save epoch checkpoint
        save_checkpoint(model, lora_layers, optimizer, epoch, global_step, avg_val_reward, CHECKPOINT_DIR)
    
    print("Training completed!")
    
    # Stop metrics tracker
    metrics_tracker.stop()
    
    # Save final LoRA adapter
    final_adapter_path = Path(CHECKPOINT_DIR) / "final_grpo_lora_adapter.pt"
    save_lora_adapter(lora_layers, str(final_adapter_path))
    
    # Create and save merged model
    print("Creating merged model...")
    merged_model = ChatterboxTTS.from_pretrained(DEVICE)
    
    # Re-inject LoRA layers and load final weights
    merged_lora_layers = inject_lora_layers(
        merged_model.t3.tfmr,
        target_modules,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )
    
    # Copy trained weights
    for name, layer in lora_layers.items():
        if name in merged_lora_layers:
            merged_lora_layers[name].lora_A.data = layer.lora_A.data.clone()
            merged_lora_layers[name].lora_B.data = layer.lora_B.data.clone()
    
    # Merge LoRA weights into base model
    merged_model = merge_lora_weights(merged_model, merged_lora_layers)
    
    # Save merged model
    merged_dir = Path(CHECKPOINT_DIR) / "merged_grpo_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(merged_model.ve.state_dict(), merged_dir / "ve.pt")
    torch.save(merged_model.t3.state_dict(), merged_dir / "t3_cfg.pt")
    torch.save(merged_model.s3gen.state_dict(), merged_dir / "s3gen.pt")
    
    # Copy tokenizer
    import shutil
    tokenizer_path = Path(hf_hub_download(repo_id="ResembleAI/chatterbox", filename="tokenizer.json"))
    shutil.copy(tokenizer_path, merged_dir / "tokenizer.json")
    
    print(f"Saved GRPO merged model to {merged_dir}")
    print("\nTraining complete!")


def load_audio_samples(audio_dir: str, whisper_model) -> List[AudioSample]:
    """Load audio files and generate transcripts using Whisper"""
    samples = []
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    # Cache file for transcripts
    cache_file = Path(audio_dir) / "transcripts_cache.json"
    transcript_cache = {}
    
    # Load existing cache if available
    if cache_file.exists():
        print(f"Loading transcript cache from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            transcript_cache = json.load(f)
    
    print(f"Loading audio files from {audio_dir}...")
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Track if we need to update cache
    cache_updated = False
    
    for audio_path in tqdm(audio_files, desc="Processing audio"):
        try:
            # Load audio for duration check
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            # Skip if too short or too long
            if duration < MIN_AUDIO_LENGTH or duration > MAX_AUDIO_LENGTH:
                continue
            
            # Check if we have cached transcript
            audio_path_str = str(audio_path.relative_to(Path(audio_dir)))
            
            if audio_path_str in transcript_cache:
                transcript = transcript_cache[audio_path_str]['transcript']
                print(f"Using cached transcript for {audio_path.name}")
            else:
                # Transcribe with Whisper
                print(f"\nTranscribing {audio_path.name}...")
                result = whisper_model(str(audio_path), return_timestamps=True)
                transcript = result['text'].strip()
                
                # Add to cache
                transcript_cache[audio_path_str] = {
                    'transcript': transcript,
                    'duration': duration,
                    'sample_rate': sr
                }
                cache_updated = True
            
            if transcript:
                samples.append(AudioSample(
                    audio_path=audio_path,
                    transcript=transcript,
                    duration=duration,
                    sample_rate=sr
                ))
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Save updated cache
    if cache_updated:
        print(f"Saving transcript cache to {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_cache, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully loaded {len(samples)} samples")
    return samples


def save_checkpoint(
    model: ChatterboxTTS,
    lora_layers: Dict[str, LoRALayer],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metric: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint"""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch}_step{step}.pt"
    if is_best:
        checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract LoRA weights
    lora_state_dict = {}
    for name, layer in lora_layers.items():
        lora_state_dict[f"{name}.lora_A"] = layer.lora_A
        lora_state_dict[f"{name}.lora_B"] = layer.lora_B
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'metric': metric,
        'lora_state_dict': lora_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def merge_lora_weights(model: ChatterboxTTS, lora_layers: Dict[str, LoRALayer]):
    """Merge LoRA weights into the base model"""
    with torch.no_grad():
        for name, lora_layer in lora_layers.items():
            # Find the corresponding linear layer in the model
            parts = name.split('.')
            module = model.t3.tfmr
            for part in parts[:-1]:
                module = getattr(module, part)
            linear_layer = getattr(module, parts[-1])
            
            # Compute LoRA update: W' = W + BA * scaling
            lora_update = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
            
            # Add to original weights
            linear_layer.weight.data += lora_update
    
    return model


def save_lora_adapter(lora_layers: Dict[str, LoRALayer], filepath: str):
    """Save LoRA adapter weights and configuration"""
    adapter_dict = {
        'lora_config': {
            'rank': LORA_RANK,
            'alpha': LORA_ALPHA,
            'dropout': LORA_DROPOUT,
            'target_modules': list(set(name.split('.')[-1] for name in lora_layers.keys())),
        },
        'lora_weights': {},
    }
    
    for name, layer in lora_layers.items():
        adapter_dict['lora_weights'][name] = {
            'lora_A': layer.lora_A.cpu(),
            'lora_B': layer.lora_B.cpu(),
        }
    
    torch.save(adapter_dict, filepath)
    print(f"Saved LoRA adapter to {filepath}")


def collate_fn(samples):
    """Custom collate function for DataLoader"""
    return {
        'audio': torch.stack([s['audio'] for s in samples]),
        'audio_16k': torch.stack([s['audio_16k'] for s in samples]),
        'text': [s['text'] for s in samples],
        'transcript': [s['transcript'] for s in samples],
        'audio_path': [s['audio_path'] for s in samples],
        'duration': torch.tensor([s['duration'] for s in samples]),
    }


if __name__ == "__main__":
    main()