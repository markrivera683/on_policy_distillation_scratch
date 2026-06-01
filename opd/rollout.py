from __future__ import annotations

import torch 
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class PromptItem:
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)
    
    prompt_id: str | None = None
    

@dataclass
class GeneratedText:
    text: str
    logprobs: torch.Tensor | None = None
    
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    

@dataclass
class RolloutGroup:
    prompt_item: PromptItem
    
    policy_version: str
    candidates: list[RolloutCandidate] = field(default_factory=list)
    
    meta: dict[str, Any] = field(default_factory=dict)
    

@dataclass
class RolloutCandidate:
    generated_text: GeneratedText
    prompt_id: str | None = None
    candidate_id: str
    
    policy_version: str
    

@dataclass
class RolloutResult:
    groups: list[RolloutGroup] = field(default_factory=list)
    
    metrics: dict[str, Any] = field(default_factory=dict)
    
@dataclass
class RolloutEngineConfig:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str | None = None

@dataclass
class RolloutConfig:
    num_n_per_prompt: int
    
    max_new_tokens: int
    policy_version: str
    
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    
    
class HFRolloutEngine:
    def __init__(
        self, 
        config: RolloutEngineConfig
    ) -> None:
        self.model = config.model
        self.tokenizer = config.tokenizer
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.inference_mode()
    def generate(
        self,
        batch_prompts: list[PromptItem],
        config: RolloutConfig,
    ) -> RolloutResult:
        prompts = [prpt.prompt for prpt in batch_prompts]
        
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding = True,
            truncation = True,
        ).to(self.device)
        
        outputs = self.model.generate(
            **batch,
            max_new_tokens=config.max_new_tokens,
            do_sample = config.do_sample,
            temperature = config.temperature,
            top_p = config.top_p,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            use_cache = True,
            num_return_sequences = config.num_n_per_prompt,
        )
        
        prompt_len = batch["input_ids"].shape[1]
        response_ids = outputs[:, prompt_len:]
        
        response_texts = self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens = True,
        )
        
        result = RolloutResult()
        
        for i, prompt_item in enumerate(batch_prompts):
            prompt_id = prompt_item.prompt_id
            group = RolloutGroup(
                prompt_item=prompt_item,
                policy_version=config.policy_version
            )
            
            for j in range(config.num_n_per_prompt):
                flat_idx = i * config.num_n_per_prompt + j
                
                candidate = RolloutCandidate(
                    generated_text=GeneratedText(
                        text=response_texts[flat_idx],
                        extra={
                            "token_ids": response_ids[flat_idx]
                        }
                    ),
                    prompt_id=prompt_id,
                    candidate_id=f"{prompt_id}_respond_{j}",
                    policy_version=config.policy_version
                    )
                
                group.candidates.append(candidate)
            
                
            result.groups.append(group)
            
        return result