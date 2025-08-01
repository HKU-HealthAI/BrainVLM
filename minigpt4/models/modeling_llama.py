import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import copy

from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLMOrig
from .modelling_llama_source import LlamaForCausalLM as LlamaForCausalLM_Local


class LlamaForCausalLM(LlamaForCausalLM_Local):

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        start_visual_idx: Optional[torch.LongTensor] = None,
        end_visual_idx: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        reduction: Optional[str] = "mean",

    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            start_visual_idx = start_visual_idx,
            end_visual_idx = end_visual_idx,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 

        hidden_states = outputs[0] 
        if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states) 
        logits = logits.float()

        

        class_labels=[15,16,17,18,19,20,21,22,23,24,605,806,717,1032]
        loss = None
        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()


            loss_fct = CrossEntropyLoss(reduction=reduction)
            shift_logits = shift_logits.view(-1, 128256)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = loss_fct(shift_logits, shift_labels)
            if False:
                class_label=torch.tensor(class_labels.index(shift_labels[-3])).to(shift_logits.device)
                class_logits=shift_logits[-3][class_labels]
                add_loss=loss_fct(class_logits,class_label)
                loss=loss+add_loss
            if reduction == "none":
                loss = loss.view(logits.size(0), -1).mean(1)

        if not return_dict:  
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        outputs.hidden_states=hidden_states
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
