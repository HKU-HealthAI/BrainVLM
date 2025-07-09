import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import open_clip
from minigpt4.common.registry import registry
from minigpt4.models.base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList,AutoTokenizer, AutoModel
from minigpt4.models.base_model import LayerNorm
from minigpt4.conversation.conversation import StoppingCriteriaSub
import math
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import pdb



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # 768
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 1,5,32,1,784   vs    1,5,32,1,4    vs      1, 160, 512
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class resolution_attention(nn.Module):

    def __init__(self, in_channels=16, out_channels=8, emb_dim=768, output_dim=768, dropout=0.1, aropout=0.0):
        super(resolution_attention, self).__init__()
        self.emb_dim = emb_dim
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
        self.attn = None
        self.output_linear = nn.Linear(emb_dim,emb_dim)
        self.dropout = nn.Dropout(p=dropout)  #  Dropout
        self.dropout_2 = nn.Dropout(p=dropout)  #  Dropout
        self.norm = nn.LayerNorm(emb_dim)  #  LayerNorm
    
    def forward(self, LR_image, HR_image, context=None, mask=None):
        '''
        :param x: [1-4, 32, 768]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        ''' 
        version = 2 # or 1 2
        if version == 0:
            # previous version  
            LR_image = LR_image.reshape(-1,LR_image.size(-1)).reshape(1,-1,LR_image.size(-1))  # 1, 128, 768
            HR_image = HR_image.reshape(-1,HR_image.size(-1)).reshape(1,-1,HR_image.size(-1))  # 1, 512, 768
        elif version == 1:
            LR_image = LR_image.view(1, -1, 32, 1, LR_image.size(-1))
            HR_image = HR_image.view(1, -1, 32, 4, HR_image.size(-1))  # 32*4
        elif version == 2:  
            # breakpoint()
            # print(f"LR_image.shape:{LR_image.shape},HR_image.shape:{HR_image.shape}")
            # 修改 ViT HR 196*224*4, 1, 768  LR 224, 1, 768
            LR_image = LR_image.view(1, -1, 32, 1, LR_image.size(-1))
            HR_image = HR_image.view(1, -1, 32, 4*196, HR_image.size(-1))  
        query_list=self.Wq(LR_image)  # version3: 1,7,32,1,768     version2: 1,5,32,1,768     version1: 1, 160, 768
        key_list=self.Wk(HR_image)    # version3: 1,7,32,784,768   version2: 1,5,32,4,768     version1: 1, 640, 768
        value_list=self.Wv(HR_image)  # version3: 1,7,32,784,768   version2: 1,5,32,4,768     version1: 1, 640, 768
        x, self.attn = attention(query_list, key_list, value_list, mask=mask, dropout=self.dropout)  # 1,5,32,1,768
        if version in [1, 2]:
            
            x = x.view(1,-1,LR_image.size(-1))
            query_list = query_list.view(1,-1,LR_image.size(-1))
        
        x = self.output_linear(x)  # [1, 128, 768]
        x = self.norm(query_list + self.dropout_2(x))  # [1, 128, 768]
        return x





class MiniGPTBase(BaseModel):


    def __init__(
        self,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=False,
        llama_model="",
        max_txt_len=32,
        max_context_len=3800,
        prompt_template="",
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=8,  # or 1 lora_r means lora is not used #lora_r=8,lora_a=32
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=32,  # or 64 128/256
        lora_dropout=0.05,
        modality_number=5,
        model_2d_or_3d="3d",
        self_training=False
    ):
        super().__init__()
        self.self_training=self_training
        freeze_vit=not self_training 
        self.model_2d_or_3d=model_2d_or_3d

        self.llama_model, self.llama_tokenizer = self.init_llm(
            llama_model_path=llama_model,
            low_resource=low_resource,
            low_res_device=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.visual_encoder_list=[]
        self.ln_vision_list=[]
        self.modality_number=modality_number

        self.llama_model = self.llama_model.cuda()

        self.llama_tokenizer.add_special_tokens({'additional_special_tokens':["<box>", "<Img>", "</Img>", "<t>"]})
        
        if len(self.llama_tokenizer) > self.llama_model.get_input_embeddings().weight.shape[0] and lora_r>0:
            print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
            self.llama_model.base_model.model.model.resize_token_embeddings(new_num_tokens = len(self.llama_tokenizer))
        else:
            self.llama_model.base_model.resize_token_embeddings(new_num_tokens = len(self.llama_tokenizer))
       
        self.start_visual_token_idx = None  
        self.end_visual_token_idx = None  

        
        
        
        self.cross_attention=resolution_attention()
        
            
        if self.model_2d_or_3d=='2d':
            
            
            model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            
            self.visual_encoder = model.visual.trunk 
            self.visual_encoder.requires_grad_(False)

            is_unfrozen = True   
            if is_unfrozen:
                self.visual_encoder.norm.weight.requires_grad_(True)
                self.visual_encoder.norm.bias.requires_grad_(True)
                
            self.llama_proj_mlps = nn.Sequential(
                nn.Linear(768, 4096),
                nn.GELU(),
                nn.Linear(4096, 4096),
            )


        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym
        
        self.prompt_template = prompt_template
        self.prompt_list = []
        self.grad_list=[]

    def freeze_model(self):
        visual_num=0
        llama_num=0
        for n,p in self.named_parameters():
            if 'visual_encoder' in n or 'llama_proj_mlps' in n or 'ln_vision' in n:
                # print(n)
                p.requires_grad=True
                visual_num+=p.numel()
            else:
                llama_num+=p.numel()
                p.requires_grad=False
                self.grad_list.append(p)
                
    def vit_to_cpu(self):

        self.visual_encoder.to("cpu")
        self.visual_encoder.float()


    
    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')  
        
        
        
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids 
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        count_current_len=1
        old_version=1
        if old_version==0:
            mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
            mixed_embs = torch.cat(mixed_embs, dim=1)

           
            bos_embed = self.embed_tokens(self.llama_tokenizer(self.llama_tokenizer.bos_token, return_tensors="pt", add_special_tokens=False).to(mixed_embs.device).input_ids)
        
            eos_embed = self.embed_tokens(self.llama_tokenizer(self.llama_tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).to(mixed_embs.device).input_ids)

            mixed_embs = torch.cat([bos_embed, mixed_embs,eos_embed], dim=1)            
        else:
            image_list=[]
            cur_att=1
            if cur_att==0:
                for index in range(1,5):
                    image_list.append(img_list[0][:,32*(index-1):32*index])
                mixed_embs = [emb for pair in zip(seg_embs[:-1], image_list) for emb in pair] + [seg_embs[-1]]

                
                mixed_embs=mixed_embs+[seg_embs[-1]]

                        
                mixed_embs = torch.cat(mixed_embs, dim=1)
                bos_embed = self.embed_tokens(self.llama_tokenizer(self.llama_tokenizer.bos_token, return_tensors="pt", add_special_tokens=False).to(mixed_embs.device).input_ids)
                mixed_embs = torch.cat([bos_embed, mixed_embs], dim=1)

            elif cur_att==1:
                image_list=[]

                for index in range(1,5):
                    image_list.append(img_list[0][:,32*(index-1):32*index])

                count_current_len = 1  
                self.start_visual_token_idx = []  
                self.end_visual_token_idx = []             
                mixed_embs=[]
                for pair in zip(seg_embs[:-1], image_list):
                    count_current_len+=1
                    self.start_visual_token_idx.append(count_current_len)
                    for emb in pair:
                        
                        mixed_embs.append(emb)
                    count_current_len += 32
                    self.end_visual_token_idx.append(count_current_len)
                self.llama_model.set_index(self.start_visual_token_idx,self.end_visual_token_idx)
                mixed_embs=mixed_embs+[seg_embs[-1]]

                mixed_embs = torch.cat(mixed_embs, dim=1)
                bos_embed = self.embed_tokens(self.llama_tokenizer(self.llama_tokenizer.bos_token, return_tensors="pt", add_special_tokens=False).to(mixed_embs.device).input_ids)
                eos_embed = self.embed_tokens(self.llama_tokenizer(self.llama_tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).to(mixed_embs.device).input_ids)

                mixed_embs = torch.cat([bos_embed, mixed_embs,eos_embed], dim=1)
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):  
        emb_lists = []
        if isinstance(prompts, str):
            prompts = [prompts] * len(img_embeds)
        self.start_visual_token_idx = None
        self.end_visual_token_idx = None
        cond_ids_list=[]
        for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
            pn = each_img_embed.shape[-2]  

            if lengths is not None:
                each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                each_img_embed = each_img_embed[:lengths[idx] * pn]
            p_segs = each_prompt.split('<ImageHere>')  
            interleave_emb = []
            interleave_ids=[]
            count_current_len = 1  
            self.start_visual_token_idx = []  
            self.end_visual_token_idx = []  
            if '<t>' in each_prompt: 
                vis_chunk_size = 32  
                for idx, seg in enumerate(p_segs[:-1]):  
                    p_tokens = self.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)  
                    interleave_ids += p_tokens.input_ids.squeeze(0).tolist()
                    
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    count_current_len += p_embed.size(1)
                    self.start_visual_token_idx.append(count_current_len)
                    interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * vis_chunk_size:(idx + 1) * vis_chunk_size]], dim=1))
                    
                    interleave_ids += self.llama_tokenizer('<t>', add_special_tokens=False).input_ids * (int(pn/4))

                    count_current_len += vis_chunk_size
                    self.end_visual_token_idx.append(count_current_len)


            else:  
                for idx, seg in enumerate(p_segs[:-1]):  
                    p_tokens = self.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)  
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    count_current_len += p_embed.size(1)
                    self.start_visual_token_idx.append(count_current_len)
                    interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
                    count_current_len += pn
                    self.end_visual_token_idx.append(count_current_len)

            wrapped_emb = torch.cat(interleave_emb, dim=1)  


            p_tokens = self.llama_tokenizer(
                p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)  
            
            interleave_ids += p_tokens.input_ids.squeeze(0).tolist()
            cond_ids_list.append(interleave_ids)
            cond_ids_list = torch.tensor(cond_ids_list).cuda()
            
            p_embed = self.embed_tokens(p_tokens.input_ids)
            wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)  

            emb_lists.append(wrapped_emb)  

        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))  

        max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len   
        wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()  
        wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)  
        
        for i, emb in enumerate(emb_lists):
            length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len  
            wrapped_embs[i, :length] = emb[:, :length]  
            wrapped_atts[i, :length] = 1
        return wrapped_embs, wrapped_atts,cond_ids_list

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):  # input_embs [1, 163, 4096] output_embs [1, 30, 4096]
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )  # cat_embs [193, 4096]
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )  # cat_atts [193]
        cat_embs = torch.stack(cat_embs)  # [1, 193, 4096]
        cat_atts = torch.stack(cat_atts)  # [1, 193]
        return cat_embs, cat_atts, input_lens

    def tokenize_conversation(self, conv_q, conv_a):
        """concatenate conversation and make sure the model is only trained to regress the answer"""

        to_regress_token_ids_list = []
        targets_list = []

        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            questions = [self.llama_tokenizer(self.llama_tokenizer.bos_token + q,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device) for q in questions[1:]]  # the first question is handled in the prompt wrap function, skip it
            answers = [self.llama_tokenizer(a + self.end_sym,
                                            return_tensors="pt",
                                            add_special_tokens=False).to(self.device) for a in answers]
            cur_id = []
            cur_target = []
            for i in range(len(questions)):
                cur_id.append(answers[i].input_ids)
                cur_target.append(answers[i].input_ids)
                cur_id.append(questions[i].input_ids)
                cur_target.append(torch.ones_like(questions[i].input_ids) * -100)

            cur_id.append(answers[-1].input_ids)
            cur_target.append(answers[-1].input_ids)

            cur_id = torch.cat(cur_id, dim=1)
            cur_target = torch.cat(cur_target, dim=1)
            to_regress_token_ids_list.append(cur_id)
            targets_list.append(cur_target)

        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)
        to_regress_token_ids = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * self.llama_tokenizer.pad_token_id
        targets = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * -100
        for batch_idx in range(batch_size):
            cur_len = to_regress_token_ids_list[batch_idx].shape[1]
            to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[batch_idx][0, :max_len]
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]

        to_regress_token_attn = (to_regress_token_ids != self.llama_tokenizer.pad_token_id).to(torch.int)

        return to_regress_token_ids, to_regress_token_attn, targets

    def preparing_embedding(self, samples):
        
        if 'image' in samples:
            if type(samples["image"])==list:
                images = torch.stack(samples["image"]).to(self.device)
            else:
                images=samples["image"].to(self.device)

            img_embeds, img_atts = self.encode_img(images, "LR_Encoding")  

            if "HR_resolution" in samples.keys() and samples['HR_resolution'] ==True:
                HR_img_embeds, HR_img_atts = self.encode_img(samples["HR_image_list"].to(self.device), "HR_Encoding")  
                img_embeds=self.cross_attention(img_embeds.reshape(1,-1,768), HR_img_embeds.reshape(1,-1,768))  
                
            img_embeds=self.llama_proj_mlps(img_embeds)  # 1, 128, 4096


            device=images.device
        else:
            img_embeds = img_atts = None


    
        if "instruction_input" in samples:
            instruction = samples["instruction_input"] 
        
        
        img_embeds=img_embeds.reshape(1,-1,4096) 
            
            
        cond_embeds, cond_atts,_ = self.prompt_wrap(img_embeds, img_atts, instruction)  # cond_embeds [1, 163, 4096] cond_atts [1, 163]

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]  # target (self.end_sym = ###) 'This patient was diagnosed with Brain Metastases Tumor, and further diagnosed as Brain Metastases Tumor###'
        regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)  
        regress_token_ids = regress_tokens.input_ids  
        regress_atts = regress_tokens.attention_mask  
        part_targets = regress_token_ids.masked_fill(
            regress_token_ids == self.llama_tokenizer.pad_token_id, -100
        )  
        
        
        regress_embeds = self.embed_tokens(regress_token_ids)  


        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, img_embeds

    def forward(self, samples, reduction='mean'):

        if self.self_training == False:
            cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets,input_embs = \
                self.preparing_embedding(samples)  # 
        else:
            self.visual_encoder=self.visual_encoder_list[0]
            image=samples['image'].to(self.device)
            if len(image.shape) > 4:
                image = image.reshape(-1, *image.shape[-3:])
            with self.maybe_autocast():
                loss,_,_,latent=self.visual_encoder(image)
            return {"loss": loss}

        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)  

        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)  
        bos_atts = cond_atts[:, :1]  

 
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)  
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)  

        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)  

        for i, target in enumerate(part_targets):  
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  

        with self.maybe_autocast():  
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                start_visual_idx = self.start_visual_token_idx,
                end_visual_idx = self.end_visual_token_idx,
                return_dict=True,
                labels=targets,
                reduction=reduction
            )
        
        loss=outputs.loss
        return {"loss": loss}

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): 
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    
    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        num_beams=1,
        max_new_tokens=300,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,  # 0.1, 0.2, 0.4, 0.6, 0.8, 1
        do_sample=False,
        stop_words_ids=[2],
    ):
        

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        img_embeds, atts_img = self.encode_img(images.to(self.device))

        image_lists = [[image_emb[None]] for image_emb in img_embeds]

        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stopping_criteria=stopping_criteria,
            )

        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers
    
    
    def get_input_embeddings(self):
        return self.llama_model.get_input_embeddings()
    
    @torch.no_grad()
    def generate_step(
        self,
        images,
        texts,
        HR_images=None,
        num_beams=1,
        echo=False,
        max_new_tokens=500,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=-1,  # -1 0.01 0.05 0.1 0.2 0.4 0.6 0.8 1
        do_sample=False,
        stop_words_ids=[2],
        output_hidden_states=True
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        img_embeds, atts_img = self.encode_img(images.to(self.device), "LR_Encoding")

        if HR_images!=[]:
            HR_embeds,HR_atts_img = self.encode_img(HR_images.to(self.device), "HR_Encoding")
            img_embeds = self.cross_attention(img_embeds.reshape(1,-1,768).float(),HR_embeds.reshape(1,-1,768).float())
        img_embeds=self.llama_proj_mlps(img_embeds)  # 1, 128, 4096
        bsz=img_embeds.shape[0]
        img_embeds=img_embeds.reshape(1,-1,*img_embeds.shape[-1:])
        if img_embeds.size(0)>8:
            
            img_embeds=img_embeds.reshape(1,-1,4096)

        else:
            img_embeds=img_embeds.reshape(1,-1,4096)
        
        inputs_embeds,attention_mask,input_ids=self.prompt_wrap(img_embeds,atts_img,texts)

        bos = torch.ones_like(attention_mask[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)  # [1, 1, 4096]
        bos_atts = attention_mask[:, :1]  # [1, 1] 

        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)  # [1, 173, 4096]
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)  # [1, 173] 
        input_ids = torch.cat([bos, input_ids], dim=1)



        
        batch_size = len(inputs_embeds)
        
        emb_dim = inputs_embeds.shape[2]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_prompt_len = min(len(t) for t in inputs_embeds)  # 84
        max_prompt_len = max(len(t) for t in inputs_embeds)  # 84
        total_len = max_new_tokens + min_prompt_len
        
       
        self.pad_token_id=self.llama_tokenizer.pad_token_id
        pad_id = self.pad_token_id
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.int, device=device)
        embs = torch.zeros([batch_size, total_len, emb_dim], dtype=dtype, device=device)
        for k, t in enumerate(inputs_embeds):
            embs[k, : len(t)] = t
        for k, t in enumerate(input_ids):
            tokens[k, : len(t)] = t
        
        prev_pos=0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        self.stop_token_id = int(self.llama_tokenizer.encode("###")[-1])
        stop_tokens = torch.tensor([self.stop_token_id], device="cuda")
        
       
        past_key_values=None
        
        with_probs = False
        calculate_prob=False
        for cur_pos in range(min_prompt_len, total_len):
            with self.maybe_autocast():
                outputs = self.llama_model(
                    
                    past_key_values=past_key_values,
                    inputs_embeds=embs[:, prev_pos:cur_pos],
                    use_cache=True,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    start_visual_idx = self.start_visual_token_idx,
                    end_visual_idx = self.end_visual_token_idx,
                )
        
            logits = outputs["logits"]  # 1 176 128256
            past_key_values = outputs["past_key_values"]

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            import copy
            if calculate_prob:                
                probs = torch.softmax(logits[:, -1], dim=-1)
                index_list=[15,16,17,18,19,20,21,22,23,24,605,806,717,1032]
                new_logits=copy.deepcopy(logits[0][0])
                tumor_list=[new_logits[k] for k in index_list]
                tumor_list = torch.stack(tumor_list)
                probabilities = torch.softmax(tumor_list, dim=0)
                token_dict={probabilities[k]:index_list[k] for k in range(len(index_list))}  
                top_2=sorted(token_dict.keys(), reverse=True)[:2]
                # print(top_2[0].item(),top_2[1].item())
                top1=token_dict[top_2[0]]
                top2=token_dict[top_2[1]]
                print(f'<class_{self.llama_tokenizer.decode(top1)}>',top_2[0].item(),f'<class_{self.llama_tokenizer.decode(top2)}>',top_2[1].item())
                with_probs=False
                calculate_prob=False
            if next_token==449:
                with_probs=True
            if with_probs:
                if next_token==62:
                    calculate_prob=True
            
            tokens[:, cur_pos] = next_token
            next_token_embed = self.llama_model.get_input_embeddings()(next_token)
            embs[:, cur_pos] = next_token_embed

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break
        
        
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(input_ids[i])
            toks = toks[start: len(input_ids[i]) + max_new_tokens]
            probs = None
            # if logprobs:
            #     probs = token_logprobs[i][start : len(input_ids[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in [self.stop_token_id]:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    # probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
        
        answers = self.llama_tokenizer.decode(out_tokens[0])
        # print('answers',answers)
        answers=answers.split('Rating:')[0]
        tumor_special_dict = {'<class_0>': 'brain metastase tumour', '<class_1>': 'germ cell tumour', '<class_2>': 'glioneuronal and neuronal tumour', '<class_3>': 'glioma', '<class_4>': 'ependymal tumour', '<class_5>': 'meningioma', '<class_6>': 'tumors of the sellar region', '<class_7>': 'mesenchymal, non-meningothelial tumour', '<class_8>': 'cranial and paraspinal nerve tumour', '<class_9>': 'choroid plexus tumour', '<class_10>': 'hematolymphoid tumour', '<class_11>': 'embryonal tumour', '<class_12>': 'pineal tumour', '<class_13>': 'melanocytic tumour'}
        diagnosis=answers.split('with ')[-1].split('.')[0]
        answers=answers.replace(diagnosis,tumor_special_dict[diagnosis])
        return answers
    
    
    
    



    

    @torch.no_grad()
    def multi_select(self, images, texts, answers, num_cand=None):
        all_losses = []
        for answer in answers:
            choice_samples = {
                'image': images,
                'instruction_input': texts,
                'answer': answer
            }
            loss = self.forward(choice_samples, reduction='none')['loss'].reshape(-1, 1)
            all_losses.append(loss)
            torch.cuda.empty_cache()
        all_losses = torch.cat(all_losses, dim=-1)
        if num_cand is not None:
            for i in range(all_losses.shape[0]):
                all_losses[i, num_cand[i]:] = 9999
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return output_class_ranks.tolist()
    
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token