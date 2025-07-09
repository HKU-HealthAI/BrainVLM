import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase





@registry.register_model("minigpt4")
class MiniGPT4(MiniGPTBase):
    """
    MiniGPT-4 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
            self,
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=False,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            modality_number=1,
            model_2d_or_3d="2d",
            self_training=False
    ):
        super().__init__(
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            modality_number=modality_number,
            model_2d_or_3d=model_2d_or_3d,
            self_training=self_training
        )
        if self.model_2d_or_3d=='2d':
            self.v_q_project=nn.Linear(self.visual_encoder.num_features,1408)
        elif self.model_2d_or_3d=='3d':
            self.v_q_project=nn.Linear(self.visual_encoder.hidden_size,1408)

        self.self_training=self_training
        self.modality_number=modality_number
        img_f_dim=self.visual_encoder.num_features         
       
        if self.self_training:
            for name,param in self.bounding_box_layer.named_parameters():
                param.requires_grad = False
            print('freeze head params')


    

    def hook_for_vit_features(self, image):
        
        activations = {}
        
        def hook_fn(module, input, output):
            activations[module] = output
        
        hook_handle_norm = self.visual_encoder.norm.register_forward_hook(hook_fn)  
        
        latent = self.visual_encoder(image)  
        
        image_token_embeds = activations[self.visual_encoder.norm][:, 1:, :]  
       
        hook_handle_norm.remove()
        activations = {}
        return image_token_embeds
    
    def encode_img(self, image, resolution_type, modality=['t1n']):

        
        device = image.device
        
        if self.model_2d_or_3d=='2d':  
            if len(image.shape) > 4:
                image = image.reshape(-1, *image.shape[-3:])  
        elif self.model_2d_or_3d=='3d':
            if len(image.shape) > 4:
                image = image.reshape(-1, 1, 32,*image.shape[-2:])
        with self.maybe_autocast():

            if self.model_2d_or_3d=='3d':
                latent_list=[]
                for i in range(len(image)):
                    sub_image=image[i].reshape(-1,1,32,256,256)
                    latent,_=self.visual_encoder(sub_image.to(device))
                    latent=latent[:,:-1,:]
                    image_embeds=latent
                    image_embeds=self.projector(image_embeds)
                    latent_list.append(image_embeds)

                latent_list=torch.stack(latent_list,dim=0)
                latent_list=latent_list.reshape(-1,latent_list.size(-1))
                inputs_llama=self.llama_proj(latent_list)
            else:
                b,c,w,h=image.shape  
                if b < 256:
                    latent = self.hook_for_vit_features(image.to(device))
                    if resolution_type == "LR_Encoding": 
                        latent = latent.mean(dim=1)  
                    elif resolution_type == "HR_Encoding": 
                        latent = latent
                    image_embeds=latent  
                else:
                    latent_list=[]
                    sub_batch=int(b/2)
                    for i in range(0,b,sub_batch):
                        sub_image=image[i:i+sub_batch]

                        latent = self.hook_for_vit_features(sub_image.to(device))
                        if resolution_type == "LR_Encoding": 
                            latent = latent.mean(dim=1) 
                        elif resolution_type == "HR_Encoding": 
                            latent = latent
                        
                        latent_list.append(latent)
                    latent_list = torch.stack(latent_list, dim=0)
                    image_embeds = latent_list.reshape(-1, latent_list.size(-1))

                

               
            if self.model_2d_or_3d=='2d':
                image_embeds=image_embeds.view(image_embeds.size(0), 1, -1)  # [128, 1, 768]

        
            
            if self.model_2d_or_3d=='2d':  
                inputs_llama = image_embeds
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)  # [128,1]
        return inputs_llama, atts_llama

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        model_2d_or_3d=cfg.get("model_2d_or_3d", '2d')
        model = cls(
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            model_2d_or_3d=model_2d_or_3d
        )

        ckpt_path = cfg.get("ckpt", "")  
        if ckpt_path:
            print("Load MiniGPT-4 Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            state_dict_a = ckpt['model']
            state_dict_b = model.state_dict()
            for name_b, param_b in state_dict_b.items():
                if name_b in state_dict_a:
                    param_a = state_dict_a[name_b]
                    if param_a.shape == param_b.shape:
                        state_dict_b[name_b].copy_(param_a)
                    else:
                        print(f"Shape mismatch at layer: {name_b}, cannot transfer weights.")
                else:
                    pass
            msg = model.load_state_dict(state_dict_b, strict=False)

        return model
