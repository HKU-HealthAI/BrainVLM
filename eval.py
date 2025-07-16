import os
import re
import json
import argparse
from scipy.ndimage import zoom
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='BrainVLM Evaluation')
parser.add_argument('--test_json_path', type=str, required=True, help='Path to the test json file')
parser.add_argument('--model_ckpt_path ', type=str, required=True, help='Path to the model checkpoint')
args = parser.parse_args()
json_file = args.test_json_path
model_ckpt = args.model_ckpt_path 


import random
import numpy as np
from PIL import Image
import torch
import minigpt4.tasks as tasks
import torchvision.transforms as transforms
from monai.transforms import Compose, RandRotate, RandFlip, ToTensor,EnsureType,RandSpatialCrop,Resize,Orientation, CenterSpatialCrop,ScaleIntensityRange

from minigpt4.tasks import *
from minigpt4.processors import *
from torchvision.transforms.functional import InterpolationMode
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"

from collections import Counter
import torch.nn.functional as F

from minigpt4.common.config import Config

def interpolate_images(images, target_count=32):

    
    num_images = images.shape[0]
    zoom_factor = target_count / num_images

    resampled_images = zoom(images, (zoom_factor, 1, 1), order=1)

    return resampled_images



def list_of_str(arg):
    return list(map(str, arg.split(',')))

torch.set_num_threads(4)

def process_modality_name(modality_list):
        final_modality_list = []
        for mod_item in modality_list:  
            item_name = ""  
            mod_item = mod_item.lower()
            if 't1 c+' in mod_item or 't1c' in mod_item  or 't1+c' in mod_item or ('t1' in mod_item and '+C' in mod_item):
                item_name += "t1c"
            elif ('t1f' in mod_item or 't1 f' in mod_item) and 'fs' not in mod_item:
                item_name += "t1f"
            elif 't1n' in mod_item:
                item_name += "t1n"
            elif 't1' in mod_item:
                item_name += "t1"
            elif 't2f' in mod_item or 't2 f' in mod_item:
                item_name += "t2f"
            elif 't2' in mod_item:
                item_name += "t2"
            else:
                item_name += "unk_modi"  
            item_name += " "  
            if 'ax' in mod_item:
                item_name += "ax"
            elif 'co' in mod_item:
                item_name += "co"
            elif 'sa' in mod_item:
                item_name += "sa"
            else:
                item_name += "unk_view"
            final_modality_list.append(item_name)
        return final_modality_list


class Args:
    def __init__(self):
        self.cfg_path = "eval_configs/3d_diangosis.yaml"
        self.options = None

parser = argparse.ArgumentParser(description="Training")


args = Args()
cfg = Config(args)


task = tasks.setup_task(cfg)

transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
            ]
        )
device=torch.device('cuda')



k_list=[]
v_list=[]
with open(json_file,'r') as file:
    data=json.load(file)

        
    
instruction_pool = [
            "There are several MRI sequence from 1 patient. Please make a short diagnosis for this patient.",
        ]
    
transform = Compose([
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=False),
            ToTensor(),
        ])
our_data_resize=Compose([Resize((224,224))]) #  < 240*240
transform_image_crop = Compose([
            CenterSpatialCrop((224,224)),
        ])        
image_encoder='2d'


def age_group_search(Age, image_dict):
        Age = float(Age)
        if 0<=Age<=20:
            Age_group = "<age_group_1>"
        elif 20<Age<=40:
            Age_group = "<age_group_2>"
        elif 40<Age<=60:
            Age_group = "<age_group_3>"
        elif 60<Age:
            Age_group = "<age_group_4>"
        else:  
            print(f"Warning! Abnormal age number:{Age}", image_dict)
            Age_group = ""
        return Age_group
    
def age_info_extraction(image_dict):
    age_background=''
    if 'Sex' in image_dict.keys() and 'Age' in image_dict.keys():
        gender_probability = np.random.rand()  
        if gender_probability < 1 and image_dict['Sex']!=None and image_dict['Sex']!="None" and image_dict['Sex']!="none" and image_dict['Age']!=None and image_dict['Age']!="None" and image_dict['Age']!="none":
            Sex=image_dict['Sex'].lower()
            Age=image_dict['Age']
            try:  
                Age_group = age_group_search(Age, image_dict)  
            except:  
                if Age == "Child" or Age == "Newborn" or Age == "child":
                    Age_group = "<age_group_1>"
                elif Age == "Young" or Age == "young":
                    Age_group = "<age_group_2>"
                elif Age == "Middle":
                    Age_group = "<age_group_3>"
                elif Age == "Older" or Age == "Elderly":
                    Age_group = "<age_group_4>"
                elif Age == "Adult":  
                    if image_dict['Overall_class'] == "Brain Metastases Tumor" or image_dict['tumor'] == "Brain Metastases Tumor":  
                        Age_group = "<age_group_3>"
                    else:  
                        age_probability = np.random.rand()
                        if age_probability < 0.5:
                            Age_group = "<age_group_2>"
                        else:
                            Age_group = "<age_group_3>"
                else:  
                    try:
                        if "-year" in Age:  
                            Age_prefix = float(Age.split("-")[0])
                            Age_group = age_group_search(Age_prefix, image_dict) 
                        elif "-" in Age:  
                            Age_1, Age_2 = Age.split("-")[0], Age.split("-")[1]
                            Age_avg = (float(Age_1) + float(Age_2))/2 
                            Age_group = age_group_search(Age_avg, image_dict)  
                        elif "Young" in Age:  
                            Age_group = "<age_group_2>"
                        elif Age == "15y":
                            Age_group = "<age_group_1>"
                        elif "~" in Age:  
                            Age_1, Age_2 = Age.split("~")[0], Age.split("~")[1]
                            Age_avg = (float(Age_1) + float(Age_2))/2
                            Age_group = age_group_search(Age_avg, image_dict)  
                        elif Age == "19`":
                            Age_group = "<age_group_1>"
                        else: 
                            print("age info error:", image_dict)
                            Age_group = ""
                    except:  
                        print("age info error:", image_dict)
                        Age_group = ""
            age_background=f'{Sex}, age group {Age_group}.'
        else:
            age_background='None'
    else:
        age_background='None'

    return age_background


def images_process(images,image_dict,modalities=None):#get_item
    
    sequence_num=len(images)
    
    LR_image_list = []
    image_name_list = []
    HR_resolution = True  
    HR_image_list = []
    for image_name in images: 
        image_name_list.append(image_name)
        if 'npy' in image_name:
            image_array = np.load(image_name)  # (32, 630, 637, 3)
            image_array=interpolate_images(image_array,32)
        elif 'nii.gz' in image_name:
            nii_img = nib.load(image_name)
            image_array = nii_img.get_fdata()
            image_array = image_array.astype(np.uint8)
            image_array=interpolate_images(image_array,32)


        image_sequence=[]
        HR_image_sequence = []  
        image_shape = image_array[0].shape 

        for image in image_array:  
            if image_encoder=='2d':
                image = Image.fromarray(image).convert('L')  
                image_sequence.append(image)
                
        if image_encoder=='2d':
            image_sequence = np.stack(image_sequence, axis=0)  # 转np array len(image_sequence) 32    (32, 630, 637)
                

            image_sequence = transform(np.expand_dims(image_sequence, axis=1))  # [32, 1, 630, 637]

            aug_version = 2  # or 2  

                
            aug_probability = np.random.rand() 
            
            image_sequence = image_sequence.squeeze().cuda()  # torch.Size([32, 630, 637])
                        
            image_sequence_LR = our_data_resize(image_sequence)  # [32, 256, 256] -> [32, 224, 224]
            
 
              
            image_sequence_LR = image_sequence_LR.repeat(3,1,1).view(3, -1, 224, 224).permute(1, 0, 2, 3)  # Target [32, 224, 224] -> [96, 224, 224] -> [3, 32, 224, 224] -> [3, 32, 224, 224]
            LR_image_list.append(image_sequence_LR)


               
            HR_resolution=True
            if HR_resolution == True:
                
                num_images, LR_h, LR_w = image_sequence.shape  # ([32, 630, 637])
                for image_idx in range(num_images):

                    HR_h, HR_w = LR_h // 2, LR_w // 2  # HR  HR_h 315  HR_w 318
                    HR_image_1 = image_sequence[image_idx][:HR_h, :HR_w]  #  315, 318
                    HR_image_2 = image_sequence[image_idx][HR_h:LR_h, :HR_w]  #  315, 318
                    HR_image_3 = image_sequence[image_idx][:HR_h, HR_w:LR_w]  #  315, 319
                    HR_image_4 = image_sequence[image_idx][HR_h:LR_h, HR_w:LR_w]  #  315, 319
                        
                    HR_image_1 = F.interpolate(HR_image_1.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)
                    HR_image_2 = F.interpolate(HR_image_2.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)
                    HR_image_3 = F.interpolate(HR_image_3.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)
                    HR_image_4 = F.interpolate(HR_image_4.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)

                    HR_images = torch.stack([HR_image_1, HR_image_2, HR_image_3, HR_image_4], dim=0)  # 4, 224, 224  source图里的全部HR patch
                    HR_image_sequence += HR_images
            HR_image_sequence = torch.stack(HR_image_sequence, dim=0)
            HR_image_sequence = HR_image_sequence.repeat(3,1,1).view(3, -1, 224, 224).permute(1, 0, 2, 3)  # [128, 3, 224, 224]
            HR_image_list.append(HR_image_sequence)


            
    LR_image_list = torch.stack(LR_image_list,dim=0).cuda()  # torch.Size([7, 32, 3, 224, 224])

    if HR_resolution==True and HR_image_list!=[]:
        HR_image_list = torch.stack(HR_image_list, dim=0).cuda()  # torch.Size([7, 128, 3, 224, 224])
    else:
        HR_resolution=False
        HR_image_list = []
    

    instruction=instruction_pool[0]  # 0-163,1-154
    
    age_select=1
    age_background=''
    if age_select==0:
        if 'Sex' in image_dict.keys() and 'Age' in image_dict.keys():
            gender_probability = random.randint(0,1)
            gender_probability=1

            if gender_probability==1 and image_dict['Sex']!=None and image_dict['Age']!=None:
                Sex=image_dict['Sex']
                Age=image_dict['Age']
                age_background=f'There are several MRI sequence from one {Sex} patient, {Age} years old, '
            else:
                age_background='There are several MRI sequence from one patient, '
        else:
            age_background='There are several MRI sequence from one patient, '
    else:
        age_background=age_info_extraction(image_dict)

    is_split_volume_in_instruction = 2  # 0 
    if is_split_volume_in_instruction == 0:
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)  
    
    elif is_split_volume_in_instruction == 1:
        
        modality_num = sequence_num  # 
        image_prompt = ""
        for img_idx in range(modality_num):  # prompt: <Img><ImageHere><t><ImageHere><t><ImageHere><t><ImageHere></Img>
            image_prompt += '<ImageHere>'
            if img_idx < sequence_num - 1:  
                image_prompt += '<t>'
        instruction = "<Img>{}</Img> [caption] {} ".format(image_prompt, instruction)            
    elif is_split_volume_in_instruction == 2:
        modality_num = sequence_num  
        image_prompt = ""
        for img_idx in range(modality_num):  
            image_prompt += f' {modalities[img_idx]}:<ImageHere>'
            if img_idx < sequence_num - 1:  
                image_prompt += '<t>'
        instruction = "<Img>{}</Img> [caption] {} ".format(image_prompt, instruction) 
    
    
    # prompt for tumor classification and report generation
    prompt=f"You are a language vision assistant. Based on the MRI sequences, question, context, please make a brief description for the patient include tumor's location (Note: In the AX view, the left and right of the MRI images are opposite to the left and right of the picture.) and intensity of MRI signal and other important information. Besides, giving an appropriate tumor choice among WHO tumor classification label. USER:{instruction} Context:This is WHO tumor classification label: 'brain metastase tumour': '<class_0>', 'germ cell tumour': '<class_1>', 'glioneuronal and neuronal tumour': '<class_2>', 'glioma': '<class_3>','ependymal tumour':'<class_4>','meningioma': '<class_5>', 'tumors of the sellar region': '<class_6>', 'mesenchymal, non-meningothelial tumour': '<class_7>', 'cranial and paraspinal nerve tumour': '<class_8>', 'choroid plexus tumour': '<class_9>', 'hematolymphoid tumour': '<class_10>', 'embryonal tumour': '<class_11>', 'pineal tumour': '<class_12>', 'melanocytic tumour': '<class_13>',{age_background}, ASSISTANT:"


    instruction=[prompt]

    return instruction, LR_image_list, HR_image_list


cfg.model_cfg.ckpt=model_ckpt
model = task.build_model(cfg).to(device)
model.eval()

step1_list=[]
step2_list=[]
step3_list=[]

for k,v in data.items():
    image_list=v['image_list']
    modality_list=v['modality']
    combination_diagnosis=[]
    combination_list=[]
    for comb_idx, (image_combination,modality_combination) in enumerate(zip(image_list,modality_list)):
        instruction,image_list,HR_image_list=images_process(image_combination,v,modality_combination)        
        answer=model.generate_step(image_list,instruction,HR_image_list)        
        combination_list.append(answer.split('This paitent')[0])
        combination_diagnosis.append(answer.split('.')[-2])

    if len(combination_diagnosis) > 0:
        from collections import Counter
        most_common_diagnosis = Counter(combination_diagnosis).most_common(1)[0][0]
        final_report = max(combination_list, key=len)
        print(f"Final diagnosis: {most_common_diagnosis}")
        print(f"Final report: {final_report}")
        
        

        
