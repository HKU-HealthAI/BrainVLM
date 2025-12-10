

"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
import os
from collections import OrderedDict
import torch
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random
import pandas as pd
import json
import pydicom
from monai.transforms import Compose, RandRotate, RandFlip, ToTensor, EnsureType,RandSpatialCrop, Resize, Orientation, CenterSpatialCrop, ScaleIntensityRange
import torchio as tio
from collections import Counter
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler



from torchvision.transforms.functional import InterpolationMode


from scipy.interpolate import interp1d

import torch.nn.functional as F

from scipy.ndimage import zoom





def interpolate_images(images, target_count=32):
    """
    resample images to target_count

    parameters:
    - images: numpy array, shape (num_images, height, width, channels).
    - target_count: target number of images.

    return:
    - resampled_images: numpy array, shape (target_count, height, width, channels).
    """
    
    num_images = images.shape[0]
    zoom_factor = target_count / num_images

    # resample the time axis (axis=0)
    resampled_images = zoom(images, (zoom_factor, 1, 1), order=1)

    return resampled_images




def interpolation_padding(tensor, target_length=32):
    current_length = tensor.size(0)
    if current_length >= target_length:
        return tensor
    # add a batch dimension and channel dimension to use interpolate
    
    tensor = tensor.unsqueeze(0)
    
    #[32,1,256,256]
    interpolated_tensor = F.interpolate(tensor, size=(target_length, tensor.size(-2), tensor.size(-1)), mode='trilinear', align_corners=False)
    

    # remove the added dimension
    interpolated_tensor = interpolated_tensor.squeeze(0)
    return interpolated_tensor


def interpolation_padding_key_frame(tensor, target_length=32*3):
    current_length = tensor.size(0)
    if current_length >= target_length:
        return tensor
    tensor=tensor.reshape(1,-1,tensor.size(-2),tensor.size(-1))
    tensor = tensor.unsqueeze(0)#1,12,3,224,224

    interpolated_tensor = F.interpolate(tensor, size=(target_length, tensor.size(-2), tensor.size(-1)), mode='trilinear', align_corners=False)
    

    interpolated_tensor=interpolated_tensor.squeeze(0).reshape(32,3,224,224)

    return interpolated_tensor


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


def random_motion(images):
    degrees, translation = 10, 10
    B, H, W = images.shape  # batch size, height, and width
    angle = torch.randn(B).cuda() * degrees
    trans_x = torch.randn(B).cuda() * translation
    trans_y = torch.randn(B).cuda() * translation
    angle = angle * torch.pi / 180
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    affine_matrix = torch.stack([cos, -sin, zero, sin, cos, zero], dim=-1).view(B, 2, 3)
    affine_matrix[:, :, -1] = torch.stack([trans_x, trans_y], dim=-1)
    grid = F.affine_grid(affine_matrix, [B, 1, H, W], align_corners=False)
    transformed_images = F.grid_sample(images[:, None], grid, mode='nearest', align_corners=False).squeeze(1)
    return transformed_images

def random_affine_transform(images_embeddings_, degrees=10, translate=(0.1, 0.1)):
    # Randomly select angles and translation values
    angle = torch.empty(images_embeddings_.size(0)).uniform_(-degrees, degrees).cuda()
    translations = (torch.empty(images_embeddings_.size(0), 2).uniform_(-translate[0], translate[0]).cuda(),
                    torch.empty(images_embeddings_.size(0), 2).uniform_(-translate[1], translate[1]).cuda())

    # Use torchvision.transforms to apply affine transformation
    grid = F.affine_grid(
        theta=torch.zeros((images_embeddings_.size(0), 2, 3)).cuda(), 
        size=images_embeddings_.size(),
        align_corners=False
    )

    # Apply translation and rotation to image
    for i in range(images_embeddings_.size(0)):
        theta = torch.tensor([
            [torch.cos(torch.deg2rad(angle[i])), -torch.sin(torch.deg2rad(angle[i])), translations[0][i][0]],
            [torch.sin(torch.deg2rad(angle[i])), torch.cos(torch.deg2rad(angle[i])), translations[1][i][0]]
        ]).unsqueeze(0).cuda()
        grid[i] = F.affine_grid(theta, images_embeddings_[i:i+1].size(), align_corners=False)

    return F.grid_sample(images_embeddings_, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def add_random_noise(images_embeddings_, mean=0, std=0.1):
    # Randomly generate noise and add it to the image
    noise = torch.randn_like(images_embeddings_).cuda() * std + mean
    return images_embeddings_ + noise

def adjust_brightness_contrast(images_embeddings_, brightness_factor=0.5, contrast_factor=0.5):
    # Adjust brightness and contrast
    mean = images_embeddings_.mean(dim=(1, 2, 3), keepdim=True)
    images_embeddings_ = (images_embeddings_ - mean) * contrast_factor + mean  # adjust contrast
    images_embeddings_ = images_embeddings_ * brightness_factor  # adjust brightness
    return images_embeddings_

def random_brightness_tensor(image_tensor):
    """
    Apply random brightness transformation to the input image tensor
    """
    # Generate a random factor between 0.5 and 1.0
    factor = torch.empty(1).uniform_(0.5, 1.0).item()
    # Adjust image brightness
    # print(factor)
    brightened_image = torch.clamp(image_tensor * factor, 0, 1)  # Assume the tensor is in the [0, 1] range
    return brightened_image



class xiangya_training(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, ann_paths,vis_root, HR_resolution=True, image_encoder='2d'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor,ann_paths, vis_root)
        self.HR_resolution = HR_resolution
        self.image_encoder = image_encoder
        self.img_ids = {}
        n = 0
        self.image_path_list=[]

        self.anns_path=ann_paths
        print('type of anns_path',len(self.anns_path))
        i=0
        tumor_sample=True    

        self.tumor_list=[]
        if type(self.anns_path)==str:
            with open(self.anns_path,'r',encoding = 'utf-8') as f:
                self.dict=json.load(f)
            for k,v in self.dict.items():
                self.image_path_list.append(k)
        elif len(self.anns_path)>=1:  
            self.final_image_path_list=[]
            self.image_path_list=[]
            self.value_list=[]
            for json_index,path in enumerate(self.anns_path):  
                # print('path',path)
                    
                self.cur_image_path_list=[]
                self.cur_value_list=[]
                self.cur_tumor_list=[]
                with open(path,'r',encoding='utf-8') as f:
                    data=json.load(f)
                
                # germ cell, copy
                dataset_num=0
                for k,v in data.items():  
                    self.cur_image_path_list.append(k+str(i))
                    self.cur_value_list.append(v)
                    self.cur_tumor_list.append(v['Overall_class'].lower())
                    i=i+1
                print('before expansion')
                print(Counter(self.cur_tumor_list))
                if tumor_sample==True:#
                    tumor_count = Counter(self.cur_tumor_list)
                    max_samples = max(tumor_count.values())
                    expansion_factors = {tumor: min(max_samples // count,3) for tumor, count in tumor_count.items()}
                    # max(max_samples // count,,2)
                    print('expansion_factors')
                    print(expansion_factors)
                    expanded_image_list=[]
                    expanded_tumor_list=[]
                    for image, value,tumor in zip(self.cur_image_path_list, self.cur_value_list,self.cur_tumor_list):
                        expansion_factor = expansion_factors[tumor]
                        if "Germ cell tumour" in tumor:
                            expanded_image_list.extend([image] * 3)
                            expanded_tumor_list.extend([value] * 3)

                        elif "Choroid plexus" in tumor:
                            expanded_image_list.extend([image] * 4)
                            expanded_tumor_list.extend([value] * 4) # Looks useless
                        else:
                            expanded_image_list.extend([image] * expansion_factor)
                            expanded_tumor_list.extend([value] * expansion_factor)
                    self.cur_image_path_list=expanded_image_list
                    self.cur_value_list=expanded_tumor_list
                    # expanded_tumor_count = Counter(expanded_tumor_list)
                    # max_samples = max(expanded_tumor_count.values())
                self.image_path_list=self.image_path_list+self.cur_image_path_list
                self.value_list=self.value_list+self.cur_value_list
                cur_tumor_list=[]
                for k in self.value_list:
                    cur_tumor_list.append(k['Overall_class'])
                print('---------------------------')
                print('after expansion')
                print(Counter(cur_tumor_list))

                print(path,len(self.cur_image_path_list))
        
        self.dict={k:v for k,v in zip(self.image_path_list,self.value_list)}        
        random.shuffle(self.image_path_list)    

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.transform_2d = transforms.Compose(
            [
                transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.transform_HR = transforms.Compose(
            [
                transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                self.normalize,
            ]
        )
        self.downsample_list=[Compose([Resize((256,256))]),Compose([Resize((480,480))]),Compose([])] # 600+100*700
        self.our_data_resize=Compose([Resize((256,256))]) #  < 240*240
        self.transform = Compose([
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=False),
            RandFlip(spatial_axis=0, prob=0.5),   # Random horizontal flip
            RandRotate(range_x=10, prob=0.5),     # Random rotation
            # CenterSpatialCrop((224,224)),
            ToTensor(),
        ])
        self.transform_image_crop = Compose([
            CenterSpatialCrop((224,224)),
        ])
        self.motion_transform = tio.transforms.RandomMotion(
            degrees=10,
            translation=10,
            num_transforms=8,
            image_interpolation='nearest',
        )


        self.noise_transform=tio.transforms.RandomNoise(mean=0, std=0.1)
        self.intensity_transform = tio.RandomBiasField(coefficients=0.5, order=3)
        
        self.aug_transform_list = [self.motion_transform,self.noise_transform,self.intensity_transform]
        self.aug_transform_list2 = [random_affine_transform, add_random_noise, adjust_brightness_contrast]
        
        
        self.tumor_dict={}
        self.instruction_pool = [
             "There are several MRI sequence from 1 patient. Please make a detailed diagnosis for this patient including the tumor's detailed type.",
            "There are several MRI sequence from 1 patient. Please make a short diagnosis for this patient.",
            "There are several MRI sequence from 1 patient. Please make a short description of this patient, and make a detailed diagnosis for this patient including the tumor's detailed type."
        ]
    def age_group_search(self, Age, image_dict):
        # group token
        Age = float(Age)
        if 0<=Age<=20:
            Age_group = "<age_group_1>"
        elif 20<Age<=40:
            Age_group = "<age_group_2>"
        elif 40<Age<=60:
            Age_group = "<age_group_3>"
        elif 60<Age:
            Age_group = "<age_group_4>"
        else:  # if abnormal age number, output info and set to empty
            print(f"Warning! Abnormal age number:{Age}", image_dict)
            Age_group = ""
        return Age_group
    
    def age_info_extraction(self, image_dict):
        age_background=''
        if 'Sex' in image_dict.keys() and 'Age' in image_dict.keys():
            gender_probability = np.random.rand()  # random number between 0 and 1
            if gender_probability < 0.8 and image_dict['Sex']!=None and image_dict['Sex']!="None" and image_dict['Sex']!="none" and image_dict['Age']!=None and image_dict['Age']!="None" and image_dict['Age']!="none":
                Sex=image_dict['Sex'].lower()
                Age=image_dict['Age']
                try:  # if age is a normal float number, assign group token based on the number
                    Age_group = self.age_group_search(Age, image_dict)  # assign group token to Age
                except:  # if age is not a float number, add some extra rules to classify
                    if Age == "Child" or Age == "Newborn" or Age == "child":
                        Age_group = "<age_group_1>"
                    elif Age == "Young" or Age == "young":
                        Age_group = "<age_group_2>"
                    elif Age == "Middle":
                        Age_group = "<age_group_3>"
                    elif Age == "Older" or Age == "Elderly":
                        Age_group = "<age_group_4>"
                    elif Age == "Adult":  # this type of patient's age needs more judgment
                        if image_dict['Overall_class'] == "Brain Metastases Tumor" or image_dict['tumor'] == "Brain Metastases Tumor":  # if the patient is a metastasis tumor, assign to 40-60
                            Age_group = "<age_group_3>"
                        else:  # otherwise, 50% probability 20-40, 50% probability 40-60
                            age_probability = np.random.rand()
                            if age_probability < 0.5:
                                Age_group = "<age_group_2>"
                            else:
                                Age_group = "<age_group_3>"
                    else:  # handle some special cases
                        try:
                            if "-year" in Age:  # handle "65-year" and "45-year-old"
                                Age_prefix = float(Age.split("-")[0])
                                Age_group = self.age_group_search(Age_prefix, image_dict)  # assign group token to Age
                            elif "-" in Age:  # ages like "10-15", take the average of 10 and 15 as the patient's age
                                Age_1, Age_2 = Age.split("-")[0], Age.split("-")[1]
                                Age_avg = (float(Age_1) + float(Age_2))/2
                                Age_group = self.age_group_search(Age_avg, image_dict)  # assign group token to Age
                            elif "Young" in Age:  # combinations like "\"Young\"", all are assigned to 20-40
                                Age_group = "<age_group_2>"
                            elif Age == "15y":
                                Age_group = "<age_group_1>"
                            elif "~" in Age:  # ages like "5~10", take the average of 5 and 10 as the patient's age
                                Age_1, Age_2 = Age.split("~")[0], Age.split("~")[1]
                            elif "~" in Age:  # some ages are given by age groups, like "5~10", so we take the average of 5 and 10 as the patient's age
                                Age_group = self.age_group_search(Age_avg, image_dict)  # assign group token to Age
                            elif Age == "19`":
                                Age_group = "<age_group_1>"
                            else:  # if there is no way to handle the age information, output information and set to empty
                                print("age info error:", image_dict)
                                Age_group = ""
                        except:  # if there is no way to handle the age information, output information and set to empty
                            print("age info error:", image_dict)
                            Age_group = ""
                age_background=f'{Sex}, age group {Age_group}.'
            else:
                age_background='None'
        else:
            age_background='None'

        return age_background

        
    def __len__(self):
        return len(self.image_path_list)
    
    def visualizes_procesed_imgs(self, img_feature, img_path):
        for visualize_idx in range(img_feature.size(0)):
            tensor_np = img_feature[visualize_idx].cpu().numpy()
            plt.imshow(tensor_np, cmap='gray')  # cmap='gray' makes it display as a grayscale image
            plt.axis('off')  # hide the axis
            # plt.show()
            print(f"saving {img_path}/{visualize_idx}.png")
            plt.savefig(f"{img_path}/{visualize_idx}.png", bbox_inches='tight', pad_inches=0)
    
    def process_modality_name(self, modality_list):
        '''
        This method aims to standardize the representation of modalities and views, and update modality_list
        Some background knowledge:
        At the beginning, we only distinguished 4 types: t1, t2, t1c, t2f

        Some questions:
        Which is better, view or modality?
        Is t2f T2FSE or t2 flair, or both?
        '''
        final_modality_list = []
        for mod_item in modality_list:  # iterate through each modality
            item_name = ""  # record the modality and view information of the current sample
            # add modality information
            mod_item = mod_item.lower()
            # justify t1c
            if 't1 c+' in mod_item or 't1c' in mod_item  or 't1+c' in mod_item or ('t1' in mod_item and '+C' in mod_item):
                item_name += "t1c"
            # justify t1f
            elif ('t1f' in mod_item or 't1 f' in mod_item) and 'fs' not in mod_item:
                item_name += "t1f"
            # justify t1n
            elif 't1n' in mod_item:
                item_name += "t1n"
            # justify t1
            elif 't1' in mod_item:
                item_name += "t1"
            # justify t2f
            elif 't2f' in mod_item or 't2 f' in mod_item:  # how to identify t2f
                item_name += "t2f"
            # justify t2
            elif 't2' in mod_item:
                item_name += "t2"
            # not classified to above modalities
            else:
                item_name += "unk_modi"  # means unknown, if this information does not exist, automatically put it at the end
            
            item_name += " "  # leave a space
            # add view information
            # justify ax
            if 'ax' in mod_item or 'tra' in mod_item:
                item_name += "ax"
            # justify co
            elif 'co' in mod_item:
                item_name += "co"
            # justify sa
            elif 'sa' in mod_item:
                item_name += "sa"
            # not classified to above views
            else:
                item_name += "unk_view"

            final_modality_list.append(item_name)
        return final_modality_list


    def detect_modality(self,modality_list):
        t1=0
        t2=0
        t1c=0
        t2f=0
        for modality in modality_list:
            modality=modality.lower()
            if 't1' in modality and 'c+' not in modality:
                t1=1
            elif 't1' in modality and 'c+' in modality:
                t1c=1
            elif 't2f' in modality:
                t2f=1
            elif 't2' in modality:
                t2=1
        return t1,t2,t1c,t2f

    def find_same_view(self,modality_list,use_num=4):
        t1=[]
        t1c=[]
        t1_index_list=[]
        t1c_index_list=[]
        t2_index_list=[]
        other_index_list=[]
        same_view=[]
        # Separately collect t1, t1c, t2 modalities and others, including their indices and names
        for index, modality in enumerate(modality_list):
            if 't1c+' in modality:
                t1c.append(modality)
                t1c_index_list.append(index)
            elif 't1' in modality:
                t1.append(modality)
                t1_index_list.append(index)
            elif 't2' in modality:
                t2_index_list.append(index)
            else:
                other_index_list.append(index)
        # Arrange in order: t1c, t1, t2, other
        total_index_list=t1c_index_list+t1_index_list+t2_index_list+other_index_list  # Get all modality indices
        t1=set(t1)
        t1c=set(t1c)
        for sub_t1 in t1:
            if sub_t1 +'c+' in t1c:  # Check if there are t1 and t1c with the same view
                # Add t1 and t1c with the same view
                same_view.append({'t1':[i for i, x in enumerate(modality_list) if x == sub_t1],'t1c':[i for i, x in enumerate(modality_list) if x == sub_t1+'c+']})
        index_list=[]
        if same_view!=[]:  # If there are t1 and t1c with the same view
            # Select t1 and t1c of the same view, and randomly select a t2 (not including t2f) as the input modalities for the model, and store into index_list
            combination=random.choice(same_view)
            index_list.append(random.choice(combination['t1c']))
            index_list.append(random.choice(combination['t1']))  # If there is no matching modality, randomly select t1c and t1
            # if t2_index_list!=[]:
            #     index_list.append(random.choice(t2_index_list))

            # This code means to supplement the number of modalities in index_list to 4
            total_index_list=[i for i in total_index_list if i not in index_list]
            last_sample=random.sample(total_index_list,(use_num-len(index_list)))
            for sample in last_sample:
                index_list.append(sample)
            
            return index_list
        else:
            # print(t1c_index_list)
            index_list.append(random.choice(t1c_index_list))
            index_list.append(random.choice(t1_index_list))
            total_index_list=[i for i in total_index_list if i not in index_list]
            last_sample=random.sample(total_index_list,(use_num-len(index_list)))
            for sample in last_sample:
                index_list.append(sample) 
            return index_list

    def __getitem__(self, index):
        use_num=5
        image_dict=self.dict[self.image_path_list[index]]
        images=image_dict['image_list']  # all modalities of the patient
        sequence_num=min(len(images), use_num)  # minimum of 5 or less
        if 'modality_list' in image_dict.keys():
            modalities = image_dict['modality_list']  # all modalities of the patient
        elif "modality" in image_dict.keys():
            modalities = image_dict['modality']  # all modalities of the patient

        is_sort = 2  # whether to sort the sample images by modality and view, 2 using t1-t1ax corresponding
        if is_sort == 0:
            # the previous CN4 method, randomly select sequence_num images
            images = random.sample(images,sequence_num)  # randomly select sequence_num images
        elif is_sort==1:
            # select 4 images and unify the modality name
            picked_index = random.sample(list(range(len(images))),sequence_num)  # randomly get 4 indices, for model
            images = [images[i] for i in picked_index]  # select 4 images according to the random indices
            modalities = [modalities[i] for i in picked_index]  # the 4 images corresponding to the modality
            modalities = self.process_modality_name(modalities)
            
            # sort the 5 images by modalities, so that the sample images are roughly in order (currently sorted by file name)
            sorted_pairs = sorted(zip(modalities, images))  # use zip() to combine the two lists and sort by modalities
            sorted_modalities, sorted_images = zip(*sorted_pairs)  # unzip the sorted results back to two lists
            images = sorted_images
        elif is_sort == 2:
            # if len(images)>4:
            # breakpoint()
            t1,t2,t1c,t2f=self.detect_modality(modalities)
            if t1>0 and t1c>0 and len(images)>=use_num:  # if t1 and t1c exist, and the total number of modalities is greater than 4
                image_index_list=self.find_same_view(modalities,use_num)  #           
                images=[images[img_i] for img_i in image_index_list]
                modalities=[modalities[img_i] for img_i in image_index_list]
            else:
                images = random.sample(images,min(use_num,len(images)))
                picked_index = random.sample(list(range(len(images))),sequence_num)  # randomly get 4 indices, for model
                images = [images[i] for i in picked_index]  # select 4 images according to the random indices
                modalities = [modalities[i] for i in picked_index]  # the 4 images corresponding to the modality
                modalities = self.process_modality_name(modalities)
                
                # sort the 5 images by modalities, so that the sample images are roughly in order (currently sorted by file name)
                sorted_pairs = sorted(zip(modalities, images))  # use zip() to combine the two lists and sort by modalities
                sorted_modalities, sorted_images = zip(*sorted_pairs)  # unzip the sorted results back to two lists
                images = sorted_images
            #   images = random.sample(images,sequence_num)  # randomly select sequence_num images

        
        LR_image_list = []
        image_name_list = []
        HR_resolution = self.HR_resolution  # True
        HR_image_list = []
        # start_time_sample = time.time()
        # new_image_array = new_image_array[::-1]
        if_reverse=False
        aug_probability = np.random.rand()  
        if aug_probability < 0.15:
            if_reverse=True

        for image_name in images:  
            start_time_volume = time.time()
            image_name_list.append(image_name)
            image_array = np.load(os.path.join(os.path.join(self.vis_root,image_name)))  # (32, 630, 637, 3)
            if len(image_array) > 32:
                rand_index=random.randint(0,len(image_array)-32)
                image_array=image_array[rand_index:rand_index+32]     

            if if_reverse:
                image_array=image_array[::-1]
            # start_time_volume_LR = time.time()

            image_sequence=[]
            HR_image_sequence = []  # Store HR patches for 32 slices, total 128
            image_shape = image_array[0].shape  # (630, 637, 3)
            # convert to grayscale
            for image in image_array:  # image_array (32, 630, 637, 3), each image is 630, 637, 3
                if self.image_encoder=='2d':
                    image = Image.fromarray(image).convert('L')  # image PIL object
                    image_sequence.append(image)

            if self.image_encoder=='2d':
                image_sequence = np.stack(image_sequence, axis=0)  # convert to numpy array, len(image_sequence) 32, (32, 630, 637)
                
                image_sequence = self.transform(np.expand_dims(image_sequence, axis=1))  # [32, 1, 630, 637]

                aug_version = 3  # or 2  (1 means using previous tio library without GPU support, 2 supports GPU simulation method, 3 mixes 1 and 2)
                is_visualize = 0  # or 0 

                image_sequence=image_sequence.reshape(1,32,image_sequence.shape[-2],image_sequence.shape[-1])
                image_sequence=image_sequence.squeeze()
                # image_sequence = self.transform_image_crop(image_sequence)  # [32, 256, 256] -> [32, 224, 224]

                resize_num = random.randint(0,2)  # set to 0 or 1 randomly
                # print(image_shape)
                if image_shape[0] < 255 or image_shape[1] < 255:
                    image_sequence = self.our_data_resize(image_sequence)  # resize to 256 torch.Size([32, 256, 256])
                else:
                    image_sequence = self.downsample_list[resize_num](image_sequence)
                # print(image_sequence.shape)
                image_sequence=image_sequence.unsqueeze(0)
                
                aug_probability = np.random.rand()  
                if aug_probability < 0.3:
                    if aug_version == 1:
                        # The data augmentation used in previous code is the tio library, which does not support GPU
                        aug_transform = random.sample(self.aug_transform_list, 1)[0]
                        image_sequence = aug_transform(image_sequence).squeeze(1)  # 32, 630, 637
                        image_sequence = image_sequence.unsqueeze(1)  # 32, 1, 630, 637
                    elif aug_version == 2:
                        # Simulate tio's data augmentation method to support GPU
                        image_sequence = image_sequence.cuda()  
                        aug_transform = random.sample(self.aug_transform_list2, 1)[0]
                        image_sequence = aug_transform(image_sequence).squeeze(1)
                    elif aug_version == 3:  
                        aug_method_idx = random.randint(0, 3)  # 0, 1, 2
                        if aug_method_idx == 0:  # tio's motion blur
                            image_sequence = self.motion_transform(image_sequence).squeeze(1)
                        elif aug_method_idx == 1:  # manually written random noise
                            image_sequence = add_random_noise(image_sequence.cuda()).squeeze(1)
                        elif aug_method_idx == 2:  # tio's intensity_transform
                            image_sequence = self.intensity_transform(image_sequence).squeeze(1)
                        elif aug_method_idx ==3:
                            image_sequence = random_brightness_tensor(image_sequence).squeeze(1)
                image_sequence = image_sequence.squeeze().cuda()  # torch.Size([32, 630, 637])    

               
                image_sequence_LR = self.transform_image_crop(image_sequence)  # [32, 256, 256] -> [32, 224, 224]

                
                image_sequence_LR = image_sequence_LR.repeat(3,1,1).view(3, -1, 224, 224).permute(1, 0, 2, 3)  # Target [32, 224, 224] -> [96, 224, 224] -> [3, 32, 224, 224] -> [3, 32, 224, 224]
                LR_image_list.append(image_sequence_LR)

                

                #### Process HR images ####
                if HR_resolution == True:
                    # Iterate over each image
                    # image_sequence=self.downsample_list[resize_num](image_sequence)
                    num_images, LR_h, LR_w = image_sequence.shape  # ([32, 630, 637])
                    
                
                    
                    
                    for image_idx in range(num_images):
                        # Slice each image [630, 637] 
                        HR_h, HR_w = LR_h // 2, LR_w // 2  # HR target size for each block HR_h 315  HR_w 318
                        HR_image_1 = image_sequence[image_idx][:HR_h, :HR_w]  # lefttop 315, 318
                        HR_image_2 = image_sequence[image_idx][HR_h:LR_h, :HR_w]  # leftbottom 315, 318
                        HR_image_3 = image_sequence[image_idx][:HR_h, HR_w:LR_w]  # righttop 315, 319
                        HR_image_4 = image_sequence[image_idx][HR_h:LR_h, HR_w:LR_w]  # rightbottom 315, 319
                    
                    
                        

                        HR_image_1 = F.interpolate(HR_image_1.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)
                        HR_image_2 = F.interpolate(HR_image_2.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)
                        HR_image_3 = F.interpolate(HR_image_3.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)
                        HR_image_4 = F.interpolate(HR_image_4.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).reshape(224, 224)

                        HR_images = torch.stack([HR_image_1, HR_image_2, HR_image_3, HR_image_4], dim=0)  # 4, 224, 224  all HR patches in source images
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
            
       
        instruction_index = 1
        instruction=self.instruction_pool[instruction_index]
        age_background=self.age_info_extraction(image_dict)
        
        if image_dict['Overall_special_token']=='<class_2>' or image_dict['Overall_special_token']=='<class_4>':
            image_dict['Overall_special_token']='<class_3>'

        is_split_volume_in_instruction = 2  # 
        if is_split_volume_in_instruction == 0:
            # use the original instruction
            instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)  
        elif is_split_volume_in_instruction == 1:
            # modify the instruction after splitting the volume  把图像用<t>分隔开
            # build the prompt for the right example: <Img><ImageHere><t><ImageHere><t><ImageHere><t><ImageHere></Img>
            modality_num = sequence_num  # the total number of volumes, usually 5
            image_prompt = ""
            for img_idx in range(modality_num):  
                image_prompt += '<ImageHere>'
                if img_idx < sequence_num - 1:  # the last <ImageHere> is not followed by a separator <t>
                    image_prompt += '<t>'
            instruction = "<Img>{}</Img> [caption] {} ".format(image_prompt, instruction)   
        elif is_split_volume_in_instruction == 2:
            # build the prompt for the right example: <Img>t1c co:<ImageHere><t>t1c sa:<ImageHere><t>t2 ax:<ImageHere><t>unk_modi ax:<ImageHere></Img> [caption] There are several MRI sequence from 1 patient. Please make a detailed diagnosis for this patient including the tumor's detailed type. 
            modality_num = sequence_num  # the total number of volumes, usually 5
            image_prompt = ""
            for img_idx in range(modality_num):  
                image_prompt += f' {modalities[img_idx]}:<ImageHere>'
                if img_idx < sequence_num - 1:  # the last <ImageHere> is not followed by a separator <t>
                    image_prompt += '<t>'
            instruction = "<Img>{}</Img> [caption] {} ".format(image_prompt, instruction)   
        # print(instruction)
        #
        if 'Overall_class' in image_dict.keys() and instruction_index==0:  # if there is tumor label and the first instruction —— There are several MRI sequence from 1 patient. Please make a detailed diagnosis for this patient including the tumor's detailed type.
            report='This patient was diagnosed with {}, and further diagnosed as {}'.format(image_dict['Overall_special_token'],image_dict['tumor'])
        elif 'Overall_class' in image_dict.keys() and instruction_index==1:    # if there is tumor label and the second instruction —— There are several MRI sequence from 1 patient. Please make a short diagnosis for this patient.
            report='This patient was diagnosed with {}'.format(image_dict['Overall_special_token'])
        elif 'Overall_class' in image_dict.keys() and 'location' in image_dict.keys() and instruction_index==2:  # if there is tumor label and location description, and the third instruction —— There are several MRI sequence from 1 patient. Please make a short description of this patient, and make a detailed diagnosis for this patient including the tumor's detailed type.
            report='Tumor is located in {}. '.format(image_dict['location']) +'The patient is likely diagnosed with {}'.format(image_dict['Overall_class'])

        
        if "Overall_class" in image_dict.keys():
            Overall_class=image_dict['Overall_special_token']
        else:
            Overall_class=[]
        
        prompt=f"You are a language vision assistant. Based on the MRI sequences, question, context, please give an appropriate tumor choice among WHO tumor classification label. USER:{instruction} Context:This is WHO tumor classification label: 'brain metastase tumour': '<class_0>', 'germ cell tumour': '<class_1>', 'glioma': '<class_3>','meningioma': '<class_5>', 'tumors of the sellar region': '<class_6>', 'mesenchymal, non-meningothelial tumour': '<class_7>', 'cranial and paraspinal nerve tumour': '<class_8>', 'choroid plexus tumour': '<class_9>', 'hematolymphoid tumour': '<class_10>', 'embryonal tumour': '<class_11>', 'pineal tumour': '<class_12>', 'melanocytic tumour': '<class_13>',{age_background}, ASSISTANT:"

        
        return {
            "image_name":image_name_list,
            "image": LR_image_list.cpu(),  # need to be transferred to cpu, otherwise the loader will report an error
            "answer": report,
            "Overall_class": Overall_class,
            "instruction_input": prompt,
            "HR_image_list":HR_image_list.cpu(),  # need to be transferred to cpu, otherwise the loader will report an error
            "HR_resolution":HR_resolution
        }
