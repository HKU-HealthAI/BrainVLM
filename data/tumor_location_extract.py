import argparse
import json
from openai import OpenAI
import os
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('input_path')
parser.add_argument('output_path')
parser.add_argument('img_folder')
parser.add_argument('template_folder')
args = parser.parse_args()
dataset = args.dataset
input_path = args.input_path                    ###input json path
output_path = args.output_path                  ###output json path
img_folder = args.img_folder                    ###image(MRI and segment) nii.gz folder path
template_folder = args.template_folder          ###image(template registration result) nii.gz folder path

def judge(direction, numid):
    flag1, flag2, flag3 = 0, 0, 0
    if direction[numid]=='A' or direction[numid]=='P':
        if (numid==0 and (direction[numid+1]=='L' or direction[numid+1]=='R')) or \
           (numid==1 and (direction[numid-1]=='L' or direction[numid-1]=='R')) or \
           (numid==2 and (direction[numid-2]=='L' or direction[numid-2]=='R')):
            flag1=1
        if str(direction).find('S')!=-1:
            flag2=1
        if str(direction).find('R')!=-1:
            flag3=1
    if direction[numid]=='R' or direction[numid]=='L':
        if (numid==0 and (direction[numid+1]=='A' or direction[numid+1]=='P')) or \
           (numid==1 and (direction[numid-1]=='A' or direction[numid-1]=='P')) or \
           (numid==2 and (direction[numid-2]=='A' or direction[numid-2]=='P')):
            flag1=1
        if str(direction).find('S')!=-1:
            flag2=1
        if str(direction).find('A')!=-1:
            flag3=1
    if direction[numid]=='S' or direction[numid]=='I':
        if (numid==0 and (direction[numid+1]=='L' or direction[numid+1]=='R')) or \
           (numid==1 and (direction[numid-1]=='L' or direction[numid-1]=='R')) or \
           (numid==2 and (direction[numid-2]=='L' or direction[numid-2]=='R')):
            flag1=1
        if str(direction).find('A')!=-1:
            flag2=1
        if str(direction).find('R')!=-1:
            flag3=1
    return flag1, flag2, flag3

loc1={
    'Frontal Lobe': (1, 68), 'Temporal Lobe': (69, 124), 'Parietal Lobe': (125, 162), 'Insular Lobe': (163, 174), 
    'Limbic Lobe': (175, 188), 'Occipital Lobe': (189, 210), 'Subcortical Nuclei': (211, 246)
}

def init():
    count={
        'Frontal Lobe': 0, 'Temporal Lobe': 0, 'Parietal Lobe': 0, 'Insular Lobe': 0, 
        'Limbic Lobe': 0, 'Occipital Lobe': 0, 'Subcortical Nuclei': 0
    }
    return count


flag=0
with open(path, 'r') as f:
    lines=f.readlines()
    for line in lines:
        obj=json.loads(line)
        if len(obj['box'])==0:
            continue
        boxes=[]
        for box in obj['box']:
            if box[0]<box[2]:
                box[0], box[2] = box[2], box[0]
            if box[1]<box[3]:
                box[1], box[3] = box[3], box[1]
        maxx, maxy, minx, miny=-1e10, -1e10, 1e10, 1e10
        for box in obj['box']:
            maxx=max(maxx,box[0])
            maxy=max(maxy,box[1])
            minx=min(minx,box[2])
            miny=min(miny,box[3])
        obj['box']=[[maxx, maxy, minx, miny]]
        
        idx=obj['figure'].split('_')[0]
        
        modality=obj['modality']
        if modality.find('t2f')!=-1:
            modality='t2f'
        elif modality.find('t2w')!=-1:
            modality='t2w'
        elif modality.find('t1n')!=-1:
            modality='t1n'
        if modality.find('t1c')!=-1:
            modality='t1c'
            
        figure=idx+'-'+modality+'.nii.gz'
        imgpath=os.path.join(img_folder, idx)
        tppath=os.path.join(template_folder,idx+'-'+modality+'-atlas.nii.gz')
        
        img = nib.load(os.path.join(imgpath, figure))
        direction=nib.aff2axcodes(img.affine)
        image_data = img.get_fdata()

        template = nib.load(tppath)
        template_data = template.get_fdata()

        if obj['idx']==0:
            imgbl=image_data[obj['num'], :, :]
            imgtp=template_data[obj['num'], :, :]
        elif obj['idx']==1:
            imgbl=image_data[:, obj['num'], :]
            imgtp=template_data[:, obj['num'], :]
        else:
            imgbl=image_data[:, :, obj['num']]
            imgtp=template_data[:, :, obj['num']]
        flag1, flag2, flag3 = judge(direction, obj['idx'])
        if flag1:
            imgbl=imgbl.transpose((1, 0))
            imgtp=imgtp.transpose((1, 0))
        if flag2:
            imgbl=imgbl[::-1, :]
            imgtp=imgtp[::-1, :]
        if flag3:
            imgbl=imgbl[:, ::-1]
            imgtp=imgtp[:, ::-1]

        segment=idx+'-seg.nii.gz'

        seg=nib.load(os.path.join(imgpath, segment))
        seg_data = seg.get_fdata()
        if obj['idx']==0:
            seg_bl=seg_data[obj['num'], :, :]
        elif obj['idx']==1:
            seg_bl=seg_data[:, obj['num'], :]
        else:
            seg_bl=seg_data[:, :, obj['num']]
        if flag1:
            seg_bl=seg_bl.transpose((1, 0))
        if flag2:
            seg_bl=seg_bl[::-1, :]
        if flag3:
            seg_bl=seg_bl[:, ::-1]

        cnt = init()
        total=0
        for i in range(len(seg_bl)):
            for j in range(len(seg_bl[i])):
                if seg_bl[i][j] and imgtp[i][j]:
                    total+=1
                    for location in loc:
                        if int(imgtp[i][j])>=loc[location][0] and int(imgtp[i][j])<=loc[location][1]:
                            cnt[location]+=1
        ff1=0
        tumor_location=''
        for location in cnt:
            if cnt[location]/total>=0.5:
                if len(tumor_location):
                    tumor_location+=', '
                ff1=1
                tumor_location+=location
        if ff1==0:
            for location in cnt:
                if cnt[location]/total>=0.35:
                    if len(tumor_location):
                        tumor_location+=', '
                    ff1=1
                    tumor_location+=location
        dictionary=obj
        dictionary['tumor location']=tumor_location
        with open(jsonpath, 'a') as ff:
            json.dump(dictionary,ff)
            ff.write('\n')
