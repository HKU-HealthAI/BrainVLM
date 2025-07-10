## A Foundation Model for Presurgical Brain Tumor Diagnosis and MRI Interpretation
BrainVLM is a foundation model designed for comprehensive brain tumor analysis. It leverages multi-parametric MRI scans (T1, T1c, T2, and FLAIR sequences) and _optional_ patient metadata as input, and provides diagnosis (among 12 brain tumor categories of WHO-CNS5) and radiology report for patient.

Currently, only the evaluation code and checkpoints for brain tumor diagnosis and radiology report are available. 

Upon acceptance of the paper, we will release all models' weights and relevant source code for pretraining, fine-tuning, and uncertainty qualification for formal testing to facilitate research transparency and community collaboration.


### Installation (Linux)
1. Clone this repository and navigate to the brainvlm folder

~~~~
git clone https://github.com/HKU-HealthAI/BrainVLM.git
cd BrainVLM
conda env create -f environment.yml
~~~~

### Prepare model weights
1. Downloading Llama3.1-8B Instruct
- [Download] https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- load Llama3.1-8B-Instruct in minigpt4/configs/models/minigpt4_vicuna0.yaml: line 18, "llama_model: "

2. Download BiomedCLIP
- [Download]
https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- Brainvlm will automatically load this ckpt

3. Download BrainVLM ckpts
- Diagnosis and report ckpt: [https://drive.google.com/drive/home?dmr=1&ec=wgc-drive-hero-goto](https://drive.google.com/file/d/16yiqIvVVOANpI7OoxBKXvx5NPy0c625n/view?usp=drive_link)

- load our ckpt in eval.py

### Prepare test json files:

Prepare a json file same with example_test.json. BrainVLM supports nii.gz and npy files as input.

```
/patient1
   └── /image_list
       └── /combination_1
           ├── MRI_file_1.nii.gz
           ├── MRI_file_2.nii.gz
           ├── MRI_file_3.nii.gz
           ├── MRI_file_4.nii.gz
           └── MRI_file_5.nii.gz
        └──/combination_2
           ├── MRI_file_1.nii.gz
           ├── MRI_file_2.nii.gz
           ├── MRI_file_3.nii.gz
           ├── MRI_file_4.nii.gz
           └── MRI_file_5.nii.gz
    └──/modality_list
        └── /combination_1
           ├── modality_1
           ├── modality_2
           ├── modality_3
           ├── modality_4
           └── modality_5
        └──/combination_2
           ├── modality_1
           ├── modality_2
           ├── modality_3
           ├── modality_4
           └── modality_5
```
### Evaluation
```
python eval.py --test_json_path --model_ckpt_path
```
For evaluation, replace "test_json_path" with the path to the test JSON file and "model_ckpt_path" with the path to BrainVLM's checkpoint file.

### References
Our code builds upon MiniGPT-4 and utilizes checkpoints based on LLaMA 3.1 and BiomedCLIP. We would like to thank them.
