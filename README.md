## A Foundation Model for Presurgical Brain Tumor Diagnosis and MRI Interpretation
BrainVLM is a foundation model designed for comprehensive brain tumor analysis. It leverages multi-parametric MRI scans (T1, T1c, T2, and FLAIR sequences) and _optional_ patient metadata as input, and provides diagnosis (among 12 brain tumor categories of WHO-CNS5) and radiology report for patient.

Currently, only the evaluation code and checkpoints for brain tumor diagnosis and radiology report are available. 

Upon acceptance of the paper, we will release all models' weights and relevant source code for pretraining, fine-tuning, and uncertainty qualification for formal testing to facilitate research transparency and community collaboration.


### Installation (Linux)
1. Clone this repository and navigate to the brainvlm folder

~~~~
git clone https://github.com/HKU-HealthAI/BrainVLM.git
cd BrainVLM
conda env create -f brainvlm_foundation.yml
~~~~

### Prepare model weights
1. Downloading Llama3.1-8B Instruct
- [Download] https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- load Llama3.1-8B-Instruct in minigpt4/configs/models/minigpt4_vicuna0.yaml: line 18, "llama_model: "

2. Download BiomedCLIP
- [Download]
https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- Brainvlm will automatically load this ckpt, do not need to indicate the path of BiomedCLIP in our code.

3. Download BrainVLM ckpts
- Diagnosis and report ckpt:  https://drive.google.com/drive/home?dmr=1&ec=wgc-drive-hero-goto

### Prepare test json files:

Prepare a json file same with example_test.json. BrainVLM supports nii.gz and npy files as input.
#### Patient Example in example_test.json:
For patient1, it includes 6 MRI sequences stored in the directory structure shown below. Each file corresponds to a specific MRI modality.
```
This is original paitent MRI sequences
/examples
   └──/patient1
      ├──patient1_sag t1c+.nii.gz
      ├──patient1_cor t1c+.nii.gz
      ├──patient1_ax t1.nii.gz
      ├──patient1_ax t2.nii.gz
      ├──patient1_ax t2f.nii.gz
      └──patient1_ax t1c+.nii.gz  
```
BrainVLM need 3 parts of input: 1) MRI sequence list (5 MRI sequences); 2) MRI modality information; 3) paitent meta data (Age and gender). 

BrainVLM utilized five core 3D MRI sequences as visual input—T1 (axial or another view), T1c from the same view as T1, T2 (axial or another view), FLAIR (axial or another view), and an additional T1c from a different view than T1.

During inference, BrainVLM will origanized a serise of MRI combination for diagnosis, the final diagnosis is defined by the most frequent prediction in these combinations.

To fulfill the inference process, we need generate a test json same with example_test.json.
For patient1, it have 3 parts: 1) image_list; 2) modality_list; 3) patient metadata.
#### image_list
The image_list stores combinations of MRI sequences for inference. For patient1, the available modalities are axial T1, T2, FLAIR, and T1c+, along with coronal T1c+ and sagittal T1c+. Two test combinations are defined, each containing the file paths of five .nii.gz files.
#### modality_list
The modality_list records the modalities for each combination, with each entry specifying the modality names corresponding to the file paths in the respective image_list combination.

```
/patient1
   └── /image_list
       └── /combination_1
           ├── patient1_ax t1.nii.gz
           ├── patient1_ax t1c+.nii.gz
           ├── patient1_ax t2.nii.gz
           ├── patient1_ax t2f.nii.gz
           └── patient1_cor t1c+.nii.gz
        └──/combination_2
           ├── patient1_ax t1.nii.gz
           ├── patient1_ax t1c+.nii.gz
           ├── patient1_ax t2.nii.gz
           ├── patient1_ax t2f.nii.gz
           └── patient1_sag t1c+.nii.gz
   └──/modality_list
         └── /combination_1
            ├── ax t1
            ├── ax t1c+
            ├── ax t2
            ├── ax t2f
            └── cor t1c+
         └──/combination_2
            ├── ax t1
            ├── ax t1c+
            ├── ax t2
            ├── ax t2f
            └── cor t1c+
   └──patient meta data
         ├── Age
         └── Gender
```
### Evaluation

```
python eval.py --test_json_path --model_ckpt_path
```
For evaluation, replace "test_json_path" with the path to the test JSON file and "model_ckpt_path" with the path to BrainVLM's checkpoint.

For example, you can download the BrainVLM checkpoint to ./ckpts, named checkpoint_1.pth, and then execute the following command:
```
python eval.py --example_test.json --./ckpts/checkpoint_1.pth
```

### References
Our code builds upon MiniGPT-4 and utilizes checkpoints based on LLaMA 3.1 and BiomedCLIP. We would like to thank them.