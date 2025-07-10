## A Foundation Model for Presurgical Brain Tumor Diagnosis and MRI Interpretation
BrainVLM is a foundation model designed for comprehensive brain tumor analysis. It leverages multi-parametric MRI scans (T1, T1c, T2, and FLAIR sequences) and _optional_ patient metadata as input, and provides diagnosis (among 12 brain tumor categories of WHO-CNS5) and radiology report for patient.

🔥🔥🔥 We have released our evaluation code and checkpoints for brain tumor diagnosis and radiology report generation. Welcome everyone to try it now!

## Table of Contents
[Installation](#Installation)

[Prepare model weights](#Prepare_model_weights)

[Prepare test data](#Prepare_test_data)

[Evaluation](#Evaluation)

[Example Output](#Example_Output)

## Installation (Linux)
1. Clone this repository and navigate to the brainvlm folder

~~~~
git clone https://github.com/HKU-HealthAI/BrainVLM.git
cd BrainVLM
conda env create -f brainvlm_foundation.yml
~~~~

## Prepare model weights

# Model Downloads

| Model                  | Download Link                                      | Load Instructions                              |
|------------------------|----------------------------------------------------|------------------------------------------------|
| Llama3.1-8B Instruct   | [huggingface](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Load in minigpt4/configs/models/minigpt4_vicuna0.yaml: line 18, "llama_model: " |
| BiomedCLIP             | [huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | BrainVLM automatically loads this checkpoint, no path needed |
| BrainVLM (Diagnosis and Report) | [google drive](https://drive.google.com/file/d/16yiqIvVVOANpI7OoxBKXvx5NPy0c625n/view?usp=drive_link) |  See section [Evaluation](#Evaluation)                                       |



## Prepare test data files:

We provide a json file for test: example_test.json (includes 2 patients, patient1 and patient2). BrainVLM supports nii.gz and npy files as input.
### 1. Patient Example in example_test.json:
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

During inference, BrainVLM will origanize a serise of MRI combination for diagnosis according to the combination construction rule, the final diagnosis is defined by the most frequent prediction in these combinations.

To perform the inference process, we need to generate a test JSON file with the same format as example_test.json.

### 2. Origanizing test json file for patient 1.
For patient1, it has 3 parts: 1) image_list; 2) modality_list; 3) patient metadata (_optional_).
#### 1) image_list
The image_list stores combinations of MRI sequences for inference. For patient1, the available modalities are axial T1, T2, FLAIR, and T1c+, along with coronal T1c+ and sagittal T1c+. Two test combinations have been established according to the combination construction rule, each including the file paths of five .nii.gz files. 
#### 2) modality_list
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

### 3. Testing for patient with incomplete data.
For custom data testing, you need organize the test combinations same with patient1. 

For a patient with incomplete data, such as patient2, only the following MRI sequences are available: axial T1, T1c+, and T2, along with coronal T1c+ and sagittal T1c+. The axial FLAIR (T2f) sequence is missing. Additionally, patient2 lacks gender and age information.
```
This is original paitent MRI sequences
/examples
   └──/patient2
      ├──patient2_sag t1c+.nii.gz
      ├──patient2_cor t1c+.nii.gz
      ├──patient2_ax t1.nii.gz
      ├──patient2_ax t2.nii.gz
      └──patient2_ax t1c+.nii.gz  
```


For testing patient2, we can create a combination by keeping the other modalities fixed and substituting coronal T1c+ for the missing axial FLAIR. The resulting json data is shown below:
```
/patient2
   └── /image_list
       └── /combination_1
           ├── patient2_ax t1.nii.gz
           ├── patient2_ax t1c+.nii.gz
           ├── patient2_ax t2.nii.gz
           ├── patient2_cor t1c+.nii.gz
           └── patient2_sag t1c+.nii.gz
   └──/modality_list
         └── /combination_1
            ├── ax t1
            ├── ax t1c+
            ├── ax t2
            ├── cor t1c+
            └── sag t1c+

```

## Evaluation

```
python eval.py --test_json_path --model_ckpt_path
```
For evaluation, replace "test_json_path" with the path to the test JSON file and "model_ckpt_path" with the path to BrainVLM's checkpoint.

For example, you can download the BrainVLM checkpoint([https://drive.google.com/drive/home?dmr=1&ec=wgc-drive-hero-goto](https://drive.google.com/file/d/16yiqIvVVOANpI7OoxBKXvx5NPy0c625n/view?usp=drive_link)
) to ./ckpts, named checkpoint_1.pth, and then execute the following command:
```
python eval.py example_test.json ./ckpts/checkpoint_1.pth
```
## Example Output
#### For patient 1:
```
combination modality: ['ax t1', 'ax t1c+', 'ax t2', 'ax t2f', 'cor t1c+']

combination_0: In Right cerebellum, there is a mass lesion with hypointense in T1, hyperintense in T2. After contrast administration, there is a heterogeneous enhancement. Compression of the fourth ventricle is observed. No midline structure shift. This patient was diagnosed with cranial and paraspinal nerve tumour. 
```

```
combination modality: ['ax t1', 'ax t1c+', 'ax t2', 'ax t2f', 'sag t1c+']

combination_1: In Right cerebellum, there is a mass lesion with hypointense in T1, hyperintense in T2, hyperintense in FLAIR. There is a hypointense T1 signal, hyperintense T2 signal, hyperintense FLAIR signal in Supratentorial white matter. After contrast administration, there is a marked heterogeneous enhancement. Compression of the fourth ventricle is observed. No midline structure shift. This patient was diagnosed with cranial and paraspinal nerve tumour. 
```

```
Final diagnosis:  This patient was diagnosed with cranial and paraspinal nerve tumour
```


## Acknowledgement
Upon acceptance of the paper, we will release all models' weights and relevant source code for pretraining, fine-tuning, and uncertainty qualification for formal testing to facilitate research transparency and community collaboration.

Our code builds upon MiniGPT-4 and utilizes checkpoints based on LLaMA 3.1 and BiomedCLIP. We would like to thank them.
