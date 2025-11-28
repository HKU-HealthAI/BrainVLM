## A Vision-Language Foundation Model for Comprehensive Brain Tumor Diagnosis from Preoperative Multimodal Data
Yinong Wang †, Jianwen Chen †, Zhou Chen †, Shuwen Kuang †, Haoning Jiang, Yanzhao Shi, Huichun Yuan, Yan-ran (Joyce) Wang, Bing Wang, Lei Wu, Bin Tang, Li Meng, Baihua Luo, Bin Zhou, Wei Ding, Weiming Zhong, Wei Hou, Yuanbing Chen, Zhiping Wan, Wei Wang, Zhenkun Xiao, Wenwu Wan, Allen He, Yuyin Zhou, Longbo Zhang, Feifei Wang, Zhixiong Liu, Michael Iv, Xuan Gong*, Liangqiong Qu*

† These authors contributed equally to this work.

*Corresponding authors.

BrainVLM represents a breakthrough in neuro-oncology, a powerful foundation model designed for in-depth brain tumor analysis. It intelligently processes multi-parametric MRI scans (including T1, T1c, T2, and FLAIR sequences) alongside optional patient metadata to deliver:

(1) **Tumor Diagnosis**: Accurate diagnosis across 12 brain tumor categories according to WHO-CNS5.

(2) **Automated Reporting**: Automatic generation of descriptive radiology reports.

🔥🔥🔥 We are excited to share our evaluation code and pre-trained checkpoints for brain tumor diagnosis and radiology report generation. Explore BrainVLM’s capabilities now!

## Table of Contents
[Environment Installation](#Environment_Installation)

[Prepare model weights](#Prepare_model_weights)

[Prepare test data_files](#Prepare_test_data_files)

[Model Evaluation](#Model_Evaluation)

[Example Output](#Example_Output)

## Environment Installation 
We recommend installing and running this project on Ubuntu 22.04.5.

Clone this repository and navigate to the project folder:
~~~~
git clone https://github.com/HKU-HealthAI/BrainVLM.git
cd BrainVLM
conda env create -f environment.yml
~~~~

## Prepare model weights
BrainVLM comprises three key components: (1) a vision encoder for extracting image features, (2) a MLP projector that integrates the vision encoder with the LLM, and (3) a LLM backbone based on LLaMA 3.1-8B. To utilize BrainVLM, we need to load the vision encoder, LLM backbone, and fine-tuned BrainVLM checkpoint.

| Model                  | Description                          | Download Link                                      | Load Instructions                              |
|------------------------|--------------------------------------|----------------------------------------------------|------------------------------------------------|
| Llama3.1-8B Instruct   | Backbone language model of BrainVLM | [huggingface](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Load in minigpt4/configs/models/minigpt4_vicuna0.yaml: line 18, "llama_model: " |
| BiomedCLIP             | Vision Encoder | [huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | BrainVLM automatically loads this checkpoint, no path needed |
| BrainVLM (Diagnosis and Report) | Checkpoints for report and diagnosis | [google drive](https://drive.google.com/file/d/16yiqIvVVOANpI7OoxBKXvx5NPy0c625n/view?usp=drive_link) | See section [Evaluation](#Evaluation)                                       |
| BrainVLM (Reliable) | Checkpoints for Reliable | [google drive](https://drive.google.com/file/d/1dtSQkVnqE1VUbXXQwfr-vkCLPne5VJX_/view?usp=sharing) | Code is coming soon                                       |

## Prepare test data files:

We provide a sample JSON file for testing: [example_test.json](./example_test.json), which includes data for two patients (patient1 and patient2). BrainVLM supports .nii.gz and .npy file formats as input.

### 1. Test data example:
For [patient1](examples/patient1/), the following 6 MRI sequences are provided:
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
(Note: The directory structure in the example reflects the original patient MRI sequences.)

__Input requirement:__ BrainVLM requires three inputs: 1) a list up to 5 MRI sequences, 2) MRI modality information, and 3) patient metadata (age and gender). It uses five core 3D MRI sequences as visual input: T1, T1c (same view as T1), T2, FLAIR, and an additional T1c (different view). 


__Combination Construction:__ BrainVLM utilized five core 3D MRI sequences as visual input—T1 (axial or another view), T1c from the same view as T1, T2 (axial or another view), FLAIR (axial or another view), and an additional T1c from a different view than T1.


__Inference:__ During inference, BrainVLM organizes MRI combinations based on the construction rule, with the final diagnosis determined by the most frequent prediction. 

To run inference, a test JSON file in the same format as [example_test.json](./example_test.json) is required.

### 2. Origanizing test json file.
You can prepare data folder have same structre with ./examples, then run python [./generate_test_file.py](./generate_test_file.py) to get your test json.

For patient1, it has 3 parts: 1) image_list; 2) modality_list; 3) patient metadata (_optional_).
#### 1) image_list
The image_list defines combinations of MRI sequences for inference. For patient1, available modalities include axial T1, T2, FLAIR, and T1c+, plus coronal and sagittal T1c+. Two combinations are created based on the construction rule, each containing paths to five .nii.gz files.

#### 2) modality_list
The modality_list specifies the modality names corresponding to each file path in the image_list combinations.


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
For custom data testing, you need organize the test combinations same with patient1. 

### 3. Testing for patient with incomplete data.
For a patient with incomplete data, such as [patient2](./examples/patient2/), only the following MRI sequences are available: axial T1, T1c+, and T2, along with coronal T1c+ and sagittal T1c+. The axial FLAIR (T2f) sequence is missing. Additionally, patient2 lacks gender and age information.
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

When the axial FLAIR sequence is unavailable (as with patient2), you can create valid combinations by: (1) Keeping the core sequences intact (axial T1, T1c+, and T2); (2) Substituting alternative views (using coronal and sagittal T1c+).

The resulting json data is shown below:
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

## Model Evaluation

Given the generated test JSON file, the [eval.py](./eval.py) is utilized for diagnosis and report generation, supporting brain tumor classification and radiology report generation through this command:

```
python eval.py --test_json_path <path_to_json> --model_ckpt_path <path_to_checkpoint>
```
Replace <path_to_json> with the path to your test JSON file and <path_to_checkpoint> with the BrainVLM checkpoint path. For example, if the BrainVLM checkpoint is downloaded to ./ckpts/checkpoint_1.pth, run:

```
python eval.py --test_json_path=example_test.json  --model_ckpt_path=./ckpts/checkpoint_1.pth
```

## Example Output
#### For patient 1:
```
Final diagnosis:  
Finding: In Right cerebellum, there is a mass lesion with hypointense in T1, hyperintense in T2, hyperintense in FLAIR.
There is a hypointense T1 signal, hyperintense T2 signal, hyperintense FLAIR signal in Supratentorial white matter. After
contrast administration, there is a marked heterogeneous enhancement. Compression of the fourth ventricle is observed.
No midline structure shift.
Impression: A mass in right cerebellum, considering cranial and paraspinal nerve tumour.
```


## Acknowledgement
Upon acceptance of the paper, we will release all models' weights and relevant source code for pretraining, fine-tuning, and uncertainty qualification for formal testing to facilitate research transparency and community collaboration.

Besides, we will build a online website for efficientlly and automatically brain tumor diagnosis testing.

Our code builds upon MiniGPT-4 and utilizes checkpoints based on LLaMA 3.1 and BiomedCLIP. We would like to thank them.
