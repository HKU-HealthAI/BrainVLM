import json
import os

# test data folder
test_data_folder='./examples' 

import random
def find_combination(path_list):

    combination_list = []
    
    # Group sequences by modality and view
    t1_sequences = []
    t1c_sequences = {} # key: view (ax,cor,sag), value: sequence path
    t2_sequences = []
    t2f_sequences = []
    
    modality_list=[]
    for path in path_list:
        filename = os.path.basename(path).lower()
        # print(filename)
        if 't1c+.nii.gz' in filename or 't1c.nii.gz' in filename:
            if 'ax' in filename:
                t1c_sequences['ax'] = path
            elif 'cor' in filename:
                t1c_sequences['cor'] = path 
            elif 'sag' in filename:
                t1c_sequences['sag'] = path
        elif 't1.nii.gz' in filename or 't1.nii' in filename:
            t1_sequences.append(path)
        elif 't2f.nii.gz' in filename or 't2f.nii' in filename:
            t2f_sequences.append(path)
        elif 't2.nii.gz' in filename or 't2.nii' in filename:
            t2_sequences.append(path)

    for t1_path in t1_sequences:
        t1_view = None
        if 'ax' in t1_path.lower():
            t1_view = 'ax'
        elif 'cor' in t1_path.lower():
            t1_view = 'cor'
        elif 'sag' in t1_path.lower():
            t1_view = 'sag'
            
        # Create a pool of all available sequences for substitution
        all_sequences = []
        all_sequences.extend(t1_sequences)
        all_sequences.extend(t1c_sequences.values())
        all_sequences.extend(t2_sequences) 
        all_sequences.extend(t2f_sequences)
        
        # Keep track of used combinations to avoid duplicates
        used_combinations = set()
        
        # Check if we have all required sequences for this view
        required_sequences = []
        if t1_view:
            required_sequences.append(t1_path)  # T1
            required_sequences.append(t1c_sequences.get(t1_view, None))  # Matching T1C
        
        # For each T2 sequence (or substitute)
        t2_options = t2_sequences if t2_sequences else [None]
        for t2_path in t2_options:
            if t2_path is None:
                # Find substitute excluding already selected sequences
                substitutes = [seq for seq in all_sequences if seq not in required_sequences]
                if substitutes:
                    t2_path = random.choice(substitutes)
                else:
                    continue
                    
            # For each T2F sequence (or substitute)    
            t2f_options = t2f_sequences if t2f_sequences else [None]
            for t2f_path in t2f_options:
                if t2f_path is None:
                    # Find substitute excluding already selected sequences
                    substitutes = [seq for seq in all_sequences if seq not in required_sequences + [t2_path]]
                    if substitutes:
                        t2f_path = random.choice(substitutes)
                    else:
                        continue
                        
                # For remaining T1C views
                other_t1c_options = [v for v in t1c_sequences.values() if v != t1c_sequences.get(t1_view)]
                if not other_t1c_options:
                    # Find substitute excluding already selected sequences
                    substitutes = [seq for seq in all_sequences if seq not in required_sequences + [t2_path, t2f_path]]
                    if substitutes:
                        other_t1c_options = [random.choice(substitutes)]
                    else:
                        continue
                        
                for other_t1c in other_t1c_options:
                    combination = [t1_path, t1c_sequences.get(t1_view), t2_path, t2f_path]

                    if other_t1c in combination:
                        continue
                    combination.append(other_t1c)
                    # Convert to frozenset to check for duplicates regardless of order
                    combination_set = frozenset(combination)
                    if combination_set not in used_combinations:
                        used_combinations.add(combination_set)
                        combination_list.append(combination)
                        modality=[c.split('_')[1].split('.nii.gz')[0] for c in combination]
                        modality_list.append(modality)
        
        if t1_view and t1_view in t1c_sequences:
            matching_t1c = t1c_sequences[t1_view]
            
            # Get remaining T1C views
            other_t1c_views = [v for v in t1c_sequences.values() if v != matching_t1c]
            
            # For each T2
            for t2_path in t2_sequences:
                # For each T2F
                for t2f_path in t2f_sequences:
                    # For each remaining T1C view
                    for other_t1c in other_t1c_views:
                        combination = [
                            t1_path,
                            matching_t1c,
                            t2_path,
                            t2f_path,
                            other_t1c
                        ]
                        combination_set = frozenset(combination)
                        if combination_set not in used_combinations:
                            used_combinations.add(combination_set)
                            combination_list.append(combination)
                            modality=[c.split('_')[1].split('.nii.gz')[0] for c in combination]
                            modality_list.append(modality)
    
    return combination_list,modality_list

patient_list=[]
patient_combination_list=[]
patient_modality_list=[]
for patient_folder in os.listdir(test_data_folder):
    patient_folder_path=os.path.join(test_data_folder,patient_folder)
    if os.path.isdir(patient_folder_path):
        patient_list.append(patient_folder)
        patient_list.append(patient_folder_path)
        path_list=[]
        for file in os.listdir(patient_folder_path):
            path_list.append(os.path.join(patient_folder_path,file))
        combination_list,modality_list=find_combination(path_list)
        patient_combination_list.append(combination_list)
        patient_modality_list.append(modality_list)


result={k:{'image_list':combination,'modality':modality} for k,combination,modality in zip(patient_list,patient_combination_list,patient_modality_list)}
with open('./examples/example_test.json','w') as f:
    json.dump(result,f,ensure_ascii=False,indent=4)
