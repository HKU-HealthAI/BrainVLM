import numpy as np

import json
import os
import nibabel as nib
def load_test_data():
    """
    Load and parse the example_test.json file
    
    Returns:
        dict: Loaded JSON data
    """
    with open('example_test.json', 'r') as f:
        data = json.load(f)
    return data

data=load_test_data()


for k,v in data.items():

    image_list=v['image_list']
    for image_combination in image_list:
        for image_path in image_combination:
            image=np.load(image_path)
            print(image.shape)
            import nibabel as nib
            # Create a NIfTI image from the numpy array
            # Assuming image is a 3D numpy array
            nifti_img = nib.Nifti1Image(image, affine=np.eye(4))
            
            # Generate output filename by replacing .npy extension with .nii.gz
            output_path = image_path.replace('.npy', '.nii.gz')
            output_path=os.path.join('./examples',os.path.basename(output_path))
            # breakpoint()
            # Save the NIfTI image
            nib.save(nifti_img, output_path)


