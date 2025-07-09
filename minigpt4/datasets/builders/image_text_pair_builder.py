import os
import logging
import warnings

from minigpt4.common.registry import registry

from minigpt4.datasets.datasets.caption_datasets import xiangya_training



    
    

@registry.register_builder("xiangya_training")
class xiangyatraining(BaseDatasetBuilder):
    train_dataset_cls = xiangya_training


    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/xiangya_training.yaml",
    }
    
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info

        # print(build_info)
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=self.anns_path,
            vis_root=storage_path,
            image_encoder=self.image_encoder
        )

        return datasets
    
# our_training
