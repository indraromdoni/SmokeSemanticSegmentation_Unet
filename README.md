# SmokeSemanticSegmentation_Unet
Semantic segmentation for smoke especially furnace smoke detection

- Import dataset that created in roboflow by running import_roboflow_dataset.py.
- make new folder called "masks" in downloaded database
- Run read_json_coco.py for making mask image from annotated image.
- Run unet_smoke_trainer.py to train the smoke dataset
- The file "smoke_detection.py" for detecting image
- The file "rt_smoke_detection.py" for real time detection
