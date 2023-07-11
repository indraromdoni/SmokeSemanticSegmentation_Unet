from roboflow import Roboflow
rf = Roboflow(api_key="OdWJd7CgMLShi3sZ0oYW")
project = rf.workspace("indra-romdoni-sfxza").project("smoke-detector-doxwl")
dataset = project.version(4).download("coco-segmentation")