# CONVERT PTH TO ONNX

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as vstrans
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name
from datetime import datetime

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}

class AntiSpoofPredict():
    def __init__(self):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "/mnt/ssd/ekyc_myanmar/project_fake_profile/Silent-Face-Anti-Spoofing/saved_logs/snapshot/Anti_Spoofing_2_80x80_add_fake_prof_august_martin/2024-10-07-14-55_Anti_Spoofing_2_80x80_add_fake_prof_august_model_iter-4710.pth"
        self.h_input = 80
        self.w_input = 80
        self.model_type = 'MiniFASNetV2SE'
        self.zoom_ratio = 2


    def _load_model(self):
        # define model
        # h_input = , w_input, model_type, _ = parse_model_name(model_name)
        timestamp = (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
        self.onnx_output_path = f"ProfileAntiSpoof__{timestamp}_{self.zoom_ratio}_{self.h_input}x{self.w_input}_{self.model_type}.onnx"

        self.kernel_size = get_kernel(self.h_input, self.w_input,)
        self.model = MODEL_MAPPING[self.model_type](conv6_kernel=self.kernel_size, num_classes=2).to(self.device)

        # load model weight
        state_dict = torch.load(self.model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.model.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'rator' not in key:
                    name_key = key[13:]
                    new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        return None


    def convert(self):
        self._load_model()
        dummy_input = torch.randn(1, 3, 80, 80, device=self.device)
        torch.onnx.export(self.model,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            self.onnx_output_path,       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=10,      # ONNX version to export the model to
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names=['input'],     # The model's input names
            output_names=['output'],   # The model's output names
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Enable dynamic batching
        ) 
        print('Model has been converted to ONNX')   


model_test = AntiSpoofPredict()
prediction = model_test.convert()