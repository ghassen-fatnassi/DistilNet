from Unet import studentUnet,teacherUnet,Unet
from myTransformers import teacherSegformer
import torch
import safetensors.torch
from torch.profiler import profile, record_function, ProfilerActivity

student=studentUnet(num_classes=19,
                    in_channels=3,
                    start_filts=8,
                    depth=4)

studentPath='Student_D=4_Filts=16_MobileNetv2_NoDropout_EP19.safetensors'

student.load_state_dict(safetensors.torch.load_file(studentPath))

teacher=teacherUnet(num_classes=19,
                    in_channels=3,
                    start_filts=32,
                    depth=5)

inputs = torch.randn(5, 3, 224, 224)


