
import os
import subprocess
import sys


if __name__ == '__main__':
    img_path = sys.argv[1]
    # subprocess.Popen(['python3', 'other_models/SRGAN-PyTorch/inference.py', '--inputs_path', f'{img_path}', '--output_path', f'{os.path.join(os.path.dirname(img_path), "3-SRGAN.png")}'])
    subprocess.run(['python3', 'other_models/ESRGAN-PyTorch/inference.py', '--inputs_path', f'{img_path}',
                    '--output_path', f'{os.path.join(os.path.dirname(img_path), "3-ESRGAN.png")}',
                    '--model_weights_path', '/cs/labs/yweiss/ariel1/repos/Generative/GPDM/super_resolution/other_models/ESRGAN_x4-DFO2K-25393df7.pth.tar'])
    subprocess.run(['python3', 'other_models/Real_ESRGAN-PyTorch/inference.py', '--inputs_path', f'{img_path}',
                    '--output_path', f'{os.path.join(os.path.dirname(img_path), "3-RESRGAN.png")}',
                    '--model_weights_path', '/cs/labs/yweiss/ariel1/repos/Generative/GPDM/super_resolution/other_models/RealESRGAN_x4-DFO2K-678bf481.pth.tar'])
    subprocess.run(['python3', 'other_models/pytorch-ZSSR/run_ZSSR_single_input.py', f'{img_path}', f'{os.path.join(os.path.dirname(img_path), "3-ZSSR.png")}'])

    # subprocess.Popen(['./super_resolution/realesrgan-bin/realesrgan-ncnn-vulkan', '-i', f'{img_path}', '-o', f'{os.path.join(os.path.dirname(img_path), "3-Real-ESRGAN.png")}'])
