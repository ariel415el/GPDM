
import os
import subprocess


if __name__ == '__main__':
    models_path = f'{os.path.dirname(__file__)}/other_models'

    dirpath = "outputs"
    for dirname in os.listdir("outputs"):
        img_path = os.path.join(dirpath, dirname, '2-corrupt_image.png')
        print(f"### Working ing {img_path}")
        # subprocess.Popen(['python3', 'other_models/SRGAN-PyTorch/inference.py', '--inputs_path', f'{img_path}', '--output_path', f'{os.path.join(os.path.dirname(img_path), "3-SRGAN.png")}'])

        subprocess.run(['python3', f'{models_path}/ESRGAN-PyTorch/inference.py', '--inputs_path', f'{img_path}',
                        '--output_path', f'{os.path.join(os.path.dirname(img_path), "3-ESRGAN.png")}',
                        '--model_weights_path', f'{models_path}/ESRGAN_x4-DFO2K-25393df7.pth.tar'])

        # subprocess.run(['python3', f'{models_path}/Real_ESRGAN-PyTorch/inference.py', '--inputs_path', f'{img_path}',
        #                 '--output_path', f'{os.path.join(os.path.dirname(img_path), "3-RESRGAN.png")}',
        #                 '--model_weights_path', f'{models_path}/RealESRGAN_x4-DFO2K-678bf481.pth.tar'])

        subprocess.run(['python3', f'{models_path}/pytorch-ZSSR/run_ZSSR_single_input.py', f'{img_path}', f'{os.path.join(os.path.dirname(img_path), "3-ZSSR.png")}'])

        # subprocess.Popen(['./super_resolution/realesrgan-bin/realesrgan-ncnn-vulkan', '-i', f'{img_path}', '-o', f'{os.path.join(os.path.dirname(img_path), "3-Real-ESRGAN.png")}'])
