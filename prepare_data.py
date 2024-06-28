import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import list_image_files_recursive, add_dict_to_argparser


def create_argparser():
    defaults = dict(
        data_dir=".",
        output_dir=".",
        output_name="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
def extract_and_save_images_to_bin(data_dir: str, output_file: str):
    img_files = list_image_files_recursive(data_dir)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240620 # magic
    header[1] = len(img_files) # number of images
    header[2] = 3 # channels
    header[3] = 64 # height
    header[4] = 64 # width

    with open(output_file, "wb") as f:
        f.write(header.tobytes())
        print(f"writing images to {output_file}")
        for img_file in tqdm(img_files):
            pil_img = Image.open(img_file)
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            # preprocess to floats
            arr = arr.astype(np.float32) / 127.5 - 1
            arr = arr.transpose(2, 0, 1)
            f.write(arr.astype("float32").tobytes())

def load_img_from_bin(filename: str):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240620, "Bad magic number"
        n_imgs = header[1]
        C = header[2]
        H = header[3]
        W = header[4]
        imgs = np.frombuffer(f.read(n_imgs * C * H * W * 4), dtype=np.float32)
        imgs = imgs.reshape(n_imgs, C, H, W)

    return imgs

if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    output_name = args.output_name
    if not output_name:
        output_name = os.path.basename(args.data_dir) + ".bin"
        if not output_name:
            raise ValueError("Please provide an output name")
    output_file = os.path.join(args.output_dir, output_name)
    extract_and_save_images_to_bin(args.data_dir, output_file)
    print("Done")