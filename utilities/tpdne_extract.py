"""
Script to download synthetic images from thispersondoesnotexist.com
API available: https://github.com/David-Lor/ThisPersonDoesNotExistAPI
"""

from thispersondoesnotexist import \
    get_online_person, get_checksum_from_picture, save_picture
from pathlib import Path

import argparse
import os


def main(args):
    checksum_list = Path(args.checksum_list)
    image_id_count = 0
    
    if checksum_list.exists():
        assert checksum_list.is_file()
        checksum_file = open(checksum_list, "r")
        checksums = set(checksum_file.read.splitlines())
        image_id_count += len(checksums)
        checksum_file.close()
    else:
        checksums = set()
        
    def retrieve_unique_image():
        # recursive function
        img = get_online_person()
        img_checksum = get_checksum_from_picture(img)
        if img_checksum in checksums:
            return retrieve_unique_image()
        else:
            return img
        
    for image_count in range(args.NUM_IMAGES):
        image = retrieve_unique_image()
        checksums.add(get_checksum_from_picture(image))
        save_picture(image, f"{args.OUTPUT_DIR.rstrip('/')}/{image_id_count}.jpg")
        image_id_count += 1
        
    if checksum_list.exists():
        os.remove(checksum_list)
    checksum_file = open(checksum_list, "w")
    for csum in checksums:
        checksum_file.write(f"{csum}\n")
    checksum_file.close()
    

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "NUM_IMAGES", help="Number of images to retrieve", type=int)
    parser.add_argument(
        "OUTPUT_DIR", help="Directory of output images", type=str)
    parser.add_argument(
        "--checksum_list", "-c", default="checksums.txt",
        help="Text file containing used checksums", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())