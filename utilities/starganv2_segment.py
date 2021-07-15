import argparse
import numpy as np

from PIL import Image
from pathlib import Path


"""
Ad-hoc script to extract individual images from StarGANv2 output collages.
"""


def main(args):
    Image.MAX_IMAGE_PIXELS = None
    collage_paths = map(lambda x: Path(x), args.COLLAGES)
    collage_paths = list(collage_paths)
    for collage_path in collage_paths:
        assert collage_path.is_file()
    
    image_id_count = 0
    for collage_path in collage_paths:
        with Image.open(collage_path) as collage:
            num_rows = (collage.size[1] / args.image_size) - 1
            num_cols = (collage.size[0] / args.image_size) - 1
            collage_array = np.asarray(collage)
            
        images = []
        collage_array = collage_array[args.image_size:, args.image_size:, :]
        rows = np.vsplit(collage_array, num_rows)
        for row in rows:
            images.extend(np.hsplit(row, num_cols))
            
        for image in images:
            Image.fromarray(image).save(
                f"{args.output_dir.rstrip('/')}/{image_id_count}.jpg")
            image_id_count += 1
        print(f"Segmentation complete: {str(collage_path)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("COLLAGES", help="StarGANv2 image collages", \
                        type=str, nargs="+")
    parser.add_argument("--output_dir", "-o", help="Image output directory.", \
                        type=str, default="./output")
    parser.add_argument("--image_size", "-s", \
                        help="Pixel resolution per output image (e.g. 1024)", \
                        type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())