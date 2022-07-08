# Data preparation guide for Generator_Attribution


# GAN Fingerprints (GANFP) Dataset

1. Download the **[GAN Fingerprints repository](https://github.com/ningyu1991/GANFingerprints)** and associated pretrained models for all four GAN architectures. Use the models named `celeba_align_png_cropped`.
2. Generate 30,000 images for each GAN using the default configurations and seed 0.
3. Store all generated images in the following directories:
   ```
   SNGAN: dataset/ganfp/clean/raw/fake/sngan_celeba
   ProGAN: dataset/ganfp/clean/raw/fake/progan_celeba
   MMDGAN: dataset/ganfp/clean/raw/fake/mmdgan_celeba
   CramerGAN: dataset/ganfp/clean/raw/fake/cramergan_celeba
   ```
4. Download the **[CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** and run `data_crop.py` on the first 30,000 images:
   ```
   python data_crop.py \
      dataset/temp/celeba/images \
      dataset/ganfp/clean/raw/real/real_celeba \
      -r 128 -a 30000 --celeba
   ```
5. Preprocess the raw images into normalized numpy arrays for training/testing.
   - **WARNING: Very space-intensive! Requires approx. 46 GB per complete set.**
   - Use `data_prep.py` to create RGB pixel dataset:
      ```
      python data_prep.py \
         dataset/ganfp/clean/raw/fake \
         dataset/ganfp/clean/raw/real \
         dataset/ganfp/clean/pixel \
         --raw --colour --normalize --seed 0 \
         --train_size 20000 --val_size 4000 \
         --test_size 6000 normal
      ```
   - For frequency domain analysis, first calculate the training set statistics using `data_statistics.py` or use those already provided with this repository:
      ```
      python data_statistics.py \
         dataset/ganfp/clean/raw/fake \
         dataset/ganfp/clean/raw/real \
         dataset/statistics/ganfp/dctnorm \
         --input_size_pos 20000 --input_size_neg 20000 --dctnorm
      ```
      then apply the 2D Discrete Cosine Transform (DCT) using `data_prep.py` with statistics via `--normstats` :
      ```
      python data_prep.py \
         dataset/ganfp/clean/raw/fake \
         dataset/ganfp/clean/raw/real \
         dataset/ganfp/clean/dct \
         --log --normalize --seed 0 \
         --normstats dataset/statistics/ganfp/dctnorm \
         --train_size 20000 --val_size 4000 \
         --test_size 6000 normal
      ```
6. For preparing augmented sets, use `data_augment.py` (replace `multi` with `blur, crop, jpeg,` or `noise` for individual augmentations):
   ```
   python data_augment.py multi \
      dataset/ganfp/clean/raw \
      dataset/ganfp/multi/raw \
      --seed 0
   ```
   - Repeat step (5) for each augmented set once ready.
   - Augmentations are applied to images with 50% probability.
   - `multi` augmentations are applied in the following order:
     1. `blur` (Gaussian blur)
     2. `crop` (Random asymmetric cropping)
     3. `jpeg` (JPEG compression)
     4. `noise` (Additive Gaussian noise)
   - 90-95% of images are augmented at least once via `multi`. 


# FacesHQ+ dataset

1. Download the **[Faces-HQ dataset (19 GB)](https://github.com/cc-hpc-itwm/DeepFakeDetection)** and save it to `dataset/temp` .
2. Download 10,000 unique images using the `utilities/tpdne_extract.py` script **([dependency required](https://github.com/DTrimarchi10/confusion_matrix))**.
   - Images used in official experiments will be released later.
   - Checksum list will be generated to ensure unique images, can be deleted afterwards.
   - `utilities/tpdne_extract.py` usage: 
      ```
      python utilities/tpdne_extract.py 10000 \
         dataset/temp/stylegan2_tpdne
      ```
3. Run `data_crop.py` on every image directory in Faces-HQ to resize all images from 1024x1024 to 256x256:
   ```
   python data_crop.py \
      dataset/temp/celebA-HQ_10K \
      dataset/faceshq+/clean/raw/real/real_celebahq \
      -r 256 -a 10000 --jpeg_output
   ```
4. Repeat (3) until all resized images are saved into `dataset/faceshq+/clean/raw` :
   ```
   dataset/temp/celebA-HQ_10K: 
      --> dataset/faceshq+/clean/raw/real/real_celebahq
   dataset/temp/Flickr-Faces-HQ_10K:
      --> dataset/faceshq+/clean/raw/real/real_ffhq
   dataset/temp/thispersondoesntexists_10K:
      --> dataset/faceshq+/clean/raw/fake/stylegan_tpdne
   dataset/temp/100KFake_10K:
      --> dataset/faceshq+/clean/raw/fake/stylegan_100k
   dataset/temp/stylegan2_tpdne:
      --> dataset/faceshq+/clean/raw/fake/stylegan2_tpdne
   ```
5. (Optional) To create StarGANv2 images as out-of-distribution test data:
   1. **[Download the StarGANv2 CelebA-HQ pretrained generator here.](https://github.com/clovaai/stargan-v2)**
   2. Generate as many collages of style transferred images as you like (at least 2000 images).
   3. Run `utilities/starganv2_segment.py` to extract individual images from the collages: 
      ```
      python utilities/stargan2_segment.py \
         ../stargan-v2/expr/result/round1/reference.jpg \
         ../stargan-v2/expr/result/round2/reference.jpg \
         [filepaths to remaining image collages] \
         --image_size 256 \
         --output_dir dataset/faceshq+/clean/raw/fake_testonly/starganv2_celebahq
      ```
   4. Images used in official experiments will be released later.
6. Preprocess the raw images into normalized numpy arrays for training/testing.
   - **WARNING: Very space-intensive! Requires approx. 64 GB per complete set.**
   - Use `data_prep.py` to create RGB pixel dataset:
      ```
      python data_prep.py \
         dataset/faceshq+/clean/raw/fake \
         dataset/faceshq+/clean/raw/real \
         dataset/faceshq+/clean/pixel \
         --raw --colour --normalize --seed 2021 \
         -t dataset/faceshq+/clean/raw/fake_testonly \
         --train_size 7000 --val_size 1000 \
         --test_size 2000 normal
      ```
   - For frequency domain analysis, first calculate the training set statistics using `data_statistics.py` or use those already provided with this repository:
      ```
      python data_statistics.py \
         dataset/faceshq+/clean/raw/fake \
         dataset/faceshq+/clean/raw/real \
         dataset/statistics/faceshq+/dctnorm \
         --input_size_pos 7000 --input_size_neg 7000 --dctnorm
      ```
      then apply the 2D Discrete Cosine Transform (DCT) using `data_prep.py` with statistics via `--normstats` :
      ```
      python data_prep.py \
         dataset/faceshq+/clean/raw/fake \
         dataset/faceshq+/clean/raw/real \
         dataset/faceshq+/clean/dct \
         --log --normalize --seed 2021 \
         --normstats dataset/statistics/faceshq+/dctnorm \
         -t dataset/faceshq+/clean/raw/fake_testonly \
         --train_size 7000 --val_size 1000 \
         --test_size 2000 normal
      ```
7. For preparing augmented sets, use `data_augment.py` (replace `multi` with `blur, crop, jpeg,` or `noise` for individual augmentations):
   ```
   python data_augment.py multi \
      dataset/faceshq+/clean/raw \
      dataset/faceshq+/multi/raw \
      --seed 2021 --jpeg_output
   ```
   - Repeat step (6) for each augmented set once ready.
   - Augmentations are applied to images with 50% probability.
   - `multi` augmentations are applied in the following order:
     1. `blur` (Gaussian blur)
     2. `crop` (Random asymmetric cropping)
     3. `jpeg` (JPEG compression)
     4. `noise` (Additive Gaussian noise)
   - 90-95% of images are augmented at least once via `multi`. 


# Runtime examples

1. Complete steps (1) to (4) for either dataset.
2. Copy any **sufficiently representative** number of raw images from each source/class and place them accordingly in another directory separate from the rest of the dataset.
   - For example, move images from `dataset/ganfp/clean/raw/fake/progan_celeba` to `dataset/test-ganfp/clean/progan_celeba`.
   - To recreate the official experiments here, use the first 16 images of each source/class where numerical image IDs are arranged in **descending** order.
     - Some of the selected images would fall within the full (actual) test set if the size of the full test set is almost evenly divisible by the full testing batch size.