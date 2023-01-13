from sklearn.ensemble import IsolationForest
from numpy import random, where
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from skimage import io
from PIL import Image
import submodlib
import os
# Inspired from https://github.com/JohannesBuchner/imagehash repository
from PIL import Image,ImageStat
import hashlib
import os
import diversipy


def hash_image(image_path):
    img = Image.open(image_path).resize((8,8), Image.LANCZOS).convert(mode="L")
    mean = ImageStat.Stat(img).mean[0]
    return sum((1 if p > mean else 0) << i for i, p in enumerate(img.getdata()))


def main(d_path, out_path):


    directories = [d for d in os.listdir(d_path)
                   if os.path.isdir(os.path.join(d_path, d))]


    for d in directories:


        hashes = set()

        label_directory = os.path.join(d_path, d+'/00000/')
        for filename in sorted(os.listdir(label_directory)):
            if filename.endswith(".jpg"):
              path = os.path.join(label_directory, filename)
              digest=hash_image(path)
            #digest = hashlib.sha1(open(path,'rb').read()).digest()
              if digest not in hashes:
                 hashes.add(digest)
              else:
                 os.remove(path)

        images=[]


        file_names = [os.path.join(label_directory, f)
                      for f in sorted(os.listdir(label_directory))
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(io.imread(f))

        print(np.array(images).shape)
        imgs=[Image.fromarray(i) for i in images]



        model = SentenceTransformer('clip-ViT-L-14')

        embs=model.encode(imgs)

        sub=diversipy.subset.select_greedy_energy(embs,30, exponent=30)

        indices= np.array([np.where(embs==i)i[0][0] for i in sub])

        image_subset=[np.array(images)[i] for i in indices]

        os.mkdir(out_path+'/'+str(d)+'/')

   #Write  images to directory

        for i in np.arange(0,len(image_subset)):

             io.imsave(os.path.join(out_path+'/'+str(d)+'/' , str(i)+'.jpg'), image_subset[i])

if __name__== "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Enter required inputs')

    parser.add_argument('--d_path', metavar='path', required=True, help='the path to input diectory')

    parser.add_argument('--out_path', metavar='out_p', required=True, help='Output directory path')

    args = parser.parse_args()

    main(d_path=args.d_path,out_path=args.out_path)

