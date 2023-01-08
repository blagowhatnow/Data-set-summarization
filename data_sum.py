from sklearn.ensemble import IsolationForest
from numpy import random, where
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from skimage import io
from PIL import Image
import submodlib
from submodlib import FacilityLocationFunction
import os

def main(d_path, rep_percent, samp_percent,out_path):

    images=[]

    for filename in sorted(os.listdir(d_path)):
      if filename.endswith(".jpg"):
        img = io.imread(os.path.join(d_path,filename))
        images.append(img)

   
    imgs=[Image.fromarray(i) for i in images]

    model = SentenceTransformer('clip-ViT-L-14')

    embs=model.encode(imgs) 

    IF=IsolationForest(max_samples='auto', random_state=42, contamination='auto')

    pred = IF.fit_predict(embs)

    sklearn_score_anomalies = IF.decision_function(embs)

    sorted_inds=sklearn_score_anomalies.argsort()[::-1]
    
    samp=embs[sorted_inds][0:int(np.round_(len(embs)*(float(samp_percent)/100)))]
   
    objFL = FacilityLocationFunction(n=len(embs), data=np.array(embs), separate_rep=True, n_rep=samp.shape[0], data_rep=np.array(samp), mode="dense", metric="cosine")


    subset_indices = objFL.maximize(budget=round((embs.shape[0])*(int(samp_percent)/100)),optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    
    np.array(subset_indices).shape
    
    image_subset=[images[i[0]] for i in subset_indices]

   #Write  images to directory

    for i in np.arange(0,len(image_subset)):

        io.imsave(os.path.join(out_path , str(i)+'.jpg'), image_subset[i])


if __name__== "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Enter required inputs')

    parser.add_argument('--d_path', metavar='path', required=True, help='the path to input diectory')

    parser.add_argument('--rep_percent', metavar='rep_p', required=True, help='Isolation forest representative subset percent')

    parser.add_argument('--samp_percent', metavar='samp_p', required=True, help='Final collection percentage of the original dataset')

    parser.add_argument('--out_path', metavar='out_p', required=True, help='Output directory path')

    args = parser.parse_args()

    main(d_path=args.d_path,rep_percent=args.rep_percent,samp_percent=args.samp_percent,out_path=args.out_path)
