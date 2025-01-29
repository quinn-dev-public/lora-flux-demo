<iframe width="560" height="315" src="https://brandbot.studio/public/.tmp/2389jsd.mp4" frameborder="0" allow="autoplay"></iframe>

---

Trains a LoRA model on images of a person, which you can use with Flux to generate images. Then aligns faces and combines into a video. 
1. Train a model on a person.  
2. Generate images of them aging over time.  
3. Align the faces.  
4. Combine images into a video.  

---

```bash
pip install requirements.txt
```
---

## Step 1: Train the Model  
```bash
python 01-train.py
```
You'll need ~20 photos of a person. Crop out any other people. The web ui will give you other caveats about hair / age. See the example input photos I used (input/), generally photos from a 10-year time period, i wasn't too careful with this.
- Runs on an **H100 GPU** ($0.001525/sec).  
- Training on the **20 images** in this repo with the default parameters took **11m 22s ($1.04)**.  
- **i wouldn't recommend using `01-train.py`** for this, just go to the web UI instead:  
  **[https://replicate.com/ostris/flux-dev-lora-trainer/train](https://replicate.com/ostris/flux-dev-lora-trainer/train)**  
- After it runs you can download your model and it'll also display a 'Use this model'. From there all I changed was setting a 16:9 aspect ratio and jpg format in generating images for step 2. Step 2 just iterates on that with a prompt to generate images over time to age the person.

---

## Step 2: Generate Images  
```bash
python 02-generate-images.py 
```
Set to generate six images vs. the ~500 I generated in the example video
- Running the trained model on an H100 took on average 7 seconds ($0.0108 per image).

---

## Step 3) Align the faces of all images
sorry i just have this in 03-align.sh for now, which runs
```bash
python face-align/align.py -images output-v1 -target output-v1/2022_july.jpg -overlay -border 0 -outdir aligned_output-v1
```

face-align is from https://github.com/andrewdcampbell/face-movie, I only changed it to not fail if a face isn't detected, and save progress of annotations (I was using this on non-AI generated images before - for this use-case it'll very rarely prompt you to even confirm the right subject)

---

## Step 4) python face-align/main.py -morph -images aligned_output-v1 -td 0.4 -pd 0.2 -fps 20 -out aligned_output-v1.mp4
