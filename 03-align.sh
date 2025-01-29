python face-align/align.py -images output-v1 -target output-v1/2022_july.jpg -overlay -border 0 -outdir aligned_output-v1

python face-align/main.py -morph -images aligned_output-v1 -td 0.4 -pd 0.2 -fps 20 -out aligned_output-v1.mp4
