# USAGE: python face-movie/main.py (-morph | -average) -images IMAGES [-td TD] [-pd PD] [-fps FPS] -out OUT

from scipy.spatial import Delaunay
from PIL import Image
from face_morph import morph_seq, warp_im
from subprocess import Popen, PIPE
import argparse
import numpy as np
import dlib
import os
import cv2
import time
import json
import os.path
from multiprocessing import Pool, cpu_count
from functools import partial

########################################
# FACIAL LANDMARK DETECTION CODE
########################################

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

def load_face_choices():
    if os.path.exists('face-movie/dir3.json'):
        with open('face-movie/dir3.json', 'r') as f:
            return json.load(f)
    return {}

FACE_CHOICES = load_face_choices()

def get_boundary_points_bak(shape):
    h, w = shape[:2]
    boundary_pts = [
        (1,1), (w-1,1), (1, h-1), (w-1,h-1), 
        ((w-1)//2,1), (1,(h-1)//2), ((w-1)//2,h-1), ((w-1)//2,(h-1)//2)
    ]
    return np.array(boundary_pts)

def get_boundary_points(shape):
    return np.array([])  # No boundary points


def get_landmarks(im, filename=None):
    # If we're working with already aligned images, just return a grid of points
    h, w = im.shape[:2]
    x = np.linspace(0, w-1, 8)
    y = np.linspace(0, h-1, 8)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.column_stack((xv.flatten(), yv.flatten()))
    return np.array(grid_points)

    # Original face detection code (now skipped for aligned images)
    # rects = DETECTOR(im, 1)
    # if len(rects) == 0:
    #     rects = DETECTOR(im, 0)
    # if len(rects) == 0:
    #     print("No faces found!")
    #     return None
    
    # if len(rects) > 1:
    #     if filename and filename in FACE_CHOICES:
    #         return np.matrix([[p.x, p.y] for p in PREDICTOR(im, rects[FACE_CHOICES[filename]]).parts()])
    #     return prompt_user_to_choose_face(im, rects)
    # return np.matrix([[p.x, p.y] for p in PREDICTOR(im, rects[0]).parts()])

def prompt_user_to_choose_face(im, rects):
    im = im.copy()
    h, w = im.shape[:2]
    for i in range(len(rects)):
        d = rects[i]
        x1, y1, x2, y2 = d.left(), d.top(), d.right()+1, d.bottom()+1
        cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)
        cv2.putText(im, str(i), (d.center().x, d.center().y),
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1.5,
                    color=(255, 255, 255),
                    thickness=5)

    DISPLAY_HEIGHT = 650
    resized = cv2.resize(im, (int(w * DISPLAY_HEIGHT / float(h)), DISPLAY_HEIGHT))
    cv2.imshow("Multiple faces", resized); cv2.waitKey(1)
    target_index = int(input("Please choose the index of the target face: "))
    cv2.destroyAllWindows(); cv2.waitKey(1)
    return rects[target_index] 

########################################
# VISUALIZATION CODE FOR DEBUGGING
########################################    

def draw_triangulation(im, landmarks, triangulation):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.triplot(landmarks[:,0], landmarks[:,1], triangulation, color='blue', linewidth=1)
    plt.axis('off')
    plt.show()

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.putText(im, str(idx+1), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255))
        cv2.circle(im, pos, 3, color=(255, 0, 0))
    cv2.imwrite("landmarks.jpg", im)


########################################
# MAIN DRIVER FUNCTIONS
########################################

def average_images(out_name):
    avg_landmarks = sum(LANDMARK_LIST) / len(LANDMARK_LIST)
    triangulation = Delaunay(avg_landmarks).simplices

    warped_ims = [
        warp_im(np.float32(IM_LIST[i]), LANDMARK_LIST[i], avg_landmarks, triangulation) 
        for i in range(len(LANDMARK_LIST))
    ]

    average = (1.0 / len(LANDMARK_LIST)) * sum(warped_ims)
    average = np.uint8(average)

    cv2.imwrite(out_name, average)

def morph_images(duration, fps, pause_duration, out_name):
    first_im = cv2.cvtColor(IM_LIST[0], cv2.COLOR_BGR2RGB)
    h = max(first_im.shape[:2])
    w = min(first_im.shape[:2])    

    # Setup ffmpeg process
    command = ['ffmpeg', 
        '-y', 
        '-f', 'image2pipe', 
        '-r', str(fps), 
        '-s', str(h) + 'x' + str(w), 
        '-i', '-', 
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'film',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', 
        '-pix_fmt', 'yuv420p',
        '-threads', str(cpu_count()),
        out_name,
    ]         

    p = Popen(command, stdin=PIPE)

    # Write pause frames at start
    pause_frames = int(fps * pause_duration)
    first_frame = Image.fromarray(first_im)
    for _ in range(pause_frames):
        first_frame.save(p.stdin, 'JPEG', quality=95)

    # Process in batches of 5 pairs
    BATCH_SIZE = 5
    pairs = [(i, i+1) for i in range(len(IM_LIST) - 1)]
    
    for i in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[i:i+BATCH_SIZE]
        
        # Process batch in parallel
        with Pool(processes=min(cpu_count(), len(batch_pairs))) as pool:
            morph_pair_partial = partial(
                morph_pair_wrapper,
                duration=duration,
                fps=fps,
                im_list=IM_LIST,
                landmark_list=LANDMARK_LIST
            )
            batch_frames = pool.map(morph_pair_partial, batch_pairs)
        
        # Write frames from this batch
        for frame_seq in batch_frames:
            for frame in frame_seq:
                frame_img = Image.fromarray(cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB))
                frame_img.save(p.stdin, 'JPEG', quality=95)
        
        print(f"Processed {i+len(batch_pairs)}/{len(pairs)} transitions")

    p.stdin.close()
    p.wait()

def morph_pair_wrapper(pair, duration, fps, im_list, landmark_list):
    """Generate frames for a pair of images without writing to stream"""
    idx1, idx2 = pair
    im1 = im_list[idx1]
    im2 = im_list[idx2]

    total_frames = int(duration * fps)
    frames = []

    # Enhanced fade transition
    for j in range(total_frames):
        t = j / (total_frames - 1)
        # Smoother sigmoid-like curve for more overlap in the middle
        alpha = 3 * t * t - 2 * t * t * t  # Enhanced fade curve
        
        # Adjust weights to increase fade overlap
        w1 = max(0, min(1, 2 * (1-alpha)))  # Extend first image's presence
        w2 = max(0, min(1, 2 * alpha))      # Extend second image's presence
        
        # Enhanced cross-fade blend
        blended = cv2.addWeighted(im1, w1, im2, w2, 0)
        frames.append(blended)

    return frames

def smooth_step(x):
    # Smoothstep function for better easing
    return x * x * (3 - 2 * x)

if __name__ == "__main__":
    start_time = time.time()
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-morph", help="Create morph sequence", action='store_true')
    group.add_argument("-average", help="Create average face", action='store_true')
    ap.add_argument("-images", help="Directory of input images", required=True)
    ap.add_argument("-td", type=float, help="Transition duration (in seconds)", default=3.0)
    ap.add_argument("-pd", type=float, help="Pause duration (in seconds)", default=0.0)
    ap.add_argument("-fps", type=int, help="Frames per second", default=25)
    ap.add_argument("-out", help="Output file name", required=True)
    args = vars(ap.parse_args())

    MORPH = args["morph"]
    IM_DIR = args["images"]
    FRAME_RATE = args["fps"]
    DURATION = args["td"]
    PAUSE_DURATION = args["pd"]
    OUTPUT_NAME = args["out"]

    valid_formats = [".jpg", ".jpeg", ".png"]
    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Constraints on input images (for morphing):
    # - Must all have same dimension
    # - Must have clear frontal view of a face (there may be multiple)
    # - Filenames must be in lexicographic order of the order in which they are to appear

    IM_FILES = [f for f in os.listdir(IM_DIR) if get_ext(f) in valid_formats]
    IM_FILES = sorted(IM_FILES, key=lambda x: x.split('/'))
    assert len(IM_FILES) > 0, "No valid images found in {}".format(IM_DIR)

    IM_LIST = [cv2.imread(IM_DIR + '/' + f, cv2.IMREAD_COLOR) for f in IM_FILES]
    print("Detecting landmarks...")
    LANDMARK_LIST = [get_landmarks(im) for im in IM_LIST]
    print("Starting...")

    if MORPH:
        morph_images(DURATION, FRAME_RATE, PAUSE_DURATION, OUTPUT_NAME)
    else:
        average_images(OUTPUT_NAME)

    elapsed_time = time.time() - start_time
    print("Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
