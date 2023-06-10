import matplotlib.pyplot as plt
import numpy as np
import rosbag, os
from tqdm import tqdm
import cv2
import argparse

parser = argparse.ArgumentParser("The argparse for transformation from Event Streams to Images")
parser.add_argument('--path', type=str, default="~/Desktop/phd/proj1/ESIM/cheetah_example/out.bag", help="The path of the event stream (.bag)")
parser.add_argument("--out_path", type=str, default="./out", help="The output path of the generated images")
parser.add_argument("--width", type=int, default=640, help="The width of frame size")
parser.add_argument("--height", type=int, default=480, help="The height of frame size")



def event2img(args):
    bag = rosbag.Bag(args.path)
    for id_f, (topic,msgs,t) in enumerate(bag.read_messages(topics=['/cam0/events'])):
        frame = np.zeros([args.height, args.width, 3])
        for single_event in tqdm(msgs.events):  
            # f.write(str(single_event.ts.secs+single_event.ts.nsecs*1.0/1000000000)+' ')        
            if single_event.polarity:
                frame[single_event.y, single_event.x, 0] += 1
            else:
                frame[single_event.y, single_event.x, 2] += 1
        
        # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # plt.imshow(frame)
        if (frame[:,:,1]!=0).any():
            print("?")
        cv2.imwrite(os.path.join(args.out_path, ('%05d' %(id_f+1))+ ".jpg" ), frame)
        np.save(os.path.join(args.out_path, ('%05d' %(id_f+1))+ ".npy" ), frame)
        

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    event2img(args)