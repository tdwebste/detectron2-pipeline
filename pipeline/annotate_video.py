import cv2
import torch
import numpy as np
import sys

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer

from pipeline.pipeline import Pipeline
from pipeline.utils.colors import colors
from pipeline.utils.text import put_text

import math
import pprint
pp = pprint.PrettyPrinter(indent=4)
outfile = sys.stdout



class AnnotateVideo(Pipeline):
    """Pipeline task for video annotation."""

    def __init__(self, dst, metadata_name, instance_mode=ColorMode.IMAGE,
                 frame_num=True, predictions=True, pose_flows=True):
        self.dst = dst
        self.metadata_name = metadata_name
        self.metadata = MetadataCatalog.get(self.metadata_name)
        self.instance_mode = instance_mode
        self.frame_num = frame_num
        self.predictions = predictions
        self.pose_flows = pose_flows

        self.cpu_device = torch.device("cpu")
        self.video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        self.size ={}
        self.size['image'] = [0,0]
        self.size['room'] = [14, 16] #size in feet
        self.size['room'][0] = self.size['room'][0]/3.28084
        self.size['room'][1] = self.size['room'][1]/3.28084
        self.size['image'][0] = 1920 #image_width in pixes
        self.size['image'][1] = 1080 #image_hieght in pixes
        self.size['scale'] = [1,1]
        self.size['scale'][0] = self.size['room'][0]/ self.size['image'][0]
        self.size['scale'][1] = self.size['room'][1]/ self.size['image'][1]

        self.framedata = {}
        self.accu = {}
        self.accu['person'] = {}
        self.items = {
            'counter': [( 1100,500), (820,340)],
            'doors': [( 1150,550), (1090,400)],
            'dish': [( 950,330), (850,270)],
            'gloves': [( 950,380), (850,350)],
            'avocado': [( 920,470), (810,380)],
        }

        super().__init__()

    def calc_movement(self):
        for frame_pidx in sorted(self.framedata.keys()):
            #2 or more frames
            if len(self.framedata) > 1:
                prevframe = self.framedata.pop(frame_pidx)
                frame_idx = list(sorted(self.framedata.keys()))[0]


                for person in self.framedata[frame_idx]:
                    if not person in self.accu['person']:
                        self.accu['person'][person] = {}
                        self.accu['person'][person]['travel_distance_meters'] = 0
                        self.accu['person'][person]['time'] = {}
                        self.accu['person'][person]['time']['left_arm'] = {}
                        self.accu['person'][person]['time']['right_arm'] = {}
                        for item in self.items:
                            self.accu['person'][person]['time']['left_arm'][item] = 0
                            self.accu['person'][person]['time']['right_arm'][item] = 0

                    if 'left_arm' in self.framedata[frame_idx][person]:
                        for item in self.framedata[frame_idx][person]['left_arm']:
                            self.accu['person'][person]['time']['left_arm'][item] = self.accu['person'][person]['time']['left_arm'][item] + 1;
                    if 'right_arm' in self.framedata[frame_idx][person]:
                        for item in self.framedata[frame_idx][person]['right_arm']:
                            self.accu['person'][person]['time']['right_arm'][item] = self.accu['person'][person]['time']['right_arm'][item] + 1;

                    if person in prevframe and person in  self.framedata[frame_idx]:
                        dis = [0,0]
                        pp.pprint(self.framedata[frame_idx][person]['loc'])
                        dis[0] = self.framedata[frame_idx][person]['loc'][0] - prevframe[person]['loc'][0]
                        dis[1] = self.framedata[frame_idx][person]['loc'][1] - prevframe[person]['loc'][1]
                        # acount for frame jitter, a better way if I have time to implement would smoothing 
                        # and the jitter needs to increase when the score decreases
                        if dis[0] < 8:
                            dis[0] = 0
                        if dis[1] < 8:
                            dis[1] = 0
                        distance = math.sqrt(dis[0]*dis[0]*self.size['scale'][0]*self.size['scale'][0] + dis[1]*dis[1]*self.size['scale'][1]*self.size['scale'][1])
                        self.accu['person'][person]['travel_distance_meters'] = self.accu['person'][person]['travel_distance_meters'] + distance

    def point_in_items(self, point):
        matches = []
        for idx, item in enumerate(self.items):
            if point[0] < self.items[item][0][0] and point[1] < self.items[item][0][1] and self.items[item][1][0] < point[0] and self.items[item][1][1] < point[1]:
                matches.append(item)
        return matches


    def map(self, data):
        dst_image = data["image"].copy()
        data[self.dst] = dst_image

        if self.frame_num:
            self.annotate_frame_num(data)
        if self.predictions:
            self.annotate_predictions(data)
        if self.pose_flows:
            self.annotate_pose_flows(data)

        return data

    def annotate_frame_num(self, data):
        dst_image = data[self.dst]
        frame_idx = data["frame_num"]

        put_text(dst_image, f"{frame_idx:04d}", (0, 0),
                 color=colors.get("white").to_bgr(),
                 bg_color=colors.get("black").to_bgr(),
                 org_pos="tl")

    def annotate_predictions(self, data):
        if "predictions" not in data:
            return

        dst_image = data[self.dst]
        dst_image = dst_image[:, :, ::-1]  # Convert OpenCV BGR to RGB format
        predictions = data["predictions"]

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_image = self.video_visualizer.draw_panoptic_seg_predictions(dst_image,
                                                                            panoptic_seg.to(self.cpu_device),
                                                                            segments_info)
        elif "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].argmax(dim=0)
            vis_image = self.video_visualizer.draw_sem_seg(dst_image,
                                                           sem_seg.to(self.cpu_device))
        elif "instances" in predictions:
            instances = predictions["instances"]
            vis_image = self.video_visualizer.draw_instance_predictions(dst_image,
                                                                        instances.to(self.cpu_device))

        # Converts RGB format to OpenCV BGR format
        vis_image = cv2.cvtColor(vis_image.get_image(), cv2.COLOR_RGB2BGR)
        data[self.dst] = vis_image

    def annotate_pose_flows(self, data):
        if "pose_flows" not in data:
            return


        predictions = data["predictions"]
        instances = predictions["instances"]
        keypoints = instances.pred_keypoints.cpu().numpy()
        l_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6),
            (5, 7),
            (7, 9), #6 arm
            (6, 8), (8, 10),
            (6, 12), (5, 11), (11, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        dst_image = data[self.dst]
        height, width = dst_image.shape[:2]

        pose_flows = data["pose_flows"]
        pose_colors = list(colors.items())
        pose_colors_len = len(pose_colors)

        #pp.pprint(pose_flows)
        frame_num = data["frame_num"]

        #work areas
        for idx, item in enumerate(self.items):
            pose_color_idx = (idx + 20) % pose_colors_len
            pose_color_item = colors.get(pose_colors[pose_color_idx][0]).to_bgr()
            cv2.rectangle(dst_image, self.items[item][0], self.items[item][1], pose_color_item, 2, cv2.LINE_AA)

        for idx, pose_flow in enumerate(pose_flows):
            pid = pose_flow["pid"]
            if pid == 0 or pid == 1:
                worker = {'person': pid}
            elif 0 in self.framedata:
                worker = {'person': 1}
            elif 1 in self.framedata:
                worker = {'person': 0}
            else:
                #lost secondary pid, pose_flow score too low, guess pid = 1
                worker = {'person': 1}

            worker['score'] =  pose_flow["score"]
            pose_color_idx = ((pid*10) % pose_colors_len + pose_colors_len) % pose_colors_len
            pose_color_bgr = pose_colors[pose_color_idx][1].to_bgr()
            (start_x, start_y, end_x, end_y) = pose_flow["box"].astype("int")
            cv2.rectangle(dst_image, (start_x, start_y), (end_x, end_y), pose_color_bgr, 2, cv2.LINE_AA)
            put_text(dst_image, f"{worker['person']:d}", (start_x, start_y),
                     color=pose_color_bgr,
                     bg_color=colors.get("black").to_bgr(),
                     org_pos="tl")

            instance_keypoints = keypoints[idx]
            l_points = {}
            p_scores = {}
            # Draw keypoints
            for n in range(instance_keypoints.shape[0]):
                score = instance_keypoints[n, 2]
                if score <= 0.05:
                    continue
                cor_x = int(np.clip(instance_keypoints[n, 0], 0, width))
                cor_y = int(np.clip(instance_keypoints[n, 1], 0, height))
                l_points[n] = (cor_x, cor_y)
                p_scores[n] = score
                cv2.circle(dst_image, (cor_x, cor_y), 2, pose_color_bgr, -1)
            # Draw limbs
            tot_x = 0
            tot_y = 0
            for i, (start_p, end_p) in enumerate(l_pairs):
                if start_p in l_points and end_p in l_points:
                    start_xy = l_points[start_p]
                    end_xy = l_points[end_p]
                    start_score = p_scores[start_p]
                    end_score = p_scores[end_p]
                    [x,y] = np.average([start_xy, end_xy], axis=0)
                    #print(i, start_xy, end_xy)
                    if i == 6:
                        worker['left_arm'] = self.point_in_items([end_xy[0],end_xy[1]])
                    if i == 8:
                        worker['right_arm'] = self.point_in_items([end_xy[0],end_xy[1]])
                    [tot_x, tot_y] = [tot_x + x, tot_y + y]

                    cv2.line(dst_image, start_xy, end_xy, pose_color_bgr, int(2 * (start_score + end_score) + 1))

            worker['loc'] = [int(tot_x/i), int(tot_y/i)]
            if not frame_num in self.framedata:
                self.framedata[frame_num] = {}
            self.framedata[frame_num][worker['person']]  = worker

        self.calc_movement()
        pp.pprint(self.accu)


