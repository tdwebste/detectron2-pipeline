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
import time
from operator import itemgetter
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

        self.size = {}
        self.size['image'] = [0,1]
        self.size['default_image'] = [0,1]
        self.size['default_image'][0] = 1920
        self.size['default_image'][1] = 1080
        self.size['room'] = [14, 16] #size in feet
        self.size['room'][0] = self.size['room'][0]/3.28084
        self.size['room'][1] = self.size['room'][1]/3.28084
        self.size['image'][0] = 1920 #default image_width in pixes
        self.size['image'][1] = 1080 #default image_hieght in pixes
        self.size['scale'] = [0,1]
        #TODO jitter should be scaled to frame fps and image resolution
        self.size['jitter'] = [0,1]
        self.size['jitter'][0] = 6
        self.size['jitter'][1] = 6

        self.ppid = {}

        self.framedata = {}
        self.accu = {}
        self.locpt = {}
        self.locpt['person'] = {}
        self.accu['person'] = {}
        self.items = {
            'counter': [[ 1100,500], [820,340]],
            'doors': [[ 1150,550], [1090,400]],
            'dish': [[ 950,330], [850,270]],
            'gloves': [[ 950,380], [850,350]],
            'avocado': [[ 920,470], [810,380]],
        }

        #possible hma periods = 4, 16, 32
        #TODO:
        # averaging periods should be scaled with frame fps and image resolution.
        self.hma_period = 16
        self.pt_period = 16
        self.hma = []
        self.first_pass = True
        super().__init__()

    def first_wma(self, pts, period, idx, person, axis):
        """
        Weighted Moving Average.

        Formula:
        (P1 + 2 P2 + 3 P3 + ... + n Pn) / K
        where K = (1+2+...+n) = n(n+1)/2 and Pn is the most recent price
        """
        k = (period * (period + 1)) / 2.0
        product = [pts[idx - period + period_idx + 1] * (period_idx + 1) for period_idx in range(0, period)]
        wma = sum(product)/k
        return wma

    def hull_moving_average(self, period, idx, person, axis):
        #hull moving average reduces position latence caused by averaging
        #TODO use hull moving average so that limb position and averaged location are closely aligned.
        """
        Hull Moving Average.

        Formula:
        HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        """
        pts = self.locpt['person'][person]['loc'][axis]
        fwma = self.first_wma

        hwma = fwma(pts, int(period/2), idx, person, axis)
        rwma = fwma(pts, period, idx, person, axis)
        return rwma

        self.wma['person'][person][axis]['half'].append(hwma)
        self.wma['person'][person][axis]['reg'].append(rwma)

        if len(self.wma['person'][person][axis]['half']) < int(math.sqrt(period)) + 1:
            return self.wma['person'][person][axis]['reg'][-1]
        else:
            wma1 = [2* self.wma['person'][person][axis]['half'][period_idx] - self.wma['person'][person][axis]['reg'][period_idx]
                for period_idx in range(0,  int(math.sqrt(period))+1)]
            #print("wma1")
            #pp.pprint(wma1)
            fwma = self.first_wma

            hma = fwma(wma1, int(math.sqrt(period)), int(math.sqrt(period)), person, axis)
            return hma

    def calc_movement(self):
        for frame_idx in sorted(self.framedata.keys()):
            for person in self.framedata[frame_idx]:
                if not person in self.locpt['person']:
                    self.locpt['person'][person] = {}
                    self.locpt['person'][person]['loc'] = [0,1]
                    self.locpt['person'][person]['loc'][0] = {}
                    self.locpt['person'][person]['loc'][1] = {}
                    self.locpt['person'][person]['hwa']= [0,1]
                    self.locpt['person'][person]['hwa'][0] = {}
                    self.locpt['person'][person]['hwa'][1] = {}
                if not person in self.accu['person']:
                    self.accu['person'][person] = {}
                    self.accu['person'][person]['travel_distance_meters'] = 0
                    self.accu['person'][person]['time'] = {}
                    self.accu['person'][person]['time']['left_arm'] = {}
                    self.accu['person'][person]['time']['right_arm'] = {}
                    for item in self.items:
                        self.accu['person'][person]['time']['left_arm'][item] = 0
                        self.accu['person'][person]['time']['right_arm'][item] = 0

                #Count frames limb interacting with defined kitchen item areas
                if 'left_arm' in self.framedata[frame_idx][person]:
                    for item in self.framedata[frame_idx][person]['left_arm']:
                        self.accu['person'][person]['time']['left_arm'][item] = self.accu['person'][person]['time']['left_arm'][item] + 1;
                if 'right_arm' in self.framedata[frame_idx][person]:
                    for item in self.framedata[frame_idx][person]['right_arm']:
                        self.accu['person'][person]['time']['right_arm'][item] = self.accu['person'][person]['time']['right_arm'][item] + 1;

                self.locpt['person'][person]['loc'][0][frame_idx] = self.framedata[frame_idx][person]['loc'][0]
                self.locpt['person'][person]['loc'][1][frame_idx] = self.framedata[frame_idx][person]['loc'][1]

                #determine average person position
                #Experments determine the best apporch.
                # 2 apporaches are used here.

                # Weighted Moving average, which assumed continuous seqeunce of frames
                #    Has less delay by averaging the box car filter.
                # box car averaging filter which does not assume a continuous sequence of frames.
                #    The box car averaging filter is more suited for gaps in video frame sequence.
                #    But causes more delay by averaging.

                # Hull weight average weight average has the least delay, but is the most sensative to video frame gaps.
                pt_len = len(self.locpt['person'][person]['loc'][0])
                dis = [0,1]
                in_range = True
                for t_idx in range (frame_idx - self.hma_period, frame_idx):
                    if not t_idx in self.locpt['person'][person]['loc'][0]:
                        in_range = False
                if in_range:
                    self.locpt['person'][person]['hwa'][0][frame_idx] = self.first_wma(self.locpt['person'][person]['loc'][0], self.hma_period, frame_idx, person, 0)
                    self.locpt['person'][person]['hwa'][1][frame_idx] = self.first_wma(self.locpt['person'][person]['loc'][1], self.hma_period, frame_idx, person, 1)
                    if frame_idx in self.locpt['person'][person]['hwa'][0] and (frame_idx - 1) in self.locpt['person'][person]['hwa'][0]:
                        dis[0] = self.locpt['person'][person]['hwa'][0][frame_idx] - self.locpt['person'][person]['hwa'][0][frame_idx - 1]
                        dis[1] = self.locpt['person'][person]['hwa'][1][frame_idx] - self.locpt['person'][person]['hwa'][1][frame_idx - 1]
                        del(self.locpt['person'][person]['hwa'][0][frame_idx - 1])
                        del(self.locpt['person'][person]['hwa'][1][frame_idx - 1])

                    if len(self.locpt['person'][person]['hwa'][0]) >= self.hma_period:
                        t_idx = sorted(self.framedata.keys())[0]
                        del(self.locpt['person'][person]['hwa'][0][t_idx])
                        del(self.locpt['person'][person]['hwa'][1][t_idx])
                elif pt_len > self.pt_period+1: #box filter
                    pt_len = len(self.locpt['person'][person]['loc'][0])
                    pt_lowl = pt_len - (self.pt_period+1)
                    pt_lowh = pt_len - 1
                    pt_highl = pt_len - self.pt_period
                    pt_highh = pt_len
                    ptot_low = [0,1]
                    ptot_high = [0,1]
                    ptav_low = [0,1]
                    ptav_high = [0,1]
                    for taxis in [0,1]:
                        ptot_low[taxis] = 0
                        ptot_high[taxis] = 0
                        #handles non-continous points
                        for cnt, tidx in enumerate(sorted(self.locpt['person'][person]['loc'][taxis])):
                            if cnt < pt_lowl:
                                continue
                            elif cnt < pt_highl:
                                ptot_low[taxis] = ptot_low[taxis] + self.locpt['person'][person]['loc'][taxis][tidx]
                            elif cnt < pt_lowh:
                                ptot_low[taxis] = ptot_low[taxis] + self.locpt['person'][person]['loc'][taxis][tidx]
                                ptot_high[taxis] = ptot_high[taxis] + self.locpt['person'][person]['loc'][taxis][tidx]
                            elif cnt < pt_highh:
                                ptot_high[taxis] = ptot_high[taxis] + self.locpt['person'][person]['loc'][taxis][tidx]

                        ptav_low[taxis] = ptot_low[taxis]/self.pt_period
                        ptav_high[taxis] = ptot_high[taxis]/self.pt_period

                        dis[taxis] = ptav_high[taxis] - ptav_low[taxis]

                    if abs(dis[0]) < self.size['jitter'][0]:
                        dis[0] = 0
                    if abs(dis[1]) < self.size['jitter'][1]:
                        dis[1] = 0
                if (frame_idx - self.hma_period) in self.locpt['person'][person]['loc'][0]:
                    del(self.locpt['person'][person]['loc'][0][frame_idx - self.hma_period])
                    del(self.locpt['person'][person]['loc'][1][frame_idx - self.hma_period])

                if abs(dis[0]) < self.size['jitter'][0]:
                    dis[0] = 0
                if abs(dis[1]) < self.size['jitter'][1]:
                    dis[1] = 0
                distance = math.sqrt(dis[0]*dis[0]*self.size['scale'][0]*self.size['scale'][0] + dis[1]*dis[1]*self.size['scale'][1]*self.size['scale'][1])
                self.accu['person'][person]['travel_distance_meters'] = self.accu['person'][person]['travel_distance_meters'] + distance


            prevframe = self.framedata.pop(frame_idx)



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
        if self.first_pass:
            for axis in [0,1]:
                self.size['image'][axis] =  instances.image_size[(axis+1)%2]
                self.size['scale'][axis] = self.size['room'][axis] / self.size['image'][axis]
                self.size['jitter'][axis] = int(self.size['jitter'][axis]/self.size['default_image'][axis]*self.size['image'][axis])

                for idx, item in enumerate(self.items):
                    self.items[item][0][axis] = int(self.items[item][0][axis]/self.size['default_image'][axis]*self.size['image'][axis])
                    self.items[item][1][axis] = int(self.items[item][1][axis]/self.size['default_image'][axis]*self.size['image'][axis])
            print(f"image_width = {self.size['image'][0]}\nimage_height = {self.size['image'][1]}")
            pp.pprint(self.items)
            self.first_pass = False


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

        frame_num = data["frame_num"]

        #work areas
        for idx, item in enumerate(self.items):
            pose_color_idx = (idx + 20) % pose_colors_len
            pose_color_item = colors.get(pose_colors[pose_color_idx][0]).to_bgr()
            cv2.rectangle(dst_image, tuple(self.items[item][0]), tuple(self.items[item][1]), pose_color_item, 2, cv2.LINE_AA)


        if not frame_num in self.framedata:
            self.framedata[frame_num] = {}
        sorted_pose = sorted(pose_flows, key=itemgetter('score'), reverse = True)
        for idx, pose_flow in enumerate(sorted_pose):
            pid = pose_flow["pid"]
            if 0 in self.framedata[frame_num] and 1 in self.framedata[frame_num]:  # assume set known person is known in advance to be set([0, 1])
                                        # only way people are added is at
                                        # at the beginning of video
                                        # or arriving through the room entrences.
                continue
            elif pid in self.locpt['person']: #known person
                if pose_flow['score'] > 2: # high score, previouly seen
                    worker = {'person': pid}
                elif pose_flow['score'] > .5:  # low score, previouly seen
                    worker = {'person': pid}
                elif pid in self.framedata[frame_num]: #low score, and already used
                    continue
                elif pose_flow['score'] < .5:  # low score, previouly seen
                    worker = {'person': pid}
                else:
                    no_op = 0
            elif pose_flow['score'] > 2:
                if pid == 0 or pid == 1: #new addition
                                        # assume set known person is known in advance to be set([0, 1])
                                        # only way people are added is at
                                        # at the beginning of video
                                        # or arriving through the room entrences.
                    worker = {'person': pid}
                elif 0 in self.framedata[frame_num]: # assume set known person is known in advance to be set([0, 1])
                                        # only way people are added is at
                                        # at the beginning of video
                                        # or arriving through the room entrences.
                    worker = {'person': 1}
                    self.ppid[pid] = 1
                elif 1 in self.framedata[frame_num]:
                    worker = {'person': 0}
                    self.ppid[pid] = 0
                elif pid in self.ppid:
                    worker = {'person': self.ppid[pid]}
                else:
                    worker = {'person': 1}
            elif pose_flow['score'] > .5:  # low score, previouly seen
                if pid == 0 or pid == 1: #new addition
                                        # assume set known person is known in advance to be set([0, 1])
                                        # only way people are added is at
                                        # at the beginning of video
                                        # or arriving through the room entrences.
                    worker = {'person': pid}
                else:
                    if 0 in self.framedata[frame_num]: # assume set known person is known in advance to be set([0, 1])
                                        # only way people are added is at
                                        # at the beginning of video
                                        # or arriving through the room entrences.
                        worker = {'person': 1}
                        self.ppid[pid] = 1
                    elif 1 in self.framedata[frame_num]:
                        worker = {'person': 0}
                        self.ppid[pid] = 0
                    elif pid in self.ppid:
                        worker = {'person': self.ppid[pid]}
                    else:
                        worker = {'person': 1}
            elif pose_flow['score'] < .5:  # low score, previouly seen
                continue



            worker['score'] =  pose_flow["score"]
            pose_color_idx = ((pid*10) % pose_colors_len + pose_colors_len) % pose_colors_len
            pose_color_bgr = pose_colors[pose_color_idx][1].to_bgr()
            (start_x, start_y, end_x, end_y) = pose_flow["box"].astype("int")
            cv2.rectangle(dst_image, (start_x, start_y), (end_x, end_y), pose_color_bgr, 2, cv2.LINE_AA)
            tot_x = (start_x + end_x)/2
            tot_y = (start_y + end_y)/2
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
            for i, (start_p, end_p) in enumerate(l_pairs):
                if start_p in l_points and end_p in l_points:
                    start_xy = l_points[start_p]
                    end_xy = l_points[end_p]
                    start_score = p_scores[start_p]
                    end_score = p_scores[end_p]
                    [x,y] = np.average([start_xy, end_xy], axis=0)
                    if i == 6:
                        worker['left_arm'] = self.point_in_items([end_xy[0],end_xy[1]])
                    if i == 8:
                        worker['right_arm'] = self.point_in_items([end_xy[0],end_xy[1]])

                    cv2.line(dst_image, start_xy, end_xy, pose_color_bgr, int(2 * (start_score + end_score) + 1))

            worker['loc'] = [int(tot_x), int(tot_y)]
            self.framedata[frame_num][worker['person']]  = worker


        self.calc_movement()
        pp.pprint(self.accu)


