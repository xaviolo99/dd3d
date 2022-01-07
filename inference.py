from copy import deepcopy
import time
from typing import List, Union

from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.data.common import MapDataset
from detectron2.modeling import build_model
from fvcore.common.checkpoint import Checkpointer
from hydra import initialize, compose
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient
import torch
from torch.utils.data import Dataset, DataLoader

from kluster import Kluster
from panoramator import Projection, Panoramator, mongo_to_shards

from tridet.data.augmentations import build_augmentation
from tridet.evaluators.kitti_3d_evaluator import convert_3d_box_to_kitti
from tridet.structures.pose import Pose
from tridet.utils.geometry import project_points3d
from tridet.utils.setup import setup
from tridet.utils.tasks import TaskManager


# Panoramator structures

class PanoramaDataset(Dataset):

    def __init__(self, mongo_args, segments, keyword, projections):
        kluster = Kluster(session=MongoClient(*mongo_args))
        segments = kluster.fetch_data(
            "segments",
            {"_id": {"$in": segments}, "street_view": {"$elemMatch": {"available": True, keyword: {"$exists": False}}}}
        )
        self.kluster = mongo_args
        lines = [
            (segment["_id"], i, line["panoramas"])
            for segment in segments for i, line in enumerate(segment["street_view"])
            if "available" in line and keyword not in line
        ]
        self.panoramas = [(sid, lidx, pidx, panorama)
                          for sid, lidx, panoramas in lines for pidx, panorama in enumerate(panoramas)]
        self.projections = projections

    def __len__(self):
        return len(self.panoramas)

    def __getitem__(self, idx):
        if type(self.kluster) == tuple:
            self.kluster = Kluster(session=MongoClient(*self.kluster))
        segment_id, line_idx, panorama_idx, panorama_id = self.panoramas[idx]
        panorama = self.kluster.kluster["street_view"].find_one({"_id": panorama_id})
        shards = mongo_to_shards(panorama["panorama"])
        panoramator = Panoramator(shards=shards, atomic_resolution=panorama["resolution"][0] // 16)
        panoramator.build_state()
        projections = [(projection_meta, panoramator.get_projection(projection_meta))
                       for projection_meta in self.projections]
        return segment_id, line_idx, panorama_id, projections


def inference(kluster, predictor, data_loader, keyword):
    current_line = None
    line_count = 0

    for i, (segment_id, line_idx, panorama_id, projections) in enumerate(data_loader):
        itime = time.time()

        if current_line is not None and current_line != (segment_id, line_idx):
            sid, lidx = current_line
            kluster.kluster["segments"].update_one({"_id": sid}, {"$set": {f"street_view.{lidx}.{keyword}": True}})
            line_count += 1
            print(f"Finished line {line_count}! (Segment:{sid};Index:{lidx})")
        current_line = (segment_id, line_idx)

        result = []
        for projection_meta, projection in projections:
            predictions = predictor(projection)
            result.append({"projection": projection_meta.get_dict(), **predictions})
        kluster.kluster["street_view"].update_one({"_id": panorama_id}, {"$set": {keyword: result}})

        print(f"Predicted panorama {i+1}/{len(data_loader)} (Time elapsed: {time.time()-itime:.2f}s) ({panorama_id})")


# DD3D structures

class ParkinkDatasetMapper:

    @configurable
    def __init__(self, is_train: bool, task_manager, augmentations: List[Union[T.Augmentation, T.Transform]],
                 image_format: str, intrinsics: list, extrinsics: dict):
        self.is_train = is_train
        self.task_manager = task_manager
        self.augmentations = T.AugmentationList(augmentations)
        print("Augmentations used: " + str(augmentations))
        self.image_format = image_format
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    @classmethod
    def from_config(cls, cfg, is_train, intrinsics, extrinsics):
        augs = build_augmentation(cfg, is_train)
        tm = TaskManager(cfg)
        return {"is_train": is_train, "task_manager": tm, "augmentations": augs, "image_format": cfg.INPUT.FORMAT,
                "intrinsics": intrinsics, "extrinsics": extrinsics}

    def __call__(self, parkink_data):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        segment_id, line_idx, panorama_id, projections = parkink_data

        kitti_projections = []
        for projection_meta, image in projections:
            kitti = {"width": image.shape[1], "height": image.shape[0],
                     "intrinsics": self.intrinsics, "extrinsics": self.extrinsics}

            if type(image) == torch.Tensor:  # When using a DataLoader, Tensors instead of arrays will be given
                image = image.numpy()
            image = image[:, :, ::-1]  # VERY IMPORTANT! CONVERT IMAGE FROM RGB (PIL format) TO BGR (model format)
            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            kitti["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            intrinsics = np.reshape(kitti["intrinsics"], (3, 3)).astype(np.float32)
            intrinsics = transforms.apply_intrinsics(intrinsics)
            kitti["intrinsics"] = torch.as_tensor(intrinsics)
            kitti["inv_intrinsics"] = torch.as_tensor(np.linalg.inv(intrinsics))

            extrinsics = Pose(wxyz=np.float32(kitti["extrinsics"]["wxyz"]),
                              tvec=np.float32(kitti["extrinsics"]["tvec"]))
            kitti["extrinsics"] = extrinsics

            kitti_projections.append((projection_meta, kitti))

        return segment_id, line_idx, panorama_id, kitti_projections


def meter_to_angle(x, y, z):
    # Convert meters coordinates to horizontal and vertical angles.
    # We negate the vertical and so that up is positive and down is negative.
    return np.array([np.arctan2(x, z), -np.arctan2(y, z)]) / np.pi * 180


def process_scene(model, input_dict, plot=False, log=False):
    CLASS_MAPPER = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")
    THRESHOLD = 0.5
    with torch.no_grad():
        raw_output = model([input_dict])[0]
    instances = raw_output["instances"].get_fields()

    # We discard
    # - instances["scores"]: scores_3d gives a more informed score, taking into account the 3d box
    # - instances["locations"]: this is ~= object center, but the center given by 3d box is more accurate
    # - instances["pred_boxes"]: 2d boxes, a priori useless if we have 3d ones (is this an intermediate step?)
    # - instances["fpn_levels"]: This is related to at which level of the net the object is detected... useless
    zipped = zip(instances["scores_3d"], instances["pred_classes"], instances["pred_boxes3d"])
    subd = {"pixels": [], "meters": [], "degrees": []}
    prediction = {"score": [], "kitti_class": [], "size": [], "orientation": [], "center": deepcopy(subd),
                  "front_upper_left": deepcopy(subd), "front_upper_right": deepcopy(subd),
                  "front_lower_right": deepcopy(subd), "front_lower_left": deepcopy(subd),
                  "back_upper_left": deepcopy(subd), "back_upper_right": deepcopy(subd),
                  "back_lower_right": deepcopy(subd), "back_lower_left": deepcopy(subd), "floor": deepcopy(subd)}
    for score_3d, kitti_class, box_3d in zipped:
        if score_3d < THRESHOLD:  # If the model is not confident enough, we skip the prediction
            continue
        prediction["score"].append(round(score_3d.item(), 3))
        if kitti_class not in (0, 3, 4):  # If the detected object is not a car, van or truck, we skip it
            continue
        kitti_class = CLASS_MAPPER[kitti_class]
        prediction["kitti_class"].append(kitti_class)

        center_pix = box_3d.proj_ctr[0].cpu().numpy()  # width (x), height (y)
        center_met = box_3d.tvec[0].cpu().numpy()  # horizontal (left->right), vertical (up->down), depth (back->front)
        center_ang = meter_to_angle(*center_met)  # horizontal (left->right, degrees), vertical (down->up, degrees)
        prediction["center"]["pixels"].append([round(e, 1) for e in center_pix.tolist()])
        prediction["center"]["meters"].append([round(e, 2) for e in center_met.tolist()])
        prediction["center"]["degrees"].append([round(e, 2) for e in center_ang.tolist()])

        size = box_3d.size[0].cpu().numpy()  # width, length, height (meters)
        prediction["size"].append([round(e, 2) for e in size.tolist()])

        floor_met = center_met + np.array([0, size[2]/2, 0])
        floor_ang = meter_to_angle(*floor_met)
        floor_pix = project_points3d(np.array([floor_met]), input_dict["intrinsics"].numpy())[0]
        prediction["floor"]["pixels"].append([round(e, 1) for e in floor_pix.tolist()])
        prediction["floor"]["meters"].append([round(e, 2) for e in floor_met.tolist()])
        prediction["floor"]["degrees"].append([round(e, 2) for e in floor_ang.tolist()])

        corners_met = box_3d.corners[0].cpu().numpy()
        corners_ang = np.array([meter_to_angle(*corner) for corner in corners_met])
        corners_pix = project_points3d(corners_met, input_dict["intrinsics"].numpy())
        corners_pix = [pix * (-1 if met[2] < 0 else 1) for met, pix in zip(corners_met, corners_pix)]
        keys = ["front_upper_left", "front_upper_right", "front_lower_right", "front_lower_left",
                "back_upper_left", "back_upper_right", "back_lower_right", "back_lower_left"]
        for key, pix, met, ang in zip(keys, corners_pix, corners_met, corners_ang):
            prediction[key]["pixels"].append([round(e, 1) for e in pix.tolist()])
            prediction[key]["meters"].append([round(e, 2) for e in met.tolist()])
            prediction[key]["degrees"].append([round(e, 2) for e in ang.tolist()])

        w, l, h, x, y, z, roty, alpha = convert_3d_box_to_kitti(box_3d)
        orientation = - alpha / np.pi * 180  # The alpha in angles.png, clockwise is positive (180 to -180 range) (90 means we see car back) (-90 means we see car front)
        prediction["orientation"].append(round(orientation, 2))

        if log:
            print(f"Confidence: {score_3d}")
            print(f"Class: {kitti_class}")
            print(f"Center (pixels): {center_pix}")
            print(f"Center (meters): {center_met}")
            print(f"Center (degrees): {center_ang}")
            print(f"Size (meters): {size}")
            print(f"Floor (pixels): {floor_pix}")
            print(f"Floor (meters): {floor_met}")
            print(f"Floor (degrees): {floor_ang}")
            print(f"Corners (pixels): {corners_pix}")
            print(f"Corners (meters): {corners_met}")
            print(f"Corners (degrees): {corners_ang}")
            print(f"Car Orientation (degrees): {orientation}")

        if plot:
            for a, b, c, d in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3), (1, 5, 6, 2), (0, 2, 1, 3)]:
                coord = [corners_pix[a], corners_pix[b], corners_pix[c], corners_pix[d], corners_pix[a]]
                xs, ys = zip(*coord)
                plt.plot(xs, ys, color='r')

    if plot:
        img = input_dict["image"].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.imshow(img)
        plt.show()

    return prediction


# Constants

CFG_PATH = "configs/"
CFG_NAME = "kitti99_defaults"
CHECKPOINT = "models/kitti_v99.pth"
INTRINSICS = [612.6, 0.0, 640.0, 0.0, 612.6, 128.0, 0.0, 0.0, 1.0]
EXTRINSICS = {"wxyz": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}
MONGO_SESSION_ARGS = ("localhost", 27017)
PREDICTION_KEYWORD = "kitti_cars"
TIMEOUT = 180
"""
PROJECTIONS = [Projection(center_horizontal=0, center_vertical=0, fov_horizontal=92.5, fov_vertical=45.36,
                          full_resolution_x=1280, full_resolution_y=512,
                          offset_x=0, offset_y=512-384, resolution_x=1280, resolution_y=384),
               Projection(center_horizontal=180, center_vertical=0, fov_horizontal=92.5, fov_vertical=45.36,
                          full_resolution_x=1280, full_resolution_y=512,
                          offset_x=0, offset_y=512-384, resolution_x=1280, resolution_y=384)]
"""
PROJECTIONS = [Projection(center_horizontal=0, center_vertical=-1, fov_horizontal=82.6, fov_vertical=38.7,
                          full_resolution_x=1280, full_resolution_y=384,
                          offset_x=0, offset_y=0, resolution_x=1280, resolution_y=384),
               Projection(center_horizontal=180, center_vertical=-1, fov_horizontal=82.6, fov_vertical=38.7,
                          full_resolution_x=1280, full_resolution_y=384,
                          offset_x=0, offset_y=0, resolution_x=1280, resolution_y=384)]  # 546.7
MIN_LAT, MAX_LAT = 41.35, 41.5
MIN_LON, MAX_LON = 2.1, 2.3
PLOT = False
LOG = False


# Main Execution

if __name__ == "__main__":
    # StreetView initializations
    main_kluster = Kluster(session=MongoClient(*MONGO_SESSION_ARGS))
    bounding_polygon = [(MIN_LAT, MIN_LON), (MIN_LAT, MAX_LON), (MAX_LAT, MAX_LON),
                        (MAX_LAT, MIN_LON), (MIN_LAT, MIN_LON)]
    bounding_polygon = {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in bounding_polygon]]}

    # DD3D initializations
    with initialize(config_path=CFG_PATH):
        cfg = compose(config_name=CFG_NAME)
    setup(cfg)
    dd3d_model = build_model(cfg).eval()
    Checkpointer(dd3d_model).load(CHECKPOINT)
    dd3d_predictor = lambda image: process_scene(dd3d_model, image, plot=PLOT, log=LOG)

    # Load segment_ids of interest
    ways = main_kluster.fetch_data("ways", {"path": {"$geoIntersects": {"$geometry": bounding_polygon}}})
    segment_ids = [seg_id for way in ways for seg_id in way["segments"].values()]

    # Do the inference, and when it finishes keep looking for new panoramas
    while True:
        dataset = PanoramaDataset(MONGO_SESSION_ARGS, segment_ids, PREDICTION_KEYWORD, PROJECTIONS)
        mapper = ParkinkDatasetMapper(cfg, is_train=False, intrinsics=INTRINSICS, extrinsics=EXTRINSICS)
        dataset = MapDataset(dataset, mapper)
        if len(dataset):
            print(f"LAUNCHING INFERENCE ON {len(dataset)} PANORAMAS")
            loader = DataLoader(dataset, batch_size=None, num_workers=4)
            inference(main_kluster, dd3d_predictor, loader, PREDICTION_KEYWORD)
        else:
            print(f"NO PANORAMAS FOUND! WAITING {TIMEOUT} seconds...")
            time.sleep(180)
