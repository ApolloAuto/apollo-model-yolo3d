""" Inference Code """

from typing import List
from PIL import Image
import cv2
from glob import glob
import numpy as np

import torch
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule
from src.utils import Calib
from src.utils.averages import ClassAverages
from src.utils.Math import compute_orientaion, recover_angle, translation_constraints
from src.utils.Plotting import Plot3DBoxBev

import dotenv
import hydra
from omegaconf import DictConfig
import os
import pyrootutils
import src.utils
from src.utils.utils import KITTIObject

import torch.onnx
from torch.onnx import OperatorExportTypes

log = src.utils.get_pylogger(__name__)

try: 
    import onnxruntime
    import openvino.runtime as ov
except ImportError:
    log.warning("ONNX and OpenVINO not installed")

dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

class Bbox:
    def __init__(self, box_2d, label, h, w, l, tx, ty, tz, ry, alpha):
        self.box_2d = box_2d
        self.detected_class = label
        self.w = w
        self.h = h
        self.l = l
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.ry = ry
        self.alpha = alpha
        
        
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  creating new folder...  ---")
        print("---  finished  ---")
    else:
        # print("---  pass to create new folder ---")
        pass

def format_img(img, box_2d):
    # transforms
    normalize = transforms.Normalize(
        mean=[0.406, 0.456, 0.485],
        std=[0.225, 0.224, 0.229])

    process = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # crop image
    pt1, pt2 = box_2d[0], box_2d[1]

    point_list1 = [pt1[0], pt1[1]]
    point_list2 = [pt2[0], pt2[1]]
    
    if point_list1[0] < 0:
        point_list1[0] = 0
    if point_list1[1] < 0:
        point_list1[1] = 0
    if point_list2[0] < 0:
        point_list2[0] = 0
    if point_list2[1] < 0:
        point_list2[1] = 0
        
    if point_list1[0] >= img.shape[1]:
        point_list1[0] = img.shape[1] - 1
    if point_list2[0] >= img.shape[1]:
        point_list2[0] = img.shape[1] - 1
    if point_list1[1] >= img.shape[0]:
        point_list1[1] = img.shape[0] - 1
    if point_list2[1] >= img.shape[0]:
        point_list2[1] = img.shape[0] - 1
        
    crop = img[point_list1[1]:point_list2[1]+1, point_list1[0]:point_list2[0]+1]
    
    try: 
        cv2.imwrite('./tmp/img.jpg', img)

        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./tmp/demo.jpg', crop)

    except cv2.error:
        print("pt1 is ", pt1, " pt2 is ", pt2)
        print("image shape is ", img.shape)
        print("box_2d is ", box_2d)

    # apply transform for batch
    batch = process(crop)

    return batch

def inference_label(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages
    class_averages = ClassAverages()

    # initialize regressor model
    if config.get("inference_type") == "pytorch":
        # pytorch regressor model
        log.info(f"Instantiating regressor <{config.model._target_}>")
        regressor: LightningModule = hydra.utils.instantiate(config.model)
        regressor.load_state_dict(torch.load(config.get("regressor_weights"), map_location="cpu"))
        regressor.eval().to(config.get("device"))
    elif config.get("inference_type") == "onnx":
        # onnx regressor model
        log.info(f"Instantiating ONNX regressor <{config.get('regressor_weights').split('/')[-1]}>")
        regressor = onnxruntime.InferenceSession(config.get("regressor_weights"), providers=providers)
        input_name = regressor.get_inputs()[0].name
    elif config.get("inference_type") == "openvino":
        # openvino regressor model
        log.info(f"Instantiating OpenVINO regressor <{config.get('regressor_weights').split('/')[-1]}>")
        core = ov.Core()
        model = core.read_model(config.get("regressor_weights"))
        regressor = core.compile_model(model, 'CPU')
        infer_req = regressor.create_infer_request()

    # initialize preprocessing transforms
    log.info(f"Instantiating Preprocessing Transforms")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # Create output directory
    os.makedirs(config.get("output_dir"), exist_ok=True)

    # loop thru images
    imgs_path = sorted(glob(os.path.join(config.get("source_dir") + "/image_2", "*")))
    image_id = 0
    for img_path in imgs_path:
        image_id += 1
        print("\r", end="|")
        print("now is saving : {} ".format(image_id) + "/ {}".format(len(imgs_path)) + " label")
        
        # read gt image ./eval_kitti/image_2_val/
        img_id = img_path[-10:-4]
        
        # dt result
        result_label_root_path = config.get("source_dir")  + '/result/'
        mkdir(result_label_root_path)
        f = open(result_label_root_path + img_id + '.txt', 'w')
        
        # read image
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_label_root_path = config.get("source_dir") + '/label_2/'
        gt_f = gt_label_root_path + img_id + '.txt'

        dets = []
        try:
            with open(gt_f, 'r') as file:
                content = file.readlines()
                for i in range(len(content)):
                    gt = content[i].split()
                    top_left, bottom_right = (int(float(gt[4])), int(float(gt[5]))), (int(float(gt[6])), int(float(gt[7])))
                    
                    bbox_2d = [top_left, bottom_right]
                    label = gt[0]

                    dets.append(Bbox(bbox_2d, label, float(gt[8]), float(gt[9]), float(gt[10]), float(gt[11]), float(gt[12]), float(gt[13]), float(gt[14]), float(gt[3])))
        except:
            continue
        DIMENSION = []

        # loop thru detections
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det.detected_class
            if(obj.name == 'DontCare'):
                continue
            if(obj.name == 'Misc'):
                continue
            if(obj.name == 'Person_sitting'):
                continue

            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = det.box_2d[0][0], det.box_2d[0][1], det.box_2d[1][0], det.box_2d[1][1]

            crop = format_img(img, det.box_2d)

            # # preprocess img with torch.transforms
            crop = crop.reshape((1, *crop.shape)).to(config.get("device"))

            # regress 2D bbox with Regressor
            if config.get("inference_type") == "pytorch":
                [orient, conf, dim] = regressor(crop)

                orient = orient.cpu().detach().numpy()[0, :, :]
                conf = conf.cpu().detach().numpy()[0, :]
                dim = dim.cpu().detach().numpy()[0, :]

            # dimension averages
            try:
                dim += class_averages.get_item(obj.name)
                DIMENSION.append(dim)
            except:
                dim = DIMENSION[-1]
            
            obj.alpha = recover_angle(orient, conf, 2)
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # output prediction label
            obj.score = 1.0
            output_line = obj.member_to_list()
            output_line = " ".join([str(i) for i in output_line])
            
            f.write(output_line + '\n')
        f.close()

def inference_image(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages
    class_averages = ClassAverages()
    
    export_onnx = config.get("export_onnx")

    # initialize regressor model
    if config.get("inference_type") == "pytorch":
        # pytorch regressor model
        log.info(f"Instantiating regressor <{config.model._target_}>")
        regressor: LightningModule = hydra.utils.instantiate(config.model)
        regressor.load_state_dict(torch.load(config.get("regressor_weights"), map_location="cpu"))
        regressor.eval().to(config.get("device"))
    elif config.get("inference_type") == "onnx":
        # onnx regressor model
        log.info(f"Instantiating ONNX regressor <{config.get('regressor_weights').split('/')[-1]}>")
        regressor = onnxruntime.InferenceSession(config.get("regressor_weights"), providers=providers)
        input_name = regressor.get_inputs()[0].name
    elif config.get("inference_type") == "openvino":
        # openvino regressor model
        log.info(f"Instantiating OpenVINO regressor <{config.get('regressor_weights').split('/')[-1]}>")
        core = ov.Core()
        model = core.read_model(config.get("regressor_weights"))
        regressor = core.compile_model(model, 'CPU')
        infer_req = regressor.create_infer_request()

    # initialize preprocessing transforms
    log.info(f"Instantiating Preprocessing Transforms")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # Create output directory
    os.makedirs(config.get("output_dir"), exist_ok=True)

    imgs_path = sorted(glob(os.path.join(config.get("source_dir") + "/image_2", "*")))
    image_id = 0
    for img_path in imgs_path:
        image_id += 1
        print("\r", end="|")
        print("now is saving : {} ".format(image_id) + "/ {}".format(len(imgs_path)) + " image")
        
        # Initialize object and plotting modules
        plot3dbev = Plot3DBoxBev(P2)

        img_name = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # check if image shape 1242 x 375
        if img.shape != (375, 1242, 3):
            # crop center of image to 1242 x 375
            src_h, src_w, _ = img.shape
            dst_h, dst_w = 375, 1242
            dif_h, dif_w = src_h - dst_h, src_w - dst_w
            img = img[dif_h // 2 : src_h - dif_h // 2, dif_w // 2 : src_w - dif_w // 2, :]

        img_id = img_path[-10:-4]
        
        # read image
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_label_root_path = config.get("source_dir") + '/label_2/'
        gt_f = gt_label_root_path + img_id + '.txt'

        # use gt 2d result as output of first stage 
        dets = []
        try:
            with open(gt_f, 'r') as file:
                content = file.readlines()
                for i in range(len(content)):
                    gt = content[i].split()
                    top_left, bottom_right = (int(float(gt[4])), int(float(gt[5]))), (int(float(gt[6])), int(float(gt[7])))
                    
                    bbox_2d = [top_left, bottom_right]
                    label = gt[0]

                    dets.append(Bbox(bbox_2d, label, float(gt[8]), float(gt[9]), float(gt[10]), float(gt[11]), float(gt[12]), float(gt[13]), float(gt[14]), float(gt[3])))
        except:
            continue
        DIMENSION = []
    
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det.detected_class
            if(obj.name == 'DontCare'):
                continue
            if(obj.name == 'Misc'):
                continue
            if(obj.name == 'Person_sitting'):
                continue
            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = det.box_2d[0][0], det.box_2d[0][1], det.box_2d[1][0], det.box_2d[1][1]

            crop = format_img(img, det.box_2d)

            crop = crop.reshape((1, *crop.shape)).to(config.get("device"))

            # regress 2D bbox with Regressor
            if config.get("inference_type") == "pytorch":
                [orient, conf, dim] = regressor(crop)
                orient = orient.cpu().detach().numpy()[0, :, :]
                conf = conf.cpu().detach().numpy()[0, :]
                dim = dim.cpu().detach().numpy()[0, :]


            if(export_onnx):
                traced_script_module = torch.jit.trace(regressor, (crop))
                traced_script_module.save("weights/yolo_libtorch_model_3d.pth")
                
                onnx_model_save_path = "weights/yolo_onnx_model_3d.onnx"
                # dynamic batch
                # dynamic_axes = {"image": {0: "batch"}, 
                #                 "orient": {0: "batch", 1: str(2), 2: str(2)}, # for multi batch
                #                 "conf": {0: "batch"}, 
                #                 "dim": {0: "batch"}}
                if True:
                    torch.onnx.export(regressor, crop, onnx_model_save_path, opset_version=11,
                                verbose=False, export_params=True, operator_export_type=OperatorExportTypes.ONNX,
                                input_names=['image'], output_names=['orient','conf','dim']
                                # ,dynamic_axes=dynamic_axes
                                )
                    print("Please check onnx model in ", onnx_model_save_path)
                
                import onnx
                onnx_model = onnx.load(onnx_model_save_path)
                
                # for dla&trt speedup
                onnx_fp16_model_save_path = "weights/yolo_onnx_model_3d_fp16.onnx"
                from onnxmltools.utils import float16_converter
                trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
                onnx.save_model(trans_model, onnx_fp16_model_save_path)

                export_onnx = False # once
            
            try:
                dim += class_averages.get_item(obj.name)
                DIMENSION.append(dim)
            except:
                dim = DIMENSION[-1]

            obj.alpha = recover_angle(orient, conf, 2)
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # output prediction label
            output_line = obj.member_to_list()
            output_line.append(1.0)
            output_line = " ".join([str(i) for i in output_line]) + "\n"

            # save results
            if config.get("save_txt"):
                with open(f"{config.get('output_dir')}/{img_name}.txt", "a") as f:
                    f.write(output_line)

            if config.get("save_result"):
                # dt
                plot3dbev.plot(
                    img=img,
                    class_object=obj.name.lower(),
                    bbox=[obj.xmin, obj.ymin, obj.xmax, obj.ymax],
                    dim=[obj.h, obj.w, obj.l],
                    loc=[obj.tx, obj.ty, obj.tz],
                    rot_y=obj.rot_global,
                    gt=False
                )
                # gt
                plot3dbev.plot(
                    img=img,
                    class_object=obj.name.lower(),
                    bbox=[obj.xmin, obj.ymin, obj.xmax, obj.ymax],
                    dim=[det.h, det.w, det.l],
                    loc=[det.tx, det.ty, det.tz],
                    rot_y=det.ry,
                    gt=True
                )
        # save images
        if config.get("save_result"):
            plot3dbev.save_plot(config.get("output_dir"), img_name)

def copy_eval_label():
    label_path = './data/KITTI/ImageSets/val.txt'
    label_root_path = './data/KITTI/label_2/'
    label_save_path = './data/KITTI/label_2_val/'

    # get all labels
    label_files = []
    sum_number = 0
    from shutil import copyfile

    with open(label_path, 'r') as file:
        img_id = file.readlines()
        for id in img_id:
            label_path = label_root_path + id[:6] + '.txt'
            copyfile(label_path, label_save_path + id[:6] + '.txt')

def copy_eval_image():
    label_path = './data/KITTI/ImageSets/val.txt'
    img_root_path = './data/KITTI/image_2/'
    img_save_path = './data/KITTI/image_2_val'

    # get all labels
    label_files = []
    sum_number = 0
    with open(label_path, 'r') as file:
        img_id = file.readlines()
        for id in img_id:
            img_path = img_root_path + id[:6] + '.png'
            img = cv2.imread(img_path)
            cv2.imwrite(f'{img_save_path}/{id[:6]}.png', img)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def main(config: DictConfig):
    if(config.get("func") == "image"):
        # inference_image: 
        # inference for kitti bev and 3d image, without model
        inference_image(config)
    else:
        # inference_label:
        # for kitti gt label, predict without model
        inference_label(config)

if __name__ == "__main__":

    # # tools for copy target files
    # copy_eval_label()
    # copy_eval_image()
    
    main()