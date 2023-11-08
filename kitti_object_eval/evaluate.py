import time
import fire
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,  # gt
             result_path,  # dt
             label_split_file,
             current_class=0,  # 0: bbox, 1: bev, 2: 3d
             coco=False,
             score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    # print("dt_annos[0] is ", dt_annos[0], " shape is ", len(dt_annos))

    # if score_thresh > 0:
    #     dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    # val_image_ids = _read_imageset_file(label_split_file)

    gt_annos = kitti.get_label_annos(label_path)
    # print("gt_annos[0] is ", gt_annos[0], " shape is ", len(gt_annos))

    if coco:
        print(get_coco_eval_result(gt_annos, dt_annos, current_class))
    else:
        print("not coco")
        print(get_official_eval_result(gt_annos, dt_annos, current_class))


if __name__ == '__main__':
    fire.Fire()
