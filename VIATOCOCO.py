'''
源文件是mmdetection 帮助文件中将balloon数据集转换为coco数据集的一段例程，修改后程序完成via标注向coco数据集转换
通过VIA-2.0.8工具多边形制作VIAmask标注，选择Annotation/export annotation as json
将生成的json文件与图片放入train文件夹，将程序与train文件夹放在同一个目录下，将ann_file用你的json文件替代
运行，生成的新json文件,完成coco数据标注转换，就可以用图片训练数据运行mmdetection maskrcnn训练了
'''
import os.path as osp
import mmcv


def convert_VIA_to_coco(ann_file, out_file, image_prefix):
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []

#多个类别标注，循环获取每一个类别
        for n in range(len(v['regions'])):
            all_obj = v['regions'][n]
            obj = all_obj['shape_attributes']
            type_obj = all_obj['region_attributes']
            type_n = type_obj['type']

            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=int(type_n)-1,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1


    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'cap'} , {'id': 1, 'name': 'tissue'}, {'id': 2, 'name': 'peanuts'}, {'id': 3, 'name': 'battery'}])#添加自己训练数据的类别
    mmcv.dump(coco_format_json, out_file)

convert_VIA_to_coco("train/via_export_json.json", "train/annotation_coco.json", "train/")
