from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops

class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

class SegmentationPredictor(DetectionPredictor):
    """
    扩展自 DetectionPredictor 类，用于基于分割模型的预测任务。

    示例用法:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化 SegmentationPredictor 类，传入配置项、覆盖参数及回调函数。

        参数:
            cfg (dict): 配置参数，默认为 DEFAULT_CFG。
            overrides (dict): 用于覆盖默认配置的参数。
            _callbacks (callable): 回调函数，可选。
        """
        # 调用父类构造函数，完成基础初始化
        super().__init__(cfg, overrides, _callbacks)
        
        # 设置任务类型为 "segment"
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """
        对预测结果进行后处理，包括非极大值抑制和掩码处理。

        参数:
            preds (tuple): 模型预测的原始输出。
            img (torch.Tensor): 处理后的输入图像张量。
            orig_imgs (list or torch.Tensor): 原始输入图像或张量。

        返回:
            list: 处理后的检测结果，每个元素对应一张输入图像。
        """
        # 应用非极大值抑制处理预测框
        processed_preds = ops.non_max_suppression(
            preds[0],                        # 预测框
            self.args.conf,                  # 置信度阈值
            self.args.iou,                   # IOU 阈值
            agnostic=self.args.agnostic_nms, # 是否使用类别无关的 NMS
            max_det=self.args.max_det,       # 最大检测数量
            nc=len(self.model.names),        # 类别数量
            classes=self.args.classes        # 过滤的类别
        )

        # 如果输入图像是张量形式，则转换为 NumPy 格式
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        # 提取分割原型数据
        proto_data = preds[1][-1] if len(preds[1]) == 3 else preds[1]

        # 遍历每张图像的预测结果
        for i, pred in enumerate(processed_preds):
            original_image = orig_imgs[i]
            image_path = self.batch[0][i]

            # 如果没有预测框，则设置空掩码
            if not len(pred):
                masks = None
            else:
                # 根据是否使用 Retina 掩码选择不同的处理方式
                if self.args.retina_masks:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], original_image.shape)
                    masks = ops.process_mask_native(proto_data[i], pred[:, 6:], pred[:, :4], original_image.shape[:2])
                else:
                    masks = ops.process_mask(proto_data[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], original_image.shape)

            # 生成每张图像的结果对象并添加到结果列表中
            result = Results(original_image, path=image_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
            results.append(result)

        return results
