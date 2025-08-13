import os
from ultralytics import YOLO
import argparse
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", type=str)
  parser.add_argument("-g", "--gpus", type=str)
  args = parser.parse_args()

  def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            # check_version(A.__version__, "1.0.3", hard=True)  # version requirement
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            # Transforms
            T = [
                A.Rotate(limit = 10, p=0.5),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0), contrast_limit=(-0.5,0), p=0.5),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),

            ]
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

  
  Albumentations.__init__ = __init__

  print(args.model, "training start")
  # Load a model
  # yolov11n.pt yolov11s.pt yolov11m.pt yolov11l.pt yolov11x.pt	
  # yolov10n.pt yolov10s.pt yolov10m.pt yolov10l.pt yolov10x.pt	
  # yolov5nu, yolov5su, yolov5mu, yolov5lu, yolov5xu, yolov5n6u, yolov5s6u, yolov5m6u, yolov5l6u, yolov5x6u	
  # yolov6n.pt yolov6s.pt yolov6m.pt yolov6l.pt yolov6x.pt
  # yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt
  # yolov9t.pt yolov9s.pt yolov9m.pt yolov9c.pt yolov9e.pt	

  cfg_dir = "ultralytics/cfg/models/11"
  model = YOLO(os.path.join(cfg_dir, args.model+".yaml"))
  
  # ON/OFF
  pretrained_weights = "yolo11n.pt"
  model.load(pretrained_weights)  # 기존 가중치 적용

  ##### default model!
  # model = YOLO(args.model + ".pt")
  
  # Train the model
  train_results = model.train(
      data="urin_class.yaml",  # path to dataset YAML
      epochs=5000,  # number of training epochs
      imgsz=224,  # training image size
      device=args.gpus,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
      patience=0,
      lr0 = 0.01,
  )
  
  # Evaluate model performance on the validation set
  # metrics = model.val()
  
  # print("benchmark start")
  # benchmark = model.benchmark(data="urin.yaml")
  
  # Perform object detection on an image
  # results = model("path/to/image.jpg")
  # results[0].show()
  
  # Export the model to ONNX format
  # path = model.export(format="onnx")  # return path to exported model
