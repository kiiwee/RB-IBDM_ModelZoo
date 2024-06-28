from simple_lama_inpainting import SimpleLama
from io import BytesIO
import torch
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import build_sam, SamPredictor
import supervision as sv
from huggingface_hub import hf_hub_download
from groundingdino.util.inference import annotate, load_image, predict
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
from groundingdino.util import box_ops
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert

from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance
import PIL
from IPython.display import display
import copy
import argparse
import os
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(
        repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(
        checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


groundingdino_model = load_model_hf(
    ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
sam_checkpoint = 'sam_vit_h_4b8939.pth'

sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
simple_lama = SimpleLama()


def single_process(image_i):


  # Read image
  transform = T.Compose(
      [
          T.RandomResize([800], max_size=1333),
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ]
  )
  image_s = image_i.convert("RGB")
  image_source = np.asarray(image_i)
  image, _ = transform(image_s, None)
  # detect insects using GroundingDINO
  def detect(image, model, text_prompt='insect . flower . cloud', box_threshold=0.25, text_threshold=0.25):
    boxes, logits, phrases = predict(
        image=image,
        model=model,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,device=device
    )

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    return annotated_frame, boxes, phrases

  annotated_frame, detected_boxes, phrases = detect(
      image, model=groundingdino_model)

  indices = [i for i, s in enumerate(phrases) if 'insect' in s]

  def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(
        boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )
    return masks.cpu()

  def draw_mask(mask, image, random_color=True):
      if random_color:
          color = np.concatenate(
              [np.random.random(3), np.array([0.8])], axis=0)
      else:
          color = np.array([30/255, 144/255, 255/255, 0.6])
      h, w = mask.shape[-2:]
      mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

      annotated_frame_pil = Image.fromarray(image).convert("RGBA")
      mask_image_pil = Image.fromarray(
          (mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

      return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

  segmented_frame_masks = segment(
      image_source, sam_predictor, boxes=detected_boxes[indices])

  # combine all masks into one for easy visualization
  final_mask = None
  for i in range(len(segmented_frame_masks) - 1):
    if final_mask is None:
      final_mask = np.bitwise_or(
          segmented_frame_masks[i][0].cpu(), segmented_frame_masks[i+1][0].cpu())
    else:
      final_mask = np.bitwise_or(
          final_mask, segmented_frame_masks[i+1][0].cpu())

  annotated_frame_with_mask = draw_mask(final_mask, image_source)

  def dilate_mask(mask, dilate_factor=15):
      mask = mask.astype(np.uint8)
      mask = cv2.dilate(
          mask,
          np.ones((dilate_factor, dilate_factor), np.uint8),
          iterations=1
      )
      return mask

  # original image
  image_source_pil = Image.fromarray(image_source)

  # create mask image
  mask = final_mask.numpy()
  mask = mask.astype(np.uint8) * 255
  image_mask_pil = Image.fromarray(mask)

  # dilate mask
  mask = dilate_mask(mask)
  dilated_image_mask_pil = Image.fromarray(mask)

  result = simple_lama(image_source, dilated_image_mask_pil)

  img1 = Image.fromarray(image_source)
  img2 = result

  diff = ImageChops.difference(img2, img1)

  threshold = 7
  # Grayscale
  diff2 = diff.convert('L')
  # Threshold
  diff2 = diff2.point(lambda p: 255 if p > threshold else 0)
  # # To mono
  diff2 = diff2.convert('1')

  img3 = Image.new('RGB', img1.size, (255, 236, 10))
  diff3 = Image.composite(img1, img3, diff2)
  return diff3
  print('Processing completed!')


def batch_process(path):
  save_path = os.path.join(path, 'GSL_output')
  if os.path.exists(save_path) == False:
    os.mkdir(save_path)

  for file in os.listdir(path):
    if file.endswith('.jpg'):
      # Read image
      image_source, image = load_image(os.path.join(path, file))

      # detect insects using GroundingDINO
      def detect(image, model, text_prompt='insect . flower . cloud', box_threshold=0.25, text_threshold=0.25):
        boxes, logits, phrases = predict(
            image=image,
            model=model,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        annotated_frame = annotate(
            image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        return annotated_frame, boxes, phrases

      annotated_frame, detected_boxes, phrases = detect(
          image, model=groundingdino_model)

      indices = [i for i, s in enumerate(phrases) if 'insect' in s]

      def segment(image, sam_model, boxes):
        sam_model.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_model.transform.apply_boxes_torch(
            boxes_xyxy.to(device), image.shape[:2])
        masks, _, _ = sam_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )
        return masks.cpu()

      def draw_mask(mask, image, random_color=True):
          if random_color:
              color = np.concatenate(
                  [np.random.random(3), np.array([0.8])], axis=0)
          else:
              color = np.array([30/255, 144/255, 255/255, 0.6])
          h, w = mask.shape[-2:]
          mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

          annotated_frame_pil = Image.fromarray(image).convert("RGBA")
          mask_image_pil = Image.fromarray(
              (mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

          return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

      segmented_frame_masks = segment(
          image_source, sam_predictor, boxes=detected_boxes[indices])

      # combine all masks into one for easy visualization
      final_mask = None
      for i in range(len(segmented_frame_masks) - 1):
        if final_mask is None:
          final_mask = np.bitwise_or(
              segmented_frame_masks[i][0].cpu(), segmented_frame_masks[i+1][0].cpu())
        else:
          final_mask = np.bitwise_or(
              final_mask, segmented_frame_masks[i+1][0].cpu())

      annotated_frame_with_mask = draw_mask(final_mask, image_source)

      def dilate_mask(mask, dilate_factor=15):
          mask = mask.astype(np.uint8)
          mask = cv2.dilate(
              mask,
              np.ones((dilate_factor, dilate_factor), np.uint8),
              iterations=1
          )
          return mask

      # original image
      image_source_pil = Image.fromarray(image_source)

      # create mask image
      mask = final_mask.numpy()
      mask = mask.astype(np.uint8) * 255
      image_mask_pil = Image.fromarray(mask)

      # dilate mask
      mask = dilate_mask(mask)
      dilated_image_mask_pil = Image.fromarray(mask)

      result = simple_lama(image_source, dilated_image_mask_pil)

      img1 = Image.fromarray(image_source)
      img2 = result

      diff = ImageChops.difference(img2, img1)

      threshold = 7
      # Grayscale
      diff2 = diff.convert('L')
      # Threshold
      diff2 = diff2.point(lambda p: 255 if p > threshold else 0)
      # # To mono
      diff2 = diff2.convert('1')

      img3 = Image.new('RGB', img1.size, (255, 236, 10))
      diff3 = Image.composite(img1, img3, diff2)
      diff3.save(os.path.join(save_path, file))
      piexif.transplant(os.path.join(path, file),
                        os.path.join(save_path, file))
  print('Batch completed, find processed images in GSL_output!')


def process(path):
  if os.path.isdir(path):
    batch_process(path)
  else:
    single_process(path)
# SAM

# LaMa
# Load image
def process(image):
    annotated_frame, detected_boxes, phrases = detect(
        image, model=groundingdino_model)


    segmented_frame_masks = segment(
        image_source, sam_predictor, boxes=detected_boxes[indices])
