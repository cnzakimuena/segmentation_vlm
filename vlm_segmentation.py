""" 
This script implements vision-language model (VLM) segmentation. It combines open-vocabulary 
object detection (OVD) and promptable segmentation (PS) to segment objects in an image based on a 
text prompt. Cat individual images are used for demonstration.
"""
import os
import random
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import pipeline
from transformers import logging as tf_logging
from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# mute transformers warnings
tf_logging.set_verbosity_error()

class VLMSegmentation:
    """ 
    This class implements vision-language model (VLM) segmentation. It combines open-vocabulary 
    object detection (OVD) and promptable segmentation (PS) to segment objects in an image based 
    on a text prompt.
    """
    def __init__(self, image_path, text_prompt):
        self.raw_image = Image.open(image_path)
        # plt.imshow(np.asarray(self.raw_image)); plt.axis('off'); plt.show()
        self.text_prompt = text_prompt
        # result variables
        self.ovd_data = {
            "boxes": [],
            "labels": None
            }
        self.ps_data = {
            "ps masks": None, 
            "semantic img": None, 
            "instance img": None
            }

    @staticmethod
    def ensure_model_downloaded(local_dir, repo_id):
        """
        Checks if the model file exists locally. If not, downloads the full repository snapshot 
        from Hugging Face into the designated local directory.
        """
        weights_file = os.path.join(local_dir, "model.safetensors")
        if not os.path.exists(weights_file):
            print(f"Model file missing in '{local_dir}'. Downloading '{repo_id}'")
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
            print("Download completed successfully.")
        else:
            print(f"Model file found locally at '{weights_file}'. Skipping download.")

    @staticmethod
    def preprocess_outputs(output):
        """ 
        Preprocess the outputs from the open-vocabulary object detection model to extract scores, 
        labels and bounding boxes.
        """
        input_scores = [x["score"] for x in output]
        input_labels = [x["label"] for x in output]
        input_boxes = []
        # for i in range(len(output)):
        for i, _ in enumerate(output):
            input_boxes.append([*output[i]["box"].values()])
        input_boxes = [input_boxes]
        return input_scores, input_labels, input_boxes

    @staticmethod
    def show_masks_on_image(raw_image, masks):
        """ 
        Shows the segmentation masks on the input image. It also provides a combined binary 
        semantic mask and an instance mask with random colors for each instance.
        """
        width, height = raw_image.size
        # create a mask image (assuming binary mask)
        image_with_mask = raw_image.convert("RGBA")
        # canvases for combined binary semantic mask and instance mask
        semantic_array = np.zeros((height, width), dtype=np.uint8)
        instance_array = np.zeros((height, width, 3), dtype=np.uint8)
        count = 0
        for mask in masks:
            mask = mask.cpu().numpy()
            mask_bool = mask.astype(bool)
            # update semantic mask; set mask regions to white (255)
            semantic_array[mask_bool] = 255
            # update instance mask; generate random color
            color_rgb = [random.randint(50, 255) for _ in range(3)]
            if count == 0:
                color_rgb = [255, 30, 90]
            elif count == 1:
                color_rgb = [30, 100, 255]
            instance_array[mask_bool] = color_rgb
            # overlay image
            mask_overlay = np.zeros((height, width, 4), dtype=np.uint8)
            color_rgba = color_rgb + [150]
            mask_overlay[mask_bool] = color_rgba
            mask_image = Image.fromarray(mask_overlay)
            # overlay the mask on the image
            image_with_mask = Image.alpha_composite(image_with_mask, mask_image)
            count += 1
        # convert numpy arrays to PIL images
        semantic_mask = Image.fromarray(semantic_array, mode="L")
        instance_mask = Image.fromarray(instance_array, mode="RGB")
        return image_with_mask, semantic_mask, instance_mask

    @staticmethod
    def show_box(box, ax):
        """ 
        Shows a bounding box on the input image.
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                edgecolor="dodgerblue",
                                facecolor=(0, 0, 0, 0), lw=2))

    def show_boxes_and_labels_on_image(self, raw_image, boxes, labels, scores):
        """ 
        Shows bounding boxes and labels on the input image. Optionally saves the visualization.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        for i, box in enumerate(boxes):
            self.show_box(box, plt.gca())
            plt.text(x=box[0], y=box[1] - 12,
                s=f"{labels[i]}: {scores[i]:,.4f}", c="white",
                path_effects=[pe.withStroke(linewidth=4, foreground="dodgerblue")])
        # plt.axis("on")
        plt.axis('off')
        # plt.show()

    def ovd(self):
        """
        Open-vocabulary object detection function. Objects in the image are detected and bounding 
        boxes are generated based on the text prompt. The results can be visualized and saved as 
        an image.
        """
        # load open-vocabulary object detection model
        owl_checkpoint = r'.\models\owlvit-base-patch32'
        repo_id = "google/owlvit-base-patch32"
        # check and download model files if missing locally
        self.ensure_model_downloaded(owl_checkpoint, repo_id)
        # load open-vocabulary object detection model
        detector = pipeline(model=owl_checkpoint, task="zero-shot-object-detection")
        # detect objects in input image based on text prompt
        output = detector(self.raw_image, candidate_labels=[self.text_prompt])
        # # show bounding boxes detected
        # print(output)
        # show input image with bounding boxes
        input_scores, input_labels, self.ovd_data["boxes"] = self.preprocess_outputs(output)
        if not self.ovd_data["boxes"][0]:
            raise ValueError("Open-vocabulary object detection unsuccessful.")
        self.show_boxes_and_labels_on_image(self.raw_image,
                                            self.ovd_data["boxes"][0], input_labels, input_scores)
        # create a list of positive labels of same length as number of predictions generated
        self.ovd_data["labels"] = np.repeat(1, len(output))
        # # show labels
        # print(labels)

    def ps(self, save_visualization=False, save_results=False):
        """ 
        Promptable segmentation function. Detected objects are segmented based on the bounding 
        boxes. The results can be visualized and saved as images.
        """
        # load promptable segmentation model
        sam_version = r'.\models\mobile_sam.pt'
        model = SAM(sam_version)
        # segment objects in input image based on bounding boxes prompt
        result = model.predict(self.raw_image,
                               bboxes=self.ovd_data["boxes"][0],
                               labels=self.ovd_data["labels"])
        self.ps_data["ps masks"] = result[0].masks.data
        # # show results
        # print(result); print(masks)
        # visualize masks
        mask_visualization, self.ps_data["semantic img"], self.ps_data["instance_img"] = \
            self.show_masks_on_image(self.raw_image, self.ps_data["ps masks"])
        if save_visualization:
            # export mask visualization
            np_image = np.asarray(mask_visualization)
            plt.imshow(np_image)
            plt.axis('off')
            plt.savefig(r'.\figure.pdf', format="pdf", bbox_inches='tight')
            plt.savefig(r'.\figure.png', format="png", bbox_inches='tight')
        # save results
        if save_results:
            self.ps_data["semantic img"].save(r'.\mask_semantic.png')
            self.ps_data["instance_img"].save(r'.\mask_instance.png')

if __name__ == '__main__':

    # --- specify image path & text prompt ---
    EXAMPLE_IMAGE_PATH = r'.\images\0144_006.jpg'
    EXAMPLE_TEXT_PROMPT = "cat"
    example_segmentation = VLMSegmentation(EXAMPLE_IMAGE_PATH, EXAMPLE_TEXT_PROMPT)

    # --- Vision-Language Model segmentation
    # get bounding boxes with OWL-ViT open-vocabulary object detection model
    example_segmentation.ovd()
    # get segmentation masks with Mobile SAM promptable segmentation model
    example_segmentation.ps(save_visualization=True, save_results=True)
