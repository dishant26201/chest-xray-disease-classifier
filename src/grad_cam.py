# src/grad_cam.py

import numpy as np
from PIL import Image
from matplotlib import cm

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

from .config import IMG_SIZE, LABELS, LAST_CONV_LAYER
from .preprocessing import preprocess_image


def generate_gradcam(input_image, model, target_label):

    # Check if the target label belongs to label set
    if target_label not in LABELS:
        raise ValueError(
            "Unknown label" # Display message if not
        )
    
    class_index = LABELS.index(target_label) # Get index of label

    img_batch = preprocess_image(input_image) # Preprocess image to a batch (1, H, W, 3)

    score = CategoricalScore([class_index]) # Indicate which class to target

    gradcam = Gradcam(model, clone=True) # Wrap model with gradcam logic. Clone true creates a clone so original model remains unchanged

    output_gradcam = gradcam(score, img_batch, penultimate_layer=LAST_CONV_LAYER) # Call the gradcam object

    heatmap = output_gradcam[0] # Extract the heatmap

    # Resize heatmap to the model input size
    heatmap_image = Image.fromarray((heatmap * 255).astype("uint8")) # Convert to uint8 so PIL can resize image
    heatmap_image = heatmap_image.resize(IMG_SIZE, Image.BILINEAR) # Resize to same size as model input
    heatmap_resized = np.array(heatmap_image).astype("float32") / 255.0 # Convert back to float32

    # Prepare base image
    base_image = input_image.convert("RGB").resize(IMG_SIZE) # Ensure heatmap image and base image are the same size
    base_resized = np.array(base_image).astype("float32") / 255.0 # Convert PIL image to float32

    # Apply colour map and blend
    colormap = cm.get_cmap("jet")           
    heatmap_color = colormap(heatmap_resized) 
    heatmap_rgb = heatmap_color[..., :3]

    transparency = 0.4 # Transparency ratio: 40% heatmap, 60% image
    overlay = base_resized * (1.0 - transparency) + heatmap_rgb * transparency # Blend the two images
    overlay = np.clip(overlay, 0.0, 1.0) # Clip to ensure no invalid colours

    overlay_uint8 = (overlay * 255).astype("uint8") # Convert the overlayed image to uint8
    overlay_image = Image.fromarray(overlay_uint8) # Turn numpy array to real image (RGB)

    return overlay_image
