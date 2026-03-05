import os
import torch
import cv2
import numpy as np
import nibabel as nib
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models import Axial3D, get_backbone
from src.utils import get_device


def main():
    device = get_device(cuda_idx=[0])

    # Path to our trained model
    model_path = r"runs\ADNI\AD vs CN\3D\axial\Axial3D\VGG16\ImageNet\rand_seed_42_val_split_0.2_epochs_5_lr_0.01_batch_size_2_dropout_0.3_wd_0.0001_freeze_0.5_slices_80_optim_adamw_scheduler_none_pretrained_false\2026-03-05-16-15\fold_1\torch_model\best_model.pth"

    # Path to test image
    image_path = r"data\Extracted\REG-ADNI-AD1.nii"

    # 1. Load model
    print("Loading model...")
    backbone, embedding_dim = get_backbone(
        model_name="VGG16", device=device, pretrained_on="ImageNet"
    )
    model = Axial3D(
        backbone=backbone,
        num_classes=2,
        embedding_dim=embedding_dim,
        num_slices=5,  # We'll dynamically reshape anyway
        return_attention_weights=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load and prep image
    from src.data.dataset import ADNIDataset

    print(f"Loading image {image_path}...")
    img = nib.load(image_path)
    img_data = img.get_fdata()
    img_data = np.squeeze(img_data)

    # Mimic dataloader pre-processing (axial)
    # The ADNIDataset Centers the slices and gets num_slices
    num_slices = img_data.shape[2]
    slices = []
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Also keep an untransformed copy for visualization
    vis_slices = []

    for i in range(num_slices):
        slice_data = img_data[:, :, i]
        # Min-max scale
        if slice_data.max() > 0:
            slice_data = slice_data - slice_data.min()
            slice_data = slice_data / slice_data.max()

        # RGB convert
        import cv2

        slice_rgb = cv2.cvtColor(np.float32(slice_data), cv2.COLOR_GRAY2RGB)

        # Save for vis
        vis_slices.append(cv2.resize(slice_rgb, (224, 224)))

        # Transform for model
        slices.append(transform(slice_rgb))

    input_tensor = (
        torch.stack(slices).unsqueeze(0).to(device)
    )  # Shape: (1, num_slices, 3, 224, 224)
    print(f"Input tensor shape: {input_tensor.shape}")

    # 3. Apply GradCAM++
    print("Generating CAM...")
    target_layers = [model.backbone[-1]]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam_tool:
        # Note: input_tensor is 5D.
        # Monkey-patch get_target_width_height so pytorch-grad-cam resizes to 2D
        cam_tool.get_target_width_height = lambda t: (t.shape[-1], t.shape[-2])
        cams = cam_tool(input_tensor=input_tensor, targets=None)

    print(f"Generated CAMs shape: {cams.shape}")

    # 4. Visualize
    print("Saving visualizations...")
    os.makedirs("explainability_test", exist_ok=True)

    # Note: cams will likely be of shape (batch * num_slices, H, W) because of the wrapper
    # Actually, pytorch-grad-cam might output (batch, H, W) or (batch*slices, H, W).
    if len(cams.shape) == 3 and cams.shape[0] == num_slices:
        for i in range(num_slices):
            grayscale_cam = cams[i, :]
            vis_slice = vis_slices[i]

            # Combine the CAM with the original image
            cam_image = show_cam_on_image(
                vis_slice, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
            )

            # Save
            plt.imsave(f"explainability_test/slice_{i}.png", cam_image)
            print(f"Saved explainability_test/slice_{i}.png")
    else:
        print(
            f"Unexpected CAM returned shape: {cams.shape}. Handling logic needs adjustment."
        )


if __name__ == "__main__":
    main()
