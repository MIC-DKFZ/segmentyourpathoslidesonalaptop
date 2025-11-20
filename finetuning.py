import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
from torchvision.transforms import functional as F
import openslide
import random

try:
    import openslide

    print("OpenSlide library detected. Using WSI data structure.")


    # Function to simulate finding the openslide level based on magnification
    def get_openslide_level(wsi_slide, target_mag):
        # Implementation depends on your WSI library
        return 0  # Placeholder: use level 2 for 20x typically
except ImportError:
    print("WARNING: openslide not found. Using Mock WSI Dataset.")





# Jaccard/IoU Loss and Binary Cross Entropy (BCE) Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE Loss (often used for pixel-wise classification in segmentation)
        bce = self.bce_loss(logits, targets)

        # Dice Loss (measures overlap, good for handling class imbalance)
        probs = torch.sigmoid(logits)
        num = 2. * (probs * targets).sum(dim=(2, 3))
        den = (probs + targets).sum(dim=(2, 3))
        dice_loss = 1 - (num + 1e-6) / (den + 1e-6)  # Add epsilon for stability

        return bce + dice_loss.mean()


class WSICystDataset(Dataset):
    """
    A PyTorch Dataset for sampling tiles from WSI slides and corresponding masks.
    We assume:
    1. SVS files are in WSI_DATA_DIR.
    2. Corresponding mask files (e.g., JSON annotations converted to binary masks or TIF) are also available.
    """

    def __init__(self, data_dir, tile_size, args):
        self.tile_size = tile_size
        self.slide_list = [f"slide_{i}.svs" for i in range(args.ANNOTATED_SLIDES)]  # 5 annotated slides
        self.all_tile_coords = []
        self.slide_handles = {}
        print(self.slide_list)

        # 1. Initialize slide handles (In a real scenario, this would load openslide objects)
        for slide_name in self.slide_list:
            slide_path = os.path.join(data_dir, slide_name)
            mask_path = slide_path.replace(".svs", "_mask.npy")  # Assuming pre-generated mask files
            # Mock initialization if openslide is not available
            slide = openslide.OpenSlide(slide_path)

            self.slide_handles[slide_name] = {
                'slide': slide,
                'mask_path': mask_path,
                'level': get_openslide_level(slide, args.MAGNIFICATION_LEVEL) if 'openslide' in globals() else 2
            }

            # 2. Pre-calculate tile coordinates for all slides
            # This is a critical step in WSI processing: efficiently deciding which tiles to sample.
            # Here we mock 10 tiles per slide.
            slide_width, slide_height = slide.level_dimensions[self.slide_handles[slide_name]['level']]
            ### >>> IDEAL TILE SELECTION (bias toward annotated mask) <<<
            N_TILES_PER_SLIDE = 100
            CANDIDATE_STRIDE = max(1, self.tile_size // 4)  # coarse stride for speed; tune as desired
            MIN_DIST = self.tile_size // 2  # suppress near-duplicates

            def _integral_image(img_uint8):
                # img is HxW uint8 {0,1}; integral is (H+1)x(W+1) for easy area sums
                img32 = img_uint8.astype(np.uint32)
                ii = img32.cumsum(axis=0).cumsum(axis=1)
                # pad with zero row/col at top-left to allow simple rectangle sum
                ii_pad = np.zeros((ii.shape[0] + 1, ii.shape[1] + 1), dtype=np.uint32)
                ii_pad[1:, 1:] = ii
                return ii_pad

            def _rect_sum(ii_pad, y, x, h, w):
                # sum over [y:y+h, x:x+w] using padded integral image
                y1, x1 = y, x
                y2, x2 = y + h, x + w
                return (ii_pad[y2, x2] - ii_pad[y1, x2] - ii_pad[y2, x1] + ii_pad[y1, x1])

            # Load or create mask aligned to target level grid

            full_mask = np.load(mask_path)


            # Ensure mask shape matches (H, W) == (slide_height, slide_width)
            # If not, try a safe fallback (resize not attempted; we keep original behavior).
            if full_mask.shape != (slide_height, slide_width):
                # Pad/crop to match (simple conservative approach)
                H, W = full_mask.shape
                Ht, Wt = slide_height, slide_width
                mask_tmp = np.zeros((Ht, Wt), dtype=np.uint8)
                h_copy = min(H, Ht)
                w_copy = min(W, Wt)
                mask_tmp[:h_copy, :w_copy] = (full_mask[:h_copy, :w_copy] > 0).astype(np.uint8)
                full_mask = mask_tmp

            # If there are positives, propose candidates that maximize mask coverage
            positives = int(full_mask.sum())
            ideal_coords = []
            if positives > 0:
                ii = _integral_image(full_mask)
                # Generate candidate top-lefts on a grid, staying within bounds
                xs = np.arange(0, max(1, slide_width - self.tile_size + 1), CANDIDATE_STRIDE, dtype=int)
                ys = np.arange(0, max(1, slide_height - self.tile_size + 1), CANDIDATE_STRIDE, dtype=int)

                # Compute mask coverage for each candidate
                scores = []
                for y in ys:
                    for x in xs:
                        s = _rect_sum(ii, y, x, self.tile_size, self.tile_size)
                        if s > 0:
                            scores.append((int(s), x, y))
                # Sort by descending coverage
                scores.sort(key=lambda t: t[0], reverse=True)

                # Greedy selection with distance-based suppression
                def _far_enough(x, y, chosen, mindist=MIN_DIST):
                    for cx, cy in chosen:
                        if (abs(cx - x) < mindist) and (abs(cy - y) < mindist):
                            return False
                    return True

                chosen = []
                for s, x, y in scores:
                    if len(chosen) >= N_TILES_PER_SLIDE:
                        break
                    if _far_enough(x, y, chosen):
                        chosen.append((x, y))
                ideal_coords = chosen

            # If not enough or no positives, backfill with random positions (original behavior)
            remaining = N_TILES_PER_SLIDE - len(ideal_coords)
            if remaining > 0:
                max_x = max(1, slide_width - self.tile_size)
                max_y = max(1, slide_height - self.tile_size)
                for _ in range(remaining):
                    x = np.random.randint(0, max_x)
                    y = np.random.randint(0, max_y)
                    ideal_coords.append((x, y))

            # Append to global coordinate list
            for (x, y) in ideal_coords:
                self.all_tile_coords.append((slide_name, x, y))
            ### >>> END IDEAL TILE SELECTION <<<

    def __len__(self):
        return len(self.all_tile_coords)

    def __getitem__(self, idx):
        slide_name, x_coord, y_coord = self.all_tile_coords[idx]
        slide_info = self.slide_handles[slide_name]
        slide = slide_info['slide']
        level = slide_info['level']
        mask_path = slide_info['mask_path']

        # --- A. Extract WSI Tile Image ---
        # Read the region from the WSI slide
        region = slide.read_region((x_coord, y_coord), level, (self.tile_size, self.tile_size))

        # Note: openslide read_region returns PIL image, MockSlide returns numpy array.
        if 'openslide' in globals():
            wsi_tile_img_np = np.array(region.convert('RGB'))[:, :, :3]  # Handle PIL Image from openslide
        else:
            wsi_tile_img_np = region[:, :, :3]  # Handle NumPy array from MockSlide

        # Convert to Pytorch format (C, H, W)
        image = F.to_tensor(wsi_tile_img_np).float()

        # --- B. Extract Corresponding Mask and Generate Prompt ---
        # In a real scenario, you'd load the full mask (e.g., from an annotation file)
        # and extract the corresponding tile region from it.
        full_mask = np.load(mask_path)

        # Extract tile mask from the full mask
        mask_tile_np = full_mask[y_coord:y_coord + self.tile_size, x_coord:x_coord + self.tile_size]

        # Convert mask to torch tensor
        gt_mask = torch.from_numpy(mask_tile_np).unsqueeze(0).float()  # (1, H, W)

        # --- C. Generate Prompt (Bounding Box) ---
        # MedSAM/SAM requires a prompt. We generate a prompt from the ground truth mask.
        # Find the bounding box of the cyst (assuming `1` represents the cyst)
        mask_indices = np.argwhere(mask_tile_np == 1)

        if mask_indices.size > 0:
            y_min, x_min = mask_indices.min(axis=0)
            y_max, x_max = mask_indices.max(axis=0)
            # The prompt is a bounding box [x_min, y_min, x_max, y_max]
            # MedSAM expects coordinates scaled to the *input* size (1024x1024 in this case)
            box_prompt = np.array([x_min, y_min, x_max + 1, y_max + 1])
        else:
            # If no cyst in this tile, create a dummy box (e.g., small box in the center)
            center = self.tile_size // 2
            box_prompt = np.array([center - 5, center - 5, center + 5, center + 5])

        # Convert box_prompt to tensor
        box_prompt_tensor = torch.as_tensor(box_prompt, dtype=torch.float).unsqueeze(0)  # (1, 4)

        return {
            'image': image,  # (3, 1024, 1024)
            'gt_mask': gt_mask,  # (1, 1024, 1024)
            'box_prompt': box_prompt_tensor,  # (1, 4)
            'original_size': (self.tile_size, self.tile_size)
        }


# --- 2. Model Initialization ---
def initialize_medsam(checkpoint_path, model_type):
    """Loads the SAM model and initializes it with MedSAM weights."""
    print(f"Loading SAM model with type: {model_type}...")
    try:
        # 1. Load the state dictionary explicitly, mapping to the determined device (CPU or CUDA)
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)

        # 2. Initialize the model architecture
        sam = sam_model_registry[model_type]()

        # 3. Load the state dictionary into the model
        sam.load_state_dict(state_dict)
    except KeyError:
        print(f"Error: Model type {model_type} not found in SAM registry.")
        return None

    # MedSAM finetuning typically only requires the prompt encoder and mask decoder to be updated
    # while keeping the heavy image encoder frozen.
    for name, param in sam.named_parameters():
        # Freeze the Image Encoder by default
        if 'image_encoder' in name:
            param.requires_grad = False
        # Unfreeze the Prompt Encoder and Mask Decoder for finetuning
        else:
            param.requires_grad = True

    sam.to(DEVICE)
    print("MedSAM model loaded and parameters set for finetuning.")
    return sam


# --- 3. Finetuning Loop ---
def finetune_medsam(model: Sam, dataloader: DataLoader, epochs: int, project_name):
    """Runs the finetuning process."""

    # Set up optimizer and loss function
    # Only optimizing parameters that require gradients (i.e., not the image encoder)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = CombinedLoss()

    model.train()  # Set model to training mode
    print(f"\nStarting finetuning on device: {DEVICE}")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, data in enumerate(pbar):
            images = data['image'].to(DEVICE)
            gt_masks = data['gt_mask'].to(DEVICE)
            box_prompts = data['box_prompt'].to(DEVICE)
            original_size = data['original_size']
            # Reset gradients
            optimizer.zero_grad()

            # 1. Image Embedding
            image_embeddings = model.image_encoder(images)

            # 2. Prompt Embedding (Box)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_prompts,
                masks=None
            )

            # 3. Mask Decoding
            # SAM outputs three masks and corresponding IoU predictions
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,  # We only want the best single mask output
            )

            # Upscale the low-res mask logits to the original input size (1024x1024)
            upscaled_masks = model.postprocess_masks(
                low_res_masks,
                original_size,
                original_size  # SAM is trained on 1024x1024 input
            )
            # The upscaled masks are logits, which we can directly compare with gt_masks (0/1)

            # Select the best mask output (if multimask_output=True, otherwise just use the single output)
            mask_logits = upscaled_masks[:, 0, :, :].unsqueeze(1)  # (B, 1, H, W)

            # Calculate Loss
            loss = criterion(mask_logits, gt_masks)

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

    print("Finetuning complete.")
    # --- 4. Save the Finetuned Weights ---
    torch.save(model.state_dict(), project_name+".pth")



# --- 5. Main Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Finetuning of a segmentation model to your specific structures."
    )
    # Required arguments
    parser.add_argument('--task', type=str, help='Your Task description, that is used to name the resulting checkpoint')
    parser.add_argument('--MEDSAM_CHECKPOINT_PATH', type=str, required=True, help='Path to the checkpoint that you have downloaded.', default="/Users/Downloads/medsam_vit_b.pth")
    parser.add_argument('--WSI_DATA_DIR', type=str, required=True, help='Path to your slide_1.svs and slide_1_mask.npy')
    parser.add_argument('--TILE_SIZE', type=int, default=1024, help='Patchsize')
    parser.add_argument('--MAGNIFICATION_LEVEL', type=int, default=40)
    parser.add_argument('--EPOCHS', type=int, default=10)
    parser.add_argument('--BATCH_SIZE', type=int, default=1)
    parser.add_argument('--ANNOTATED_SLIDES', type=int, default=1,help='The amount of slides you have annotated')
    parser.add_argument('--LEARNING_RATE', type=int, default=1e-5)
    parser.add_argument('--DEVICE', type=str, help='mps if you have a mac, cuda if you have a GPU, or cpu if you have none of the previous')

    args = parser.parse_args()
    cyst_dataset = WSICystDataset(data_dir=args.WSI_DATA_DIR, tile_size=args.TILE_SIZE, args)
    cyst_dataloader = DataLoader(cyst_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset ready: {len(cyst_dataset)} tiles prepared for finetuning.")

    # 2. Initialize MedSAM Model
    medsam_model = initialize_medsam(args.MEDSAM_CHECKPOINT_PATH, 'vit_b')

    if medsam_model:
        # 3. Start Finetuning
        finetune_medsam(medsam_model, cyst_dataloader, args.EPOCHS, args.task)

    print("\n--- Next Steps for Annotation ---")
    print("After finetuning, load the 'finetuned_medsam_cyst_segmentation.pth' weights.")
    print(
        "Use the SamPredictor class with your finetuned model to automatically generate segmentations (masks) for your 300 target slides.")
    print(
        "You will still need to tile the 300 slides, run the finetuned model on each tile, and stitch the results back together into full WSI masks.")
