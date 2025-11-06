import os
import imageio

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from skimage.segmentation import mark_boundaries
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr

from utils.stats import convert_pvalue_to_asterisks


def _get_img_dim_in_RAS(img_path):
    
    # Load img
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata()
    affine = img_nii.affine
    
    # Get transfrom to RAS
    current_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
    
    # Apply transform
    img = nib.orientations.apply_orientation(img, transform)
    
    # Dimension in RAS
    dims = np.shape(img)
    
    return dims


def _custom_bwr_cmap():
    
    # Points in colorbar
    n = 256
    half = n // 2

    # RGB Channels
    r = np.concatenate([np.linspace(0, 1, half), np.ones(half)])  # 0 → 1 → 1
    g = np.concatenate([np.linspace(0, 1, half), np.linspace(1, 0, half)])  # 0 → 1 → 0
    b = np.concatenate([np.ones(half), np.linspace(1, 0, half)])  # 1 → 1 → 0

    # Alpha Channel: opaque at ends, transparent at white
    alpha = np.concatenate([np.linspace(1, 0, half), np.linspace(0, 1, half)])  # 1 → 0 → 1

    # Stack into RGBA
    rgba = np.stack([r, g, b, alpha], axis=1)
    cm = ListedColormap(rgba)
    
    return cm


def _extract_slice_img(
        img_path,
        slice_index,
        slice_ornt,
        img_index=None,
        mask_path=None,
        outline_path=None,
        outline_c=None,
        cmap=None, 
        clim=None,
        xlim=None,
        ylim=None,
        keep_alpha=False,
        cmap_sym=False,
        **kwargs,
    ):
    
    # Load img
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata()
    affine = img_nii.affine
    
    # Get transfrom to RAS
    current_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
    
    # Apply transform
    img = nib.orientations.apply_orientation(img, transform)
    
    # Load mask
    if mask_path is not None:
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata().astype(bool)
        
        # Apply transform
        mask = nib.orientations.apply_orientation(mask, transform)
        
        # Apply mask on image
        if np.shape(img) == np.shape(mask):
            img *= mask
        else:
            img *= mask[..., np.newaxis]
      
    # Load outline
    if outline_path is not None:
        outline_nii = nib.load(outline_path)
        outline = outline_nii.get_fdata().astype(bool)
        
        # Apply transform
        outline = nib.orientations.apply_orientation(outline, transform)
        
        if outline_c is None:
            outline_c = (1,0,0) # default to red
        
    # Custom bwr colormap
    if cmap == "bwr":
        cm = _custom_bwr_cmap()
    elif cmap is not None:
        cm = plt.get_cmap(cmap)
        
    #
    if img_index is not None:
        img = img[:,:,:,img_index]

    # Treshold image
    if clim is None:
        clim = [np.quantile(img[img != 0], 0.05), np.quantile(img[img != 0], 0.95)]
    if not cmap_sym:
        img[img != 0] -= clim[0]
        img[img != 0] /= (clim[1] - clim[0])
    else:
        clim = np.max(np.abs(clim))
        clim = [-clim, clim]
        img -= clim[0]
        img /= (clim[1] - clim[0])
    img = np.clip(img, 0.0, 1.0)
    img *= 255
    img = img.astype(np.uint8)
    
    # Extract slice
    if slice_ornt == "axial":
        img = img[:,:,slice_index]
    elif slice_ornt == "coronal":
        img = img[:,slice_index,:]
    elif slice_ornt == "sagittal":
        img = img[slice_index,:,:]
    img = np.fliplr(np.rot90(img))
        
    # Image window
    if xlim is not None:
        img = img[xlim[0]:xlim[1],:]
    if ylim is not None:
        img = img[:,ylim[0]:ylim[1]]
         
    # Extract outline
    if outline_path is not None:
        if slice_ornt == "axial":
            outline = outline[:,:,slice_index]
        elif slice_ornt == "coronal":
            outline = outline[:,slice_index,:]
        elif slice_ornt == "sagittal":
            outline = outline[slice_index,:,:]
        outline = np.fliplr(np.rot90(outline))
            
        if xlim is not None:
            outline = outline[xlim[0]:xlim[1],:]
        if ylim is not None:
            outline = outline[:,ylim[0]:ylim[1]]
            
    # Colormap adn outlines
    if cmap is not None:
        # Apply colormap
        img = cm(img)
        
        # Remove alpha channel
        if not keep_alpha:
            img = img[..., 0:3]
        
        # Mark outlines
        if outline_path is not None:
            img = mark_boundaries(img, outline, color=outline_c)
          
        # to integer
        img *= 255
        img = img.astype(np.uint8)
    else:
        if outline_path is not None:
            # Mark outline
            img = mark_boundaries(img, outline, color=outline_c)
            
            # to integer
            img *= 255
            img = img.astype(np.uint8)
        
    return img


def _combine_imgs(
        rgb,
        rgba,
        global_alpha,
    ):
    
    # Convert images to float32 for safe blending
    rgb = rgb.astype(np.float32)
    rgba = rgba.astype(np.float32)
    
    # Normalize if in 0-255 range
    if rgb.max() > 1.0:
        rgb /= 255.0
    if rgba.max() > 1.0:
        rgba /= 255.0
    
    # Extract RGB and alpha from the RGBA image
    overlay_rgb = rgba[..., :3]
    overlay_alpha = rgba[..., 3:]  # shape: (H, W, 1)
    
    # Optional: apply global alpha (like alpha=0.5 in imshow)
    global_alpha = 0.5
    alpha = overlay_alpha * global_alpha  # shape: (H, W, 1)
    
    # Blend images: out = overlay * alpha + base * (1 - alpha)
    blended = overlay_rgb * alpha + rgb * (1 - alpha)
    
    # Convert back to uint8 for saving
    blended_uint8 = (blended * 255).astype(np.uint8)
    
    return blended_uint8


def tse_pipeline_plot(
        input_mag_paths,
        input_pha_paths,
        input_pha_corr_paths,
        input_sig_paths,
        input_mask_paths,
        input_sig_path,
        input_mask_path,
        input_mask_eroded_path,
        stack_view_info,
        output_plot_path=None,
        view_info=None,
        mag_clim=None, 
        pha_clim=None,
        sig_clim=None,
        sig_cmap="inferno",
        outline_c=None,
        figsize=None,
        title=None,
        save_imageio=False,
    ):
    
    # Default view
    if view_info is None:
        dims = _get_img_dim_in_RAS(input_sig_path)
        view_info = { 
            "axial": {"slice_index": dims[2]//2},
            "coronal": {"slice_index": dims[1]//2},
            "sagittal": {"slice_index": dims[0]//2},
        }
    
    # For plotting
    vmin, vmax = 0, 255
    cmap = "gray"
    
    # Initialize output path
    if output_plot_path:
        output_dir = os.path.dirname(output_plot_path)
        os.makedirs(output_dir, exist_ok=True)
   
    # Imageiodirectory
    save_imageio = output_plot_path and save_imageio
    if save_imageio:
        plot_name = os.path.splitext(os.path.basename(output_plot_path))[0]
        imageio_dir = os.path.join(output_dir, "imageio", plot_name)
        os.makedirs(imageio_dir, exist_ok=True)
        
    # Number of images in plot
    n = len(view_info) + 4
    m = len(stack_view_info)
    
    # Initialize plot
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(m, n, figure=fig)
    fig.suptitle(title)
    
    # Iter over stack view
    for i, (ornt, info) in enumerate(stack_view_info.items()):
    
        # Extract magnitude image
        mag_img = _extract_slice_img(
            img_path=input_mag_paths[info["stack_index"]], 
            slice_index=info["slice_index"],
            slice_ornt=ornt, 
            mask_path=input_mask_paths[info["stack_index"]],
            clim=mag_clim,
        )
        
        # Plot magnitude image
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(mag_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_title("Magnitude")
        ax.set_ylabel(f"{ornt.title()} stack")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save magnitude images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{ornt}_mag.png")
            imageio.imwrite(imageio_path, mag_img)
        
        # Extract phase image
        pha_img = _extract_slice_img(
            img_path=input_pha_paths[info["stack_index"]], 
            slice_index=info["slice_index"], 
            slice_ornt=ornt, 
            mask_path=input_mask_paths[info["stack_index"]], 
            clim=pha_clim,
        )
        
        # Plot phase image
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(pha_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_title("Phase")
        ax.axis('off')
        
        # Save phase image
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{ornt}_pha.png")
            imageio.imwrite(imageio_path, pha_img)
    
        # Extcract corrected phase image
        pha_corr_img = _extract_slice_img(
            img_path=input_pha_corr_paths[info["stack_index"]],
            slice_index=info["slice_index"], 
            slice_ornt=ornt, 
            mask_path=input_mask_paths[info["stack_index"]], 
            clim=pha_clim,
        )
        
        # Plot corrected phase image
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(pha_corr_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_title("Flow void artefact\ncorrected phase")
        ax.axis('off')
        
        # Save phase image
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{ornt}_pha_corr.png")
            imageio.imwrite(imageio_path, pha_corr_img)
                
        # Extract conductivity image
        sig_img = _extract_slice_img(
            img_path=input_sig_paths[info["stack_index"]],
            slice_index=info["slice_index"], 
            slice_ornt=ornt, 
            mask_path=input_mask_paths[info["stack_index"]],
            clim=sig_clim,
            cmap=sig_cmap,
        )
        
        # Plot conductivity images 
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(sig_img)
        if i == 0:
            ax.set_title(r"$\sigma$ maps")
        ax.axis('off')
        
        # Save conductivity images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{ornt}_sig.png")
            imageio.imwrite(imageio_path, sig_img)
            
    # Iter over view 
    for i, (ornt, info) in enumerate(view_info.items()):
        # Exctract average conductivity image
        sig_img = _extract_slice_img(
            img_path=input_sig_path,
            slice_index=info["slice_index"], 
            slice_ornt=ornt, 
            mask_path=input_mask_path,
            outline_path=input_mask_eroded_path,
            outline_c=outline_c,
            clim=sig_clim,
            cmap=sig_cmap,
        )
        
        # Plot average conductivity image
        ax = fig.add_subplot(gs[:, 4+i])
        ax.imshow(sig_img)
        ax.set_title(r"Average $\sigma$ map"f"\n{ornt} view")
        ax.axis('off')
        
        # Save conductivity images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{ornt}_sig_avg.png")
            imageio.imwrite(imageio_path, sig_img)
        
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    if output_plot_path:
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    
    
def epi_pipeline_plot(
        input_mag_path,
        input_pha_path,
        input_mag_topup_path,
        input_pha_topup_path,
        input_pha_corr_path,
        input_sig_path,
        input_mask_path,
        input_mask_eroded_path,
        output_plot_path=None,
        slice_index=None,
        ornt="sagittal",
        view_info=None,
        mag_clim=None,
        pha_clim=None,
        sig_clim=None,
        sig_cmap="inferno",
        outline_c=None,
        figsize=None,
        title=None,
        save_imageio=False,
    ):
    
    # Default view
    if slice_index is None:
        dims = _get_img_dim_in_RAS(input_mag_path)
        slice_index =  dims[0]//2
    if view_info is None:
        view_info = {
            "PA": {"img_index": 0},
            "AP": {"img_index": 1},
        }
        
    # For plotting
    cmap = "gray"
    vmin, vmax = 0, 255
    
    # Initializwe output dir
    if output_plot_path:
        output_dir = os.path.dirname(output_plot_path)
        os.makedirs(output_dir, exist_ok=True)
        
    # Initialize imageio dir
    save_imageio = output_plot_path and save_imageio
    if save_imageio:
        plot_name = os.path.splitext(os.path.basename(output_plot_path))[0]
        imageio_dir = os.path.join(output_dir, "imageio", plot_name)
        os.makedirs(imageio_dir, exist_ok=True)
    
    # Initialize plot
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 6, figure=fig)
    fig.suptitle(title)
    
    # Iter over views
    for i, (PE_dir, info) in enumerate(view_info.items()):
    
        # Extract magnitude image
        mag_img = _extract_slice_img(
            img_path=input_mag_path, 
            slice_index=slice_index,
            slice_ornt=ornt, 
            img_index=info["img_index"],
            clim=mag_clim,
        )
        
        # Plot magnitude image
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(mag_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_title("Magnitude")
        ax.set_ylabel(PE_dir)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save magnitude images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{PE_dir}_mag.png")
            imageio.imwrite(imageio_path, mag_img)
    
        # Extract phase image
        pha_img = _extract_slice_img(
            img_path=input_pha_path, 
            slice_index=slice_index,
            slice_ornt=ornt, 
            img_index=info["img_index"],
            clim=pha_clim,
        )
        
        # Plot phase images
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(pha_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_title("Phase")
        ax.axis('off')
        
        # Save phase image
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{PE_dir}_pha.png")
            imageio.imwrite(imageio_path, pha_img)
    
    # Extract topup magnitude image
    mag_topup_img = _extract_slice_img(
        img_path=input_mag_topup_path, 
        slice_index=slice_index,
        slice_ornt=ornt, 
        mask_path=input_mask_path,
        clim=mag_clim,
    )
    
    # Plot topup magnitude image
    ax = fig.add_subplot(gs[:, 2])
    ax.imshow(mag_topup_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("$B_0$ corrected\nmagnitude")
    ax.axis('off')
    
    # Save topup magnitude image
    if save_imageio:
        imageio_path = os.path.join(imageio_dir, "mag_topup.png")
        imageio.imwrite(imageio_path, mag_topup_img)
    
    # Extract topup phase image
    pha_topup_img = _extract_slice_img(
        img_path=input_pha_topup_path, 
        slice_index=slice_index,
        slice_ornt=ornt, 
        mask_path=input_mask_path,
        clim=pha_clim,
    )
    
    # Plot topup phase image
    ax = fig.add_subplot(gs[:, 3])
    ax.imshow(pha_topup_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("$B_0$ corrected\nphase")
    ax.axis('off')
    
    # Save topup phase image
    if save_imageio:
        imageio_path = os.path.join(imageio_dir, "pha_topup.png")
        imageio.imwrite(imageio_path, pha_topup_img)
    
    # Extract corrected phase image
    pha_corr_img = _extract_slice_img(
        img_path=input_pha_corr_path, 
        slice_index=slice_index,
        slice_ornt=ornt, 
        mask_path=input_mask_path,
        clim=pha_clim,
    )
    
    # Plot corrected phase image
    ax = fig.add_subplot(gs[:, 4])
    ax.imshow(pha_corr_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Flow void artefact\ncorrected phase")
    ax.axis('off')
    
    # Save corrected phase image
    if save_imageio:
        imageio_path = os.path.join(imageio_dir, "pha_corr.png")
        imageio.imwrite(imageio_path, pha_corr_img)
    
    # Extract conductivity image
    sig_img = _extract_slice_img(
        img_path=input_sig_path, 
        slice_index=slice_index,
        slice_ornt=ornt, 
        mask_path=input_mask_path,
        outline_path=input_mask_eroded_path,
        outline_c=outline_c,
        clim=sig_clim,
        cmap=sig_cmap,
    )
    
    # Plot conductivity image
    ax = fig.add_subplot(gs[:, 5])
    ax.imshow(sig_img)
    ax.set_title(r"$\sigma$ map")
    ax.axis('off')
    
    # Save conductivity image
    if save_imageio:
        imageio_path = os.path.join(imageio_dir, "sig.png")
        imageio.imwrite(imageio_path, sig_img)
    
    # Adjust Layout
    fig.tight_layout()
    
    # Save plot
    if output_plot_path:
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)


def gre_pipeline_plot(
        input_mag_path,
        input_pha_path,
        input_mag_degibbs_path,
        input_pha_degibbs_path,
        input_sig_path,
        input_eps_path,
        input_mask_path,
        input_mask_eroded_path,
        output_plot_path=None,
        slice_index=None,
        ornt="sagittal",
        view_info=None,
        xlim=None,
        ylim=None,
        mag_clim=None,
        pha_clim=None,
        sig_clim=None,
        eps_clim=None,
        sig_cmap="inferno",
        eps_cmap="inferno",
        outline_c=None,
        figsize=None,
        title=None,
        save_imageio=False,
    ):
    
    # Default view
    if view_info is None:
        view_info = {"head": {"img_index": 0}}
        
    # For plotting
    vmin, vmax = 0, 255
    cmap = "gray"
    
    # Initialize output path
    if output_plot_path:
        output_dir = os.path.dirname(output_plot_path)
        os.makedirs(output_dir, exist_ok=True)
    
    # Imageio path
    save_imageio = output_plot_path and save_imageio
    if save_imageio:
        plot_name = os.path.splitext(os.path.basename(output_plot_path))[0]
        imageio_dir = os.path.join(output_dir, "imageio", plot_name)
        os.makedirs(imageio_dir, exist_ok=True)
        
    # Number of views
    m = len(view_info)
    
    # Initialize plot
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(m, 6, figure=fig)
    fig.suptitle(title)
    
    # Iter over views
    for j, (coil, info) in enumerate(view_info.items()):
    
        # Extract magnitude image
        mag_img = _extract_slice_img(
            img_path=input_mag_path, 
            slice_index=slice_index,
            slice_ornt=ornt, 
            img_index=info["img_index"],
            mask_path=input_mask_path,
            xlim=xlim,
            ylim=ylim,
            clim=mag_clim,
        )

        # Plot magnitude images
        ax = fig.add_subplot(gs[j, 0])
        ax.imshow(mag_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if j == 0:
            ax.set_title("Magnitude")
        ax.set_ylabel(f"{coil.title()} coil")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save magnitude images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{coil}_mag.png")
            imageio.imwrite(imageio_path, mag_img)
        
        # Exctract phase images
        pha_img = _extract_slice_img(
            img_path=input_pha_path, 
            slice_index=slice_index,
            slice_ornt=ornt, 
            img_index=info["img_index"],
            mask_path=input_mask_path,
            xlim=xlim,
            ylim=ylim,
            clim=pha_clim,
        )
        
        # Plot phase images
        ax = fig.add_subplot(gs[j, 1])
        ax.imshow(pha_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if j == 0:
            ax.set_title("Phase")
        ax.axis('off')
        
        # Save phase images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{coil}_pha.png")
            imageio.imwrite(imageio_path, pha_img)
        
        # Exctract deggibs magnitude image
        mag_degibbs_img = _extract_slice_img(
            img_path=input_mag_degibbs_path, 
            slice_index=slice_index,
            slice_ornt=ornt, 
            img_index=info["img_index"],
            mask_path=input_mask_path,
            xlim=xlim,
            ylim=ylim,
            clim=mag_clim,
        )
        
        # Plot deggibs magnitude image
        ax = fig.add_subplot(gs[j, 2])
        ax.imshow(mag_degibbs_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if j == 0:
            ax.set_title("DeGibbs\nmagnitude")
        ax.axis('off')
        
        # Save degibbs magnitude images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{coil}_mag_degibbs.png")
            imageio.imwrite(imageio_path, mag_degibbs_img)
        
        # Extract degibbs phase images
        pha_degibbs_img = _extract_slice_img(
            img_path=input_pha_degibbs_path, 
            slice_index=slice_index,
            slice_ornt=ornt, 
            img_index=info["img_index"],
            mask_path=input_mask_path,
            xlim=xlim,
            ylim=ylim,
            clim=pha_clim,
        )
        
        # Plot degibbs phase images 
        ax = fig.add_subplot(gs[j, 3])
        ax.imshow(pha_degibbs_img, cmap=cmap, vmin=vmin, vmax=vmax)
        if j == 0:
            ax.set_title("DeGibbs\nphase")
        ax.axis('off')
    
        # Save degibbs phase images
        if save_imageio:
            imageio_path = os.path.join(imageio_dir, f"{coil}_pha_degibbs.png")
            imageio.imwrite(imageio_path, pha_degibbs_img)
    
    # Extract conductivity image
    sig_img = _extract_slice_img(
        img_path=input_sig_path, 
        slice_index=slice_index,
        slice_ornt=ornt, 
        mask_path=input_mask_path,
        outline_path=input_mask_eroded_path,
        outline_c=outline_c,
        xlim=xlim,
        ylim=ylim,
        clim=sig_clim,
        cmap=sig_cmap,
    )
    
    # Plot conductivity image
    ax = fig.add_subplot(gs[:, 4])
    ax.imshow(sig_img)
    ax.set_title(r"$\sigma$ map")
    ax.axis('off')
    
    # Save conductivity image
    if save_imageio:
        imageio_path = os.path.join(imageio_dir, "sig.png")
        imageio.imwrite(imageio_path, sig_img)
    
    # Extract permittivity image
    eps_img = _extract_slice_img(
        img_path=input_eps_path, 
        slice_index=slice_index,
        slice_ornt=ornt, 
        mask_path=input_mask_path,
        outline_path=input_mask_eroded_path,
        outline_c=outline_c,
        xlim=xlim,
        ylim=ylim,
        clim=eps_clim,
        cmap=eps_cmap,
    )
    
    # Plot permittivity image 
    ax = fig.add_subplot(gs[0:, 5])
    ax.imshow(eps_img)
    ax.set_title(r"$\epsilon_r$ map")
    ax.axis('off')
    
    # Save conductivity image
    if save_imageio:
        imageio_path = os.path.join(imageio_dir, "eps.png")
        imageio.imwrite(imageio_path, eps_img)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    if output_plot_path:
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    
    
def dhcp_atlas_plot(
        input_anat_atlas_paths,
        input_sig_atlas_paths,
        output_plot_path=None,
        view_info=None,
        ages=None,
        mag_clim=None,
        sig_clim=None,
        sig_cmap="inferno",
        figsize=None,
        title=None,
        save_imageio=False,
    ):
    
    # Default view
    if view_info is None:
        dims = _get_img_dim_in_RAS(input_anat_atlas_paths[0])
        view_info = {
            "axial": {"slice_index": dims[2]//2},
            "coronal": {"slice_index": dims[1]//2},
            "sagittal": {"slice_index": dims[0]//2},
        }
    
    # Default ages
    if ages is None:
        ages = {i: i for i in range(len(input_anat_atlas_paths))}
    
    # For plotting
    vmin, vmax = 0, 255
    cmap = "gray"
    
    # Initialize output path
    if output_plot_path:
        output_dir = os.path.dirname(output_plot_path)
        os.makedirs(output_dir, exist_ok=True)
    
    # Imageio path
    save_imageio = output_plot_path and save_imageio
    if save_imageio:
        plot_name = os.path.splitext(os.path.basename(output_plot_path))[0]
        imageio_dir = os.path.join(output_dir, "imageio", plot_name)
        os.makedirs(imageio_dir, exist_ok=True)
        
    # Number of images
    n = len(ages)
    m = len(view_info) * 2
    
    # Initialize figure
    fig, axes = plt.subplots(m, n, figsize=figsize)
    fig.suptitle(title)
    
    # Iter over ages
    for i, (age_id, age) in enumerate(ages.items()):
        
        # Iter over view
        for j, (ornt, info) in enumerate(view_info.items()):
        
            # Extract magnitude image
            mag_img = _extract_slice_img(
                img_path=input_anat_atlas_paths[age_id], 
                slice_index=info["slice_index"],
                slice_ornt=ornt, 
                clim=mag_clim,
            )
        
            # Plot magnitude image
            axes[j,i].imshow(mag_img, cmap=cmap, vmin=vmin, vmax=vmax)
            if i == 0:
                axes[j,i].set_ylabel(f"Anatomical T2w atlas\n{ornt} view")
            axes[j,i].set_xticks([])
            axes[j,i].set_yticks([])
                
            # Save magnitude image
            if save_imageio:
                imageio_path = os.path.join(imageio_dir, f"{ornt}_mag_{age}.png")
                imageio.imwrite(imageio_path, mag_img)
                
            # Extract conductivity image
            sig_img = _extract_slice_img(
                img_path=input_sig_atlas_paths[age_id], 
                slice_index=info["slice_index"],
                slice_ornt=ornt, 
                clim=sig_clim,
                cmap=sig_cmap, 
            )
            
            # Plot conductivity image
            axes[m//2+j,i].imshow(sig_img)
            if i == 0:
                axes[m//2+j,i].set_ylabel(f"Conductivity atlas\n{ornt} view")
            axes[m//2+j,i].set_xticks([])
            axes[m//2+j,i].set_yticks([])
                
            
            # Save conductivity image
            if save_imageio:
                imageio_path = os.path.join(imageio_dir, f"{ornt}_sig_{age}.png")
                imageio.imwrite(imageio_path, sig_img)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    if output_plot_path:
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    
    
def dhcp_glm_plot(
        input_anat_atlas_path,
        input_ev_map_paths,
        input_ev_labels,
        input_mask_path,
        output_plot_path=None,
        view_info=None,
        mag_clim=None,
        alpha=1.0,
        ev_cmap="bwr",
        outline_c=None,
        ev_clim=None,
        figsize=None,
        title=None,
        save_imageio=False,
    ):
    
    # Default view
    if view_info is None:
        dims = _get_img_dim_in_RAS(input_anat_atlas_path)
        view_info = {
            "axial": {"slice_index": dims[2]//2},
            "coronal": {"slice_index": dims[1]//2},
            "sagittal": {"slice_index": dims[0]//2},
        }
    
    # Default clim
    if ev_clim is None:
        ev_clim = {input_ev_label: None for input_ev_label in input_ev_labels}
    
    # Initialize output dir
    if output_plot_path:
        output_dir = os.path.dirname(output_plot_path)
        os.makedirs(output_dir, exist_ok=True)
    
    # Imageio dir
    save_imageio = output_plot_path and save_imageio
    if save_imageio:
        plot_name = os.path.splitext(os.path.basename(output_plot_path))[0]
        imageio_dir = os.path.join(output_dir, "imageio", plot_name)
        os.makedirs(imageio_dir, exist_ok=True)
    
    # Number of images
    n = len(input_ev_map_paths)
    m = len(view_info)
    
    # Initialize plot
    fig, axes = plt.subplots(m, n, figsize=figsize)
    fig.suptitle(title)
    
    # Iter over EV
    for i, (ev_map_path, ev_label) in enumerate(zip(input_ev_map_paths, input_ev_labels)):
        
        # Iter over view
        for j, (ornt, info) in enumerate(view_info.items()):
        
            # Extract magnitude image
            mag_img = _extract_slice_img(
                img_path=input_anat_atlas_path, 
                slice_index=info["slice_index"],
                slice_ornt=ornt, 
                outline_path=input_mask_path,
                outline_c=outline_c,
                clim=mag_clim,
            )
            
            # Extract EV images
            ev_img = _extract_slice_img(
                img_path=ev_map_path, 
                slice_index=info["slice_index"],
                slice_ornt=ornt, 
                clim=ev_clim[ev_label],
                cmap=ev_cmap,
                cmap_sym=True,
                keep_alpha=True,
            )
            
            # Plot magnitude and EV images
            axes[j,i].imshow(mag_img)
            axes[j,i].imshow(ev_img, alpha=alpha)
            if j == 0:
                axes[j,i].set_title(f"{ev_label} signifcance map")
            if i == 0:
                axes[j,i].set_ylabel(f"{ornt.title()} view")
            axes[j,i].set_xticks([])
            axes[j,i].set_yticks([])
            
        
            # Save images
            if save_imageio:
                img = _combine_imgs(mag_img, ev_img, alpha)
                imageio_path = os.path.join(imageio_dir, f"{ornt}_{ev_label}.png")
                imageio.imwrite(imageio_path, img)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    if output_plot_path:
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    
    
def dhcp_atlas_gif(
        input_anat_atlas_paths,
        input_sig_atlas_paths,
        output_dir,
        view_info=None,
        fps=1,
        alpha=0.5,
        mag_clim=None,
        sig_clim=None,
        sig_cmap="inferno",
    ):
    
    # Default view
    if view_info is None:
        dims = _get_img_dim_in_RAS(input_anat_atlas_paths[0])
        view_info = {
            "axial": {"slice_index": dims[2]//2},
            "coronal": {"slice_index": dims[1]//2},
            "sagittal": {"slice_index": dims[0]//2},
        }
        
    # Number of images
    n = len(input_anat_atlas_paths)
    
    # Initialize output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Iter over view
    for ornt, info in view_info.items():
        frames = []
        # Iter over images
        for i in range(n):
           # Extract magnitude image
            mag_img = _extract_slice_img(
                img_path=input_anat_atlas_paths[i], 
                slice_index=info["slice_index"],
                slice_ornt=ornt, 
                clim=mag_clim,
            )
            
            # Extract conductivity image
            sig_img = _extract_slice_img(
                img_path=input_sig_atlas_paths[i], 
                slice_index=info["slice_index"],
                slice_ornt=ornt, 
                clim=sig_clim,
                cmap=sig_cmap, 
                keep_alpha=True,
            )
                
            # Combine imgages
            rgb_mag_img = np.stack([mag_img] * 3, axis=-1)
            img = _combine_imgs(rgb_mag_img, sig_img, alpha)
            frames += [img]
        
        # Save gif
        output_path = os.path.join(output_dir, f"dhcp_atlas_{ornt}.gif")
        imageio.mimsave(output_path, frames, duration=1/fps, loop=0) 
        
        
def parameter_tuning_plot(
        input_sig_paths,
        input_mask_path,
        input_mask_eroded_path,
        param_info,
        output_plot_path=None,
        slice_index=None,
        ornt="sagittal",
        xlim=None,
        ylim=None,
        ep_clim=None,
        ep_cmap="inferno",
        outline_c=None,
        figsize=None,
        title=None,
        save_imageio=False,
    ):
    
    # Default view
    if slice_index is None:
        dims = _get_img_dim_in_RAS(input_sig_paths[0])
        slice_index =  dims[0]//2
        
    # Number of images
    n = len(input_sig_paths)    
    m = len(input_sig_paths[0])
    param_labels = list(param_info.keys())
    
    # Default 
    if ep_clim is None or not isinstance(ep_clim[0], list):
        ep_clim = [[ep_clim for j in range(m)] for i in range(n)]
    
    # Initialize output dir
    if output_plot_path:
        output_dir = os.path.dirname(output_plot_path)
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize imageio dir
    save_imageio = output_plot_path and save_imageio
    if save_imageio:
        plot_name = os.path.splitext(os.path.basename(output_plot_path))[0]
        imageio_dir = os.path.join(output_dir, "imageio", plot_name)
        os.makedirs(imageio_dir, exist_ok=True)
    
    # Initialize plot
    fig, axes = plt.subplots(m, n, figsize=figsize)
    fig.suptitle(title)
    
    # Iter over param1
    for i in range(n):
        param_i = param_info[param_labels[0]][i]
        
        # Iter over para2
        for j  in range(m):
            param_j = param_info[param_labels[1]][j]
            
            # Extract conductivity image
            sig_img = _extract_slice_img(
                img_path=input_sig_paths[i][j], 
                slice_index=slice_index,
                slice_ornt=ornt, 
                clim=ep_clim[i][j],
                cmap=ep_cmap, 
                xlim=xlim,
                ylim=ylim,
                mask_path=input_mask_path,
                outline_path=input_mask_eroded_path,
                outline_c=outline_c,
            )
            
            # Plot conductivity
            axes[j,i].imshow(sig_img)
            if j == m-1:
                axes[j,i].set_xlabel(f"{param_labels[0].title()} = {param_i}")
            if i == 0:
                axes[j,i].set_ylabel(f"{param_labels[1].title()} = {param_j}")
            axes[j,i].set_xticks([])
            axes[j,i].set_yticks([])
        
            # Save images
            if save_imageio:
                imageio_path = os.path.join(imageio_dir, f"{i}_{j}.png")
                imageio.imwrite(imageio_path, sig_img)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    if output_plot_path:
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    
    
def covariates_plot(
        input_cov_paths,
        input_cov_labels,
        input_mask_path,
        view_info,
        output_plot_path=None,
        ep_clim=None,
        ep_cmap=None,
        figsize=None,
        title=None,
        save_imageio=False,
    ):
    
    # For plotting
    vmin, vmax = 0, 255
    cmap = "gray"
    
    # Default view
    if view_info is None:
        dims = _get_img_dim_in_RAS(input_cov_paths[0])
        view_info = { 
            "axial": {"slice_index": dims[2]//2},
            "coronal": {"slice_index": dims[1]//2},
            "sagittal": {"slice_index": dims[0]//2},
        }
        
    # Number of images
    n = len(input_cov_paths)    
    m = len(view_info)
    
    # Default view
    if ep_clim is None or not isinstance(ep_clim[0], list):
        ep_clim = [[ep_clim for j in range(m)] for i in range(n)]
    if ep_cmap is None or not isinstance(ep_cmap[0], list):
        ep_cmap = [[ep_cmap for j in range(m)] for i in range(n)]
    
    # Initialize output dir
    if output_plot_path:
        output_dir = os.path.dirname(output_plot_path)
        os.makedirs(output_dir, exist_ok=True)
        
    # Initialize imageio dir
    save_imageio = output_plot_path and save_imageio
    if save_imageio:
        plot_name = os.path.splitext(os.path.basename(output_plot_path))[0]
        imageio_dir = os.path.join(output_dir, "imageio", plot_name)
        os.makedirs(imageio_dir, exist_ok=True)
    
    # Initialize figure
    fig, axes = plt.subplots(m, n, figsize=figsize)
    fig.suptitle(title)
    
    # Iter over covariate
    for i in range(n):
        # Iter over view
        for j, (ornt, info) in enumerate(view_info.items()):
            
            # Extract covariate image
            cov_img = _extract_slice_img(
                img_path=input_cov_paths[i], 
                slice_index=info["slice_index"],
                slice_ornt=ornt, 
                clim=ep_clim[i][j],
                cmap=ep_cmap[i][j],
                mask_path=input_mask_path,
            )
            
            # Plot covariate
            axes[j,i].imshow(cov_img, cmap=cmap, vmin=vmin, vmax=vmax)
            if j == m-1:
                axes[j,i].set_xlabel(input_cov_labels[i])
            if i == 0:
                axes[j,i].set_ylabel(f"{ornt.title()} view")
            axes[j,i].set_xticks([])
            axes[j,i].set_yticks([])
        
            # Save images
            if save_imageio:
                imageio_path = os.path.join(imageio_dir, f"{ornt}_{input_cov_labels[i]}.png")
                imageio.imwrite(imageio_path, cov_img)
    
    #Adjust layout
    fig.tight_layout()
    
    # Save plot
    if output_plot_path:
        plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    
    
def annotate_pairplot(x, y, hue=None, ax=None, **kws):
    
    # Calculate limits
    ylim = calculate_plot_lim(y, coeff_margin=0.15)

    # Pearson correlation
    r, p = pearsonr(x, y)

    # Prepare axis
    ax = ax or plt.gca()
    ax.set_ylim(ylim)

    # Coordinates for annotation
    xy = (.1, .9)

    # Extract variable names
    x_name = getattr(x, 'name', 'x')
    y_name = getattr(y, 'name', 'y')

    # Build subscript
    def format_var(var):
        return var.replace("r$", "").replace("$", "").replace("[", "").replace("]", "").split()[0].strip()
    
    # Label
    subscript = f"{format_var(x_name)}-{format_var(y_name)}"
    label = rf"\rho_{{\mathregular{{{subscript}}}}}"

    # Annotate with correlation and p-value significance
    ax.annotate(
        rf"${label} = {r:.2f}$ ({convert_pvalue_to_asterisks(p)})",
        xy=xy, xycoords=ax.transAxes, fontsize=10
    )
    
    
def annotate_tissue_pairplot(x, y, tissue_labels=None, hue=None, ax=None, **kws):

    # Numer of points/tissues
    n = len(x)
    m = len(tissue_labels)
    k = int(n/m)
    
    # Calculate limits
    ylim = calculate_plot_lim(y, coeff_margin=0.15)

    # Prepare axis
    ax = ax or plt.gca()
    ax.set_ylim(ylim)

    # Coordinates for annotation
    xy = (.1, .05)
    
    # Init legend
    lgd = ""

    # Iter over tissues
    for i, tissue in enumerate(tissue_labels):

        if i > 0:
            lgd += "\n"

        # Extract tissue data
        x_tissue = x[k*i:k*(i+1)]
        y_tissue = y[k*i:k*(i+1)]

        # Pearson correlation
        r, p = pearsonr(x_tissue, y_tissue)
    
        # Extract variable names
        x_name = getattr(x, 'name', 'x')
        y_name = getattr(y, 'name', 'y')
    
        # Build subscript 
        def format_var(var):
            return var.replace("r$", "").replace("$", "").replace("[", "").replace("]", "").split()[0].strip()

        # Label
        subscript = f"{format_var(x_name)}-{format_var(y_name)}"
        label = rf"\rho_{{\mathregular{{{subscript}}}}}^{{\mathregular{{{tissue}}}}}"
        lgd += rf"${label} = {r:.2f}$ ({convert_pvalue_to_asterisks(p)})"

    # Annotate with correlation and p-value significance
    ax.annotate(lgd, xy=xy, xycoords=ax.transAxes, fontsize=10)
        

def calculate_plot_lim(
        data, 
        coeff_margin=1, 
        coeff_min=1, 
        coeff_max=1,
    ):
    
    # Extract min/max, adjust with margin
    val_min, val_max = data.min(), data.max()
    val_margin = coeff_margin * (val_max - val_min)
    lim = [val_min - coeff_min * val_margin, val_max + coeff_max * val_margin]
    
    return lim