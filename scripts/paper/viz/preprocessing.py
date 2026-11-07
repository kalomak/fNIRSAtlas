import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from itertools import compress
import mne
from mne.io import Raw
from mne import Epochs
import numpy as np
import warnings
from fnirsatlas.config import get_results_dir
from fnirsatlas.load_parser import load_parser
from fnirsatlas.epoching import create_epochs_from_raw
from fnirsatlas.preprocessing import (
    _get_preprocessing_config_with_overrides,
    temporal_derivative_distribution_repair,
    apply_sci,
    beer_lambert_law,
    _has_valid_montage
)

mne.viz.set_browser_backend("matplotlib")

dataset_name = "luke2021"
config = _get_preprocessing_config_with_overrides(dataset_name)
parser = load_parser(dataset_name)
sub = parser.get_subject_ids()[0]
session = parser.get_sessions(sub)[0]
data = parser.load_data(sub, session=session)
raw = parser._raw_from_session(data)
c0, c1 = 0, 40
plot_channels = [c0, c0+1, c1, c1+1]
start=135
raw = raw.crop(start)
start=0


def plot_raw(raw, plot_channels, title, filename, start=0, duration=100, 
             pwidth=8, pheight=2, delta=0.0, y_offset=-0.05, height_offset=-0.15, scalings="auto",
             show_xlabel=False, show_stm=False):
    """Plot raw data with consistent formatting."""
    fig = (
        raw.copy()
        .pick(picks=plot_channels)
        .plot(
            start=start,
            duration=duration,
            scalings=scalings,
            show=False,
            decim=1,
            show_scrollbars=False,
            title=title,
        )
    )
    # set title (argument title is not working)
    fig.axes[0].set_title(
        title,
        fontsize=12,
        pad=0,
        loc="right",
        x=0.99,
        y=0.86,
        fontstyle="italic",
    )
    fig.set_size_inches(pwidth, pheight)
    if not show_xlabel:
        fig.axes[0].set_xticklabels([])
        fig.axes[0].set_xlabel("")
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    x0 = bbox.x0
    x1 = bbox.width + 0.02
    y0 = bbox.y0 - delta + y_offset
    y1 = bbox.height + 2 * delta + height_offset
    if show_stm:
        y1 += .3
    if show_xlabel:
        x0 -= 0.04
    new_bbox = Bbox.from_bounds(
        x0,
        y0,
        x1,
        y1,
    )
    # Save the figure with the modified bounding box
    fig.savefig(filename, dpi=300, bbox_inches=new_bbox, pad_inches=0)

def plot_epochs(epochs, plot_channels, title, filename, n_epochs=None, 
                pwidth=8, pheight=2, delta=0.0, y_offset=-0.05, height_offset=0.1, scalings="auto"):
    """Plot epochs data with consistent formatting."""
    if n_epochs is None:
        n_epochs = 3
    fig = (
        epochs.copy()
        .load_data()
        .pick(picks=plot_channels)
        .plot(
            n_epochs=n_epochs,
            scalings=scalings,
            show=False,
            decim=1,
            show_scrollbars=False,
            title=title,
            events=True,
            event_id=True,
        )
    )
    # set title (argument title is not working)
    fig.axes[0].set_title(
        title,
        fontsize=12,
        pad=0,
        loc="right",
        x=0.99,
        y=0.86,
        fontstyle="italic",
    )
    fig.set_size_inches(pwidth, pheight)
    # Get the current tight bounding box
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    # Adjust the bounding box
    new_bbox = Bbox.from_bounds(
        bbox.x0 + 0.,
        bbox.y0 - delta + y_offset,
        bbox.width + 0.22,
        bbox.height + 2 * delta + height_offset,
    )
    # Save the figure with the modified bounding box
    fig.savefig(filename, dpi=300, bbox_inches=new_bbox, pad_inches=0)

plot_dir = get_results_dir() / "preprocessing"
plot_dir.mkdir(parents=True, exist_ok=True)
plot_raw(raw, plot_channels, "Raw", plot_dir / "0raw_data.png", show_stm=True, start=start)

# current_channel_types = set(raw.get_channel_types())
raw = mne.preprocessing.nirs.optical_density(raw)
plot_raw(raw, plot_channels, "OD", plot_dir / "1raw_data.png", start=start)

# TDDR
if config.get("tddr", True):
    raw = temporal_derivative_distribution_repair(raw)
plot_raw(raw, plot_channels, "TDDR", plot_dir / "2raw_data.png", start=start)

# SCI
config['sci_threshold'] = 0.8
if config.get("sci_threshold", 0) > 0:
    raw = apply_sci(raw, config["sci_threshold"])
plot_raw(raw, plot_channels, "SCI", plot_dir / "3raw_data.png", start=start)

# Beer-Lambert Law
raw = beer_lambert_law(raw, ppf=config.get("bll_ppf", 6))
plot_raw(raw, plot_channels, "MBLL", plot_dir / "4raw_data.png", start=start)

# Channel interpolation
has_montage = _has_valid_montage(raw)
if has_montage and config.get("ch_interpolation") == "interpolate_nearest":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        raw.interpolate_bads(verbose=False)
plot_raw(raw, plot_channels, "Interp.", plot_dir / "5raw_data.png", start=start)

filter_kwargs = config.get("filter_kwargs", {})
raw = raw.filter(
    l_freq=config.get("l_freq", 0.01),
    h_freq=config.get("h_freq", 0.1),
    l_trans_bandwidth=config.get("l_trans_bandwidth", 0.004),
    h_trans_bandwidth=config.get("h_trans_bandwidth", 0.01),
    filter_length=config.get("filter_length", "auto"),
    verbose=False,
    **filter_kwargs,
)
plot_raw(raw, plot_channels, "Filtering", plot_dir / "6raw_data.png", scalings={"hbo": 4e-7, "hbr": 1e-7}, show_xlabel=True, start=start)

raw = create_epochs_from_raw(raw)
plot_epochs(raw, plot_channels, "Epochs", plot_dir / "7raw_data.png", scalings={"hbo": 3e-7, "hbr": 5e-8})

def vstack_images(image_paths, output_path, spacing=0):
    """
    Vertically stack multiple images using PIL.
    
    Parameters:
    -----------
    image_paths : list of str or Path
        List of paths to image files to stack vertically
    output_path : str or Path
        Path where the stacked image will be saved
    spacing : int, optional
        Number of pixels of spacing between images (default: 0)
    """
    from PIL import Image
    
    # Open all images and convert to RGB if needed
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    
    if not images:
        raise ValueError("No images provided")
    
    # Get the width of the widest image
    max_width = max(img.width for img in images)
    
    # Calculate total height including spacing
    total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
    
    # Create new image with white background
    stacked_image = Image.new('RGB', (max_width, total_height), color='white')
    
    # Paste each image
    y_offset = 0
    for img in images:
        # Center the image horizontally if it's narrower than max_width
        x_offset = (max_width - img.width) // 2
        stacked_image.paste(img, (x_offset, y_offset))
        y_offset += img.height + spacing
    
    # Save the result
    stacked_image.save(output_path, dpi=(300, 300))
    print(f"Stacked image saved to: {output_path}")
    
    # Close all opened images
    for img in images:
        img.close()

# Stack all images in the preprocessing directory
if plot_dir.exists():
    image_files = sorted([f for f in plot_dir.glob("*.png")])
    if image_files:
        vstack_images(image_files, "test.png", spacing=5)
    else:
        print("No PNG images found in preprocessing directory")
else:
    print(f"Preprocessing directory {plot_dir} does not exist")
