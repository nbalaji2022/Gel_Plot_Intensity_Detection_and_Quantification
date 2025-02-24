from skimage import io, measure, filters
import matplotlib.pyplot as plt
import numpy as np
import os

def detect_and_label_blobs(image_path):
    """Detect fluorescence markers using centroid-based detection and perform statistical analysis."""
    # Load the image
    image = io.imread(image_path)

    # Get image height and define the target region (3/8 to 5/8 of height)
    height = image.shape[0]
    y_start = int((20 / 50) * height)
    y_end = int((30 / 50) * height)

    # Crop the image to the defined region
    cropped_image = image[y_start:y_end, :, :]

    # Extract the green channel (assuming fluorescence is in the green channel)
    green_channel = cropped_image[:, :, 1]

    # Use Otsu's threshold and lower it for more detection
    threshold = filters.threshold_otsu(green_channel)
    lowered_threshold = threshold * 0.8  # Lower the threshold to capture more points
    binary_mask = green_channel > lowered_threshold  # Create a binary mask

    # Label the regions and calculate centroids
    labeled_regions = measure.label(binary_mask)
    region_props = measure.regionprops(labeled_regions)

    # Find min and max intensity in the cropped region for normalization
    min_intensity = np.min(green_channel)
    max_intensity = np.max(green_channel)

    # Avoid division by zero in case of uniform intensity
    if max_intensity == min_intensity:
        max_intensity += 1e-5

    # Initialize lists to store intensity data for statistical analysis
    intensities = []

    # Display the original image
    fig, ax = plt.subplots()
    ax.imshow(image[:, :, 1], cmap='gray')  # Display the green channel

    # Process each detected fluorescence region
    for region in region_props:
        y, x = region.centroid  # True fluorescence center
        y, x = int(y), int(x)  # Convert to integers
        y_full = y + y_start  # Adjust y-coordinate to match the full image

        # Expand sampling to a 5×5 region for better accuracy
        x_min, x_max = max(0, x - 2), min(green_channel.shape[1] - 1, x + 2)
        y_min, y_max = max(0, y - 2), min(green_channel.shape[0] - 1, y + 2)
        intensity = np.mean(green_channel[y_min:y_max + 1, x_min:x_max + 1])

        # Normalize intensity between 0 and 1
        norm_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)

        # Append the intensity to the list for statistical analysis
        intensities.append(norm_intensity)

        # Plot red dot at the **true centroid**
        ax.plot(x, y_full, 'ro', markersize=5)  # Red dot (size = 3 pixels)

        # Print normalized intensity value below the dot (white text)
        ax.text(x, y_full + 15, f"{norm_intensity:.2f}", color='white', fontsize=8, ha='center')

    # Statistical Analysis
    if intensities:
        mean_intensity = np.mean(intensities)
        median_intensity = np.median(intensities)
        std_intensity = np.std(intensities)
        min_detected_intensity = np.min(intensities)
        max_detected_intensity = np.max(intensities)
        num_regions = len(intensities)

        # Display the statistics in a separate figure window
        fig_stats, ax_stats = plt.subplots(figsize=(6, 4))
        ax_stats.axis('off')  # Hide the axes
        ax_stats.text(0.1, 0.9, f"Number of Detected Regions: {num_regions}", fontsize=12)
        ax_stats.text(0.1, 0.75, f"Mean Intensity: {mean_intensity:.2f}", fontsize=12)
        ax_stats.text(0.1, 0.6, f"Median Intensity: {median_intensity:.2f}", fontsize=12)
        ax_stats.text(0.1, 0.45, f"Std Dev of Intensity: {std_intensity:.2f}", fontsize=12)
        ax_stats.text(0.1, 0.3, f"Min Intensity: {min_detected_intensity:.2f}", fontsize=12)
        ax_stats.text(0.1, 0.15, f"Max Intensity: {max_detected_intensity:.2f}", fontsize=12)

        plt.title("Statistical Analysis of Detected Fluorescence Markers")
        plt.show()

    else:
        print("No regions detected.")

    # Display the image with markers and title
    ax.set_title(f"Detected Fluorescence Markers with Normalized Intensity (Centroid-Based) - {image_path}")
    ax.axis('off')  # Hide axis
    plt.show()


# Loop through multiple images in a directory
image_dir = "Assets"  # Replace with your directory path
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    detect_and_label_blobs(image_path)
