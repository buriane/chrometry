import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib.colors import rgb2hex, hex2color, to_rgb
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.metrics import silhouette_score
import colorsys
import tempfile
import os

def preprocessing_for_color_segmentation(
    image: np.ndarray, 
    denoise_strength: int = 9,
    contrast_limit: float = 2.0,
    color_space: str = 'LAB'
) -> np.ndarray:
    """
    Enhanced preprocessing for color segmentation with customizable parameters.
    
    Args:
        image (np.ndarray): Input image in BGR or RGB format.
        denoise_strength (int): Strength of the bilateral filter (higher = more smoothing).
        contrast_limit (float): CLAHE contrast limit.
        color_space (str): Color space for enhancement ('LAB', 'HSV', or 'RGB').
        
    Returns:
        np.ndarray: Processed image for color segmentation in RGB format.
    """
    # Ensure image is in the right format
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 3 and np.max(image) <= 1.0:
        image = (image * 255).astype(np.uint8)  # Normalize if image is in float format
    
    # Convert to BGR if input is in RGB format (for OpenCV operations)
    img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR) if image.shape[2] == 3 else image
    
    # Apply bilateral filter to reduce noise while preserving edges
    img_filtered = cv.bilateralFilter(img_bgr, denoise_strength, 75, 75)
    
    # Apply additional denoising for smoother regions
    img_filtered = cv.fastNlMeansDenoisingColored(img_filtered, None, 10, 10, 7, 21)
    
    # Enhance contrast using the specified color space
    if color_space == 'LAB':
        # LAB color space enhancement (better for perceptual color differences)
        lab = cv.cvtColor(img_filtered, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv.merge((l, a, b))
        img_enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    
    elif color_space == 'HSV':
        # HSV color space enhancement (better for color-based segmentation)
        hsv = cv.cvtColor(img_filtered, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        clahe = cv.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv.merge((h, s, v))
        img_enhanced = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    else:  # RGB enhancement
        # Apply CLAHE to each RGB channel independently
        r, g, b = cv.split(img_filtered)
        clahe = cv.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8, 8))
        r = clahe.apply(r)
        g = clahe.apply(g)
        b = clahe.apply(b)
        img_enhanced = cv.merge((r, g, b))
    
    # Convert back to RGB for clustering
    img_rgb = cv.cvtColor(img_enhanced, cv.COLOR_BGR2RGB)
    
    return img_rgb

def determine_optimal_clusters(
    image: np.ndarray, 
    max_clusters: int = 10, 
    min_clusters: int = 2,
    method: str = 'elbow'
) -> int:
    """
    Determine the optimal number of color clusters using either elbow method or silhouette score.
    
    Args:
        image (np.ndarray): Image in RGB format.
        max_clusters (int): Maximum number of clusters to consider.
        min_clusters (int): Minimum number of clusters to consider.
        method (str): Method to determine optimal clusters ('elbow' or 'silhouette').
        
    Returns:
        int: Optimal number of clusters.
    """
    # Reshape image for clustering
    pixels = image.reshape(-1, 3)
    
    # Take a sample to speed up computation for large images
    sample_size = min(100000, pixels.shape[0])
    pixel_sample = pixels[np.random.choice(pixels.shape[0], sample_size, replace=False)]
    
    if method == 'elbow':
        # Elbow method implementation
        distortions = []
        for i in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(pixel_sample)
            distortions.append(kmeans.inertia_)
        
        # Calculate the rate of change in distortion
        deltas = np.diff(distortions)
        # Find "elbow point" - where the rate of improvement significantly slows
        delta_changes = np.diff(deltas)
        elbow_point = np.argmin(delta_changes) + min_clusters
        
        return elbow_point
    
    elif method == 'silhouette':
        # Silhouette score implementation
        silhouette_scores = []
        for i in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pixel_sample)
            
            # Only compute silhouette if we have at least 2 clusters
            if i > 1:
                silhouette_avg = silhouette_score(pixel_sample, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        # Return the number of clusters with the highest silhouette score
        optimal_clusters = np.argmax(silhouette_scores) + min_clusters
        return optimal_clusters
    
    else:
        # Default to 5 clusters if method is not recognized
        return 5

def segment_colors(
    image: np.ndarray, 
    n_clusters: int = None,
    auto_determine: bool = True,
    max_clusters: int = 10,
    use_preprocessing: bool = False  # New parameter to toggle preprocessing
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Perform color segmentation using K-Means clustering with optional automatic cluster determination.
    
    Args:
        image (np.ndarray): Input image in BGR or RGB format.
        n_clusters (int, optional): Number of color clusters to use. If None and auto_determine is True,
                                   will use automatic determination.
        auto_determine (bool): Whether to automatically determine the optimal number of clusters.
        max_clusters (int): Maximum number of clusters to consider if auto_determine is True.
        use_preprocessing (bool): Whether to use preprocessing or work with the original image.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]: 
            - Segmented image
            - Image with cluster labels
            - Cluster centers (dominant colors)
            - List of cluster masks
    """
    # Ensure image is in RGB format for clustering
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # Check if image is BGR (OpenCV default) or RGB
        # This is a heuristic - we assume if it's from cv.imread it's BGR
        if np.mean(image[:,:,0]) < np.mean(image[:,:,2]):  # BGR tends to have B channel < R channel
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # Apply preprocessing if requested
    if use_preprocessing:
        # Preprocess image for potentially better segmentation
        preprocessed_img = preprocessing_for_color_segmentation(image_rgb)
    else:
        # Use original image
        preprocessed_img = image_rgb
    
    # Determine optimal number of clusters if requested
    if auto_determine or n_clusters is None:
        n_clusters = determine_optimal_clusters(
            preprocessed_img, 
            max_clusters=max_clusters, 
            min_clusters=2,
            method='elbow'
        )
    
    # Reshape image for clustering
    h, w, c = preprocessed_img.shape
    reshaped_img = preprocessed_img.reshape((h * w, c))
    
    # Apply K-Means for color clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reshaped_img)
    centers = kmeans.cluster_centers_
    
    # Create segmented image using cluster centers
    segmented_img = centers[labels].reshape((h, w, c)).astype(np.uint8)
    
    # Reshape labels for visualization
    label_img = labels.reshape((h, w))
    
    # Create individual cluster masks
    cluster_images = []
    for i in range(n_clusters):
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        cluster_mask[label_img == i] = 255
        cluster_images.append(cluster_mask)
    
    return segmented_img, label_img, centers, cluster_images

def classify_color(rgb_color: np.ndarray) -> Tuple[str, str, float]:
    """
    Classify a color into primary, secondary, tertiary, or other categories
    using HSV color space for better color differentiation.
    
    Args:
        rgb_color (np.ndarray): RGB color to classify (0-255 range).
        
    Returns:
        Tuple[str, str, float]: Category, color name, and confidence score
    """
    # Normalize RGB to 0-1 range
    r, g, b = rgb_color / 255.0
    
    # Convert to HSV for better color classification
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Create a hue-based classification
    hue_degrees = h * 360
    
    # Setup color categories with HSV ranges
    PRIMARY_COLORS = {
        'Merah': {'hue_range': [(0, 15), (345, 360)], 'min_saturation': 0.5, 'min_value': 0.2},
        'Hijau': {'hue_range': [(90, 150)], 'min_saturation': 0.4, 'min_value': 0.2},
        'Biru': {'hue_range': [(210, 270)], 'min_saturation': 0.4, 'min_value': 0.2},
    }
    
    SECONDARY_COLORS = {
        'Kuning': {'hue_range': [(40, 70)], 'min_saturation': 0.5, 'min_value': 0.5},
        'Magenta': {'hue_range': [(280, 320)], 'min_saturation': 0.4, 'min_value': 0.2},
        'Cyan': {'hue_range': [(170, 200)], 'min_saturation': 0.4, 'min_value': 0.2},
    }
    
    TERTIARY_COLORS = {
        'Oranye': {'hue_range': [(16, 39)], 'min_saturation': 0.5, 'min_value': 0.5},
        'Chartreuse': {'hue_range': [(71, 89)], 'min_saturation': 0.5, 'min_value': 0.5},
        'Spring green': {'hue_range': [(151, 169)], 'min_saturation': 0.4, 'min_value': 0.5},
        'Azure': {'hue_range': [(201, 209)], 'min_saturation': 0.4, 'min_value': 0.5},
        'Violet': {'hue_range': [(271, 279)], 'min_saturation': 0.4, 'min_value': 0.2},
        'Rose': {'hue_range': [(321, 344)], 'min_saturation': 0.4, 'min_value': 0.2},
    }
    
    # Function to check if a hue falls within specified ranges
    def is_in_hue_range(hue, ranges):
        return any(lower <= hue <= upper for lower, upper in ranges)
    
    # Check for grayscale
    if s < 0.15:
        if v < 0.15:
            return 'Lainnya', 'Hitam', 0.9
        elif v > 0.85:
            return 'Lainnya', 'Putih', 0.9
        else:
            gray_level = int(v * 255)
            return 'Lainnya', f'Abu-abu ({gray_level})', 0.9
    
    # Check through color categories
    best_match = None
    best_confidence = 0
    color_category = 'Lainnya'
    
    # Check primary colors
    for name, properties in PRIMARY_COLORS.items():
        if (is_in_hue_range(hue_degrees, properties['hue_range']) and 
            s >= properties['min_saturation'] and 
            v >= properties['min_value']):
            confidence = s * v  # Higher saturation and value = higher confidence
            if confidence > best_confidence:
                best_match = name
                best_confidence = confidence
                color_category = 'Primer'
    
    # Check secondary colors
    for name, properties in SECONDARY_COLORS.items():
        if (is_in_hue_range(hue_degrees, properties['hue_range']) and 
            s >= properties['min_saturation'] and 
            v >= properties['min_value']):
            confidence = s * v
            if confidence > best_confidence:
                best_match = name
                best_confidence = confidence
                color_category = 'Sekunder'
    
    # Check tertiary colors
    for name, properties in TERTIARY_COLORS.items():
        if (is_in_hue_range(hue_degrees, properties['hue_range']) and 
            s >= properties['min_saturation'] and 
            v >= properties['min_value']):
            confidence = s * v
            if confidence > best_confidence:
                best_match = name
                best_confidence = confidence
                color_category = 'Tersier'
    
    # If no match is found, use hue-based naming
    if best_match is None:
        if hue_degrees < 30 or hue_degrees > 330:
            best_match = f'Merah Keabuan ({int(rgb_color[0])},{int(rgb_color[1])},{int(rgb_color[2])})'
        elif hue_degrees < 90:
            best_match = f'Kuning Keabuan ({int(rgb_color[0])},{int(rgb_color[1])},{int(rgb_color[2])})'
        elif hue_degrees < 150:
            best_match = f'Hijau Keabuan ({int(rgb_color[0])},{int(rgb_color[1])},{int(rgb_color[2])})'
        elif hue_degrees < 210:
            best_match = f'Cyan Keabuan ({int(rgb_color[0])},{int(rgb_color[1])},{int(rgb_color[2])})'
        elif hue_degrees < 270:
            best_match = f'Biru Keabuan ({int(rgb_color[0])},{int(rgb_color[1])},{int(rgb_color[2])})'
        else:
            best_match = f'Magenta Keabuan ({int(rgb_color[0])},{int(rgb_color[1])},{int(rgb_color[2])})'
    
    return color_category, best_match, best_confidence

def analyze_color_dominance(
    image: np.ndarray, 
    centers: np.ndarray, 
    labels: np.ndarray, 
    min_percentage: float = 1.0
) -> pd.DataFrame:
    """
    Enhanced analysis of dominant colors with improved classification and filtering.
    
    Args:
        image (np.ndarray): Original image.
        centers (np.ndarray): Cluster centers (dominant colors).
        labels (np.ndarray): Pixel labels from clustering.
        min_percentage (float): Minimum percentage to include in results.
        
    Returns:
        pd.DataFrame: DataFrame with color information, filtered by minimum percentage.
    """
    # Count occurrence of each label
    label_counts = Counter(labels.flatten())
    total_pixels = len(labels.flatten())
    
    # Prepare color information
    color_info = []
    
    # Sort clusters by size (descending)
    for i, (cluster_idx, count) in enumerate(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)):
        center = centers[cluster_idx]
        percentage = (count / total_pixels) * 100
        
        # Skip colors with percentage below threshold
        if percentage < min_percentage:
            continue
        
        # Make sure RGB values are within 0-255 range
        center_normalized = np.clip(center, 0, 255)
        hex_color = rgb2hex(center_normalized / 255)
        
        # Classify color using enhanced classification
        color_category, color_name, confidence = classify_color(center_normalized)
        
        # Format RGB values for display
        rgb_values = f"({int(center_normalized[0])}, {int(center_normalized[1])}, {int(center_normalized[2])})"
        
        # Add HSV values for more complete information
        r, g, b = center_normalized / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_values = f"({int(h*360)}°, {int(s*100)}%, {int(v*100)}%)"
        
        color_info.append({
            'Cluster': i+1,
            'Persentase (%)': round(percentage, 2),
            'Kategori': color_category,
            'Nama Warna': color_name,
            'RGB': rgb_values,
            'HSV': hsv_values,
            'Hex': hex_color,
            'Confidence': round(confidence * 100, 1)
        })
    
    return pd.DataFrame(color_info)

def generate_color_palette(
    color_info: pd.DataFrame, 
    palette_size: int = 5, 
    width: int = 500, 
    height: int = 100
) -> np.ndarray:
    """
    Generate a visual color palette from color analysis results.
    
    Args:
        color_info (pd.DataFrame): DataFrame with color information.
        palette_size (int): Maximum number of colors to include in palette.
        width (int): Width of the palette image.
        height (int): Height of the palette image.
        
    Returns:
        np.ndarray: Generated color palette image.
    """
    # Limit to top colors based on percentage
    df_sorted = color_info.sort_values(by='Persentase (%)', ascending=False).head(palette_size)
    
    # Create a blank palette image
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(df_sorted) == 0:
        return palette
    
    # Calculate section width based on percentages
    total_percentage = df_sorted['Persentase (%)'].sum()
    x_position = 0
    
    for _, row in df_sorted.iterrows():
        # Calculate width of this color section
        section_percentage = row['Persentase (%)'] / total_percentage
        section_width = int(width * section_percentage)
        
        # Get RGB values from hex color
        hex_color = row['Hex']
        try:
            # Use to_rgb instead of hex2color for safer conversion
            r, g, b = [int(255 * c) for c in to_rgb(hex_color)]
        except ValueError:
            # Fallback if hex color is invalid
            r, g, b = 128, 128, 128  # Default to gray
        
        # Fill the section with this color
        if x_position + section_width <= width:
            palette[:, x_position:x_position+section_width] = [r, g, b]  # RGB for matplotlib
        else:
            palette[:, x_position:] = [r, g, b]  # Fill remaining space
        
        x_position += section_width
    
    return palette

def save_color_segmentation_plot(
    original_img: np.ndarray, 
    segmented_img: np.ndarray, 
    color_info: pd.DataFrame, 
    cluster_images: List[np.ndarray],
    max_clusters_to_show: int = 3
) -> str:
    """
    Generate and save color segmentation visualization to a temporary file.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Display original image
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        axes[0].imshow(cv.cvtColor(original_img, cv.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Citra Asli')
    axes[0].axis('off')
    
    # Display segmented image
    axes[1].imshow(segmented_img)
    axes[1].set_title('Citra Tersegmentasi')
    axes[1].axis('off')
    
    # Display color distribution pie chart
    safe_colors = []
    percentages = []
    labels = []
    
    for _, row in color_info.iterrows():
        try:
            rgb = to_rgb(row['Hex'])
            safe_colors.append(rgb)
            percentages.append(row['Persentase (%)'])
            if len(color_info) > 5:
                labels.append(f"Cluster {len(labels)+1}\n({row['Persentase (%)']}%)")
            else:
                labels.append(f"{row['Nama Warna']}\n({row['Persentase (%)']}%)")
        except ValueError:
            continue
    
    if safe_colors:
        axes[2].pie(
            percentages, 
            labels=labels, 
            colors=safe_colors,
            autopct='%1.1f%%',
            startangle=90
        )
    axes[2].set_title('Distribusi Warna')
    
    # Display individual clusters
    max_display = min(max_clusters_to_show, len(cluster_images))
    for i in range(max_display):
        if i < len(color_info):
            color_name = color_info.iloc[i]['Nama Warna']
            percentage = color_info.iloc[i]['Persentase (%)']
            category = color_info.iloc[i]['Kategori']
            
            # Create RGB overlay
            h, w = cluster_images[i].shape
            mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            
            try:
                hex_color = color_info.iloc[i]['Hex']
                r, g, b = [int(255 * c) for c in to_rgb(hex_color)]
                mask_rgb[cluster_images[i] > 0] = [r, g, b]
                
                # Create blend
                if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                    original_rgb = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
                else:
                    original_rgb = cv.cvtColor(original_img, cv.COLOR_GRAY2RGB)
                
                blend = cv.addWeighted(original_rgb, 0.7, mask_rgb, 0.3, 0)
                
                axes[i+3].imshow(blend)
                axes[i+3].set_title(f'{category}: {color_name}\n({percentage:.1f}%)')
            except ValueError:
                axes[i+3].imshow(cluster_images[i], cmap='gray')
                axes[i+3].set_title(f'Cluster {i+1}')
            
            axes[i+3].axis('off')
    
    # Turn off any unused subplots
    for i in range(max_display + 3, 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return temp_file.name

def save_color_palette_plot(color_info: pd.DataFrame) -> str:
    """
    Generate and save a color palette visualization to a temporary file.
    
    Args:
        color_info (pd.DataFrame): DataFrame with color information.
        
    Returns:
        str: Path to the saved palette file.
    """
    # Create color palette
    color_palette = generate_color_palette(color_info)
    
    # Create visualization
    fig = plt.figure(figsize=(10, 3))
    plt.imshow(color_palette)
    plt.title('Palet Warna Dominan')
    plt.axis('off')
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, bbox_inches='tight')
    plt.close(fig)
    
    return temp_file.name

def save_color_category_plot(color_info: pd.DataFrame) -> str:
    """
    Generate and save a color category distribution visualization to a temporary file.
    
    Args:
        color_info (pd.DataFrame): Information about colors from analysis.
        
    Returns:
        str: Path to the saved category plot file.
    """
    # Group by color category
    category_data = color_info.groupby('Kategori')['Persentase (%)'].sum().reset_index()
    
    # Ensure all categories are present
    for category in ['Primer', 'Sekunder', 'Tersier', 'Lainnya']:
        if category not in category_data['Kategori'].values:
            category_data = pd.concat([
                category_data, 
                pd.DataFrame({'Kategori': [category], 'Persentase (%)': [0]})
            ])
    
    # Sort categories
    category_order = ['Primer', 'Sekunder', 'Tersier', 'Lainnya']
    category_data = category_data.set_index('Kategori').reindex(category_order).reset_index()
    
    # Set colors for each category
    category_colors = {
        'Primer': '#FF5733',     # Red-orange
        'Sekunder': '#33FF57',   # Green-yellow
        'Tersier': '#5733FF',    # Purple-blue
        'Lainnya': '#AAAAAA'     # Gray
    }
    
    # Create safe colors for matplotlib (ensure they're in 0-1 range)
    colors = []
    for category in category_data['Kategori']:
        try:
            colors.append(to_rgb(category_colors.get(category, '#AAAAAA')))
        except ValueError:
            colors.append((0.5, 0.5, 0.5))  # Default to gray if conversion fails
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        category_data['Persentase (%)'], 
        labels=category_data['Kategori'], 
        colors=colors,
        autopct='%1.1f%%', 
        startangle=90,
        explode=[0.05, 0.05, 0.05, 0.05],
        shadow=True,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Enhance text properties
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax1.set_title('Distribusi Kategori Warna', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(
        category_data['Kategori'], 
        category_data['Persentase (%)'], 
        color=colors,
        edgecolor='black',
        linewidth=1,
        alpha=0.85
    )
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., 
            height + 1,
            f'{category_data["Persentase (%)"].iloc[i]:.1f}%', 
            ha='center', 
            va='bottom',
            fontweight='bold',
            fontsize=11
        )
    
    # Enhance appearance
    ax2.set_title('Persentase per Kategori Warna', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Persentase (%)', fontsize=12)
    ax2.set_ylim(0, max(100, category_data['Persentase (%)'].max() * 1.2))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add color theory legend
    fig.text(0.5, 0, """
    Keterangan:
    • Primer: Warna dasar (Merah, Hijau, Biru)
    • Sekunder: Campuran 2 warna primer (Kuning, Magenta, Cyan)
    • Tersier: Campuran warna primer dan sekunder
    • Lainnya: Warna netral atau campuran kompleks
    """, ha='center', fontsize=10, bbox=dict(facecolor='#f9f9f9', edgecolor='#ddd', boxstyle='round,pad=1'))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, bbox_inches='tight')
    plt.close(fig)
    
    return temp_file.name

def calculate_color_features(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate various color features from the image.
    
    Args:
        image (np.ndarray): Input image in BGR format (OpenCV default).
        
    Returns:
        Dict[str, float]: Dictionary of calculated color features.
    """
    # Ensure image is in RGB format
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    else:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Calculate basic color statistics
    mean_rgb = np.mean(image_rgb, axis=(0, 1))
    std_rgb = np.std(image_rgb, axis=(0, 1))
    
    # Split channels
    r, g, b = cv.split(image_rgb)
    
    # Calculate brightness (average luminance)
    brightness = np.mean(0.299 * r + 0.587 * g + 0.114 * b)
    
    # Calculate contrast
    grayscale = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
    contrast = np.std(grayscale)
    
    # Calculate colorfulness using Hasler and Süsstrunk's method
    rg = r.astype(np.float32) - g.astype(np.float32)
    yb = 0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32)
    rg_std = np.std(rg)
    rg_mean = np.mean(np.abs(rg))
    yb_std = np.std(yb)
    yb_mean = np.mean(np.abs(yb))
    colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
    
    # Calculate color diversity (normalized unique colors)
    pixels = image_rgb.reshape(-1, 3)
    # Sample pixels for performance
    sample_size = min(100000, pixels.shape[0])
    sampled_pixels = pixels[np.random.choice(pixels.shape[0], sample_size, replace=False)]
    # Round values to reduce sensitivity to minor variations
    rounded_pixels = np.round(sampled_pixels / 8) * 8
    unique_colors = np.unique(rounded_pixels, axis=0)
    color_diversity = len(unique_colors) / sample_size
    
    # Calculate color balance (normalized difference between channels)
    r_mean, g_mean, b_mean = mean_rgb
    max_channel = max(r_mean, g_mean, b_mean)
    min_channel = min(r_mean, g_mean, b_mean)
    if max_channel > 0:
        color_balance = 1.0 - ((max_channel - min_channel) / max_channel)
    else:
        color_balance = 1.0
    
    # Return all features in a dictionary
    return {
        'brightness': brightness,
        'contrast': contrast,
        'colorfulness': colorfulness,
        'color_diversity': color_diversity,
        'color_balance': color_balance,
        'mean_red': mean_rgb[0],
        'mean_green': mean_rgb[1],
        'mean_blue': mean_rgb[2],
        'std_red': std_rgb[0],
        'std_green': std_rgb[1],
        'std_blue': std_rgb[2]
    }

def process_color_segmentation(
    image_path: str,
    n_clusters: int = None, 
    auto_determine: bool = True,
    max_clusters: int = 8, 
    min_percentage: float = 1.0
) -> Tuple[Dict[str, any], List[str]]:
    """
    Process an image for color segmentation analysis.
    This function performs color segmentation, analysis, and generates visualizations.
    
    Args:
        image_path (str): Path to the image file.
        n_clusters (int, optional): Number of color clusters to use.
        auto_determine (bool): Whether to automatically determine optimal clusters.
        max_clusters (int): Maximum number of clusters to consider.
        min_percentage (float): Minimum percentage to include in results.
        
    Returns:
        Tuple[Dict[str, any], List[str]]: Dictionary with analysis results and list of temporary file paths.
    """
    try:
        # Load image
        original_img = cv.imread(image_path)
        if original_img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Perform segmentation directly on the original image
        # without preprocessing for more accurate original color representation
        segmented_img, label_img, centers, cluster_images = segment_colors(
            original_img,
            n_clusters=n_clusters,
            auto_determine=auto_determine,
            max_clusters=max_clusters,
            use_preprocessing=False  # Use original image without preprocessing
        )
        
        # Analyze color dominance
        color_info = analyze_color_dominance(
            original_img,
            centers,
            label_img,
            min_percentage=min_percentage
        )
        
        # Calculate additional color features
        color_features = calculate_color_features(original_img)
        
        # Generate and save visualizations
        temp_files = []
        
        # Save color segmentation plot
        segmentation_plot_path = save_color_segmentation_plot(
            original_img, 
            segmented_img, 
            color_info, 
            cluster_images
        )
        temp_files.append(segmentation_plot_path)
        
        # Save color palette plot
        palette_plot_path = save_color_palette_plot(color_info)
        temp_files.append(palette_plot_path)
        
        # Save category distribution plot
        category_plot_path = save_color_category_plot(color_info)
        temp_files.append(category_plot_path)
        
        # Prepare and return results
        results = {
            'color_info': color_info,
            'color_features': color_features,
            'centers': centers,
            'label_img': label_img
        }
        
        return results, temp_files
    
    except Exception as e:
        # Clean up any temp files in case of error
        if 'temp_files' in locals():
            for file_path in temp_files:
                if os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass
        
        # Re-raise the error
        raise Exception(f"Error in color segmentation: {str(e)}")

def main():
    """
    Example usage of the color segmentation module.
    """
    # Example path to an image
    image_path = input("Enter path to image file: ")
    
    # Handle default path
    if not image_path.strip():
        image_path = "sample_image.jpg"
        print(f"Using default image: {image_path}")
    
    try:
        # Perform segmentation and get results
        results, temp_files = process_color_segmentation(
            image_path,
            auto_determine=True,
            max_clusters=8,
            min_percentage=1.0
        )
        
        # Print color analysis results
        print("\n=== HASIL ANALISIS WARNA ===")
        print(results['color_info'].to_string(index=False))
        
        # Print color feature results
        print("\n=== FITUR WARNA ===")
        for feature, value in results['color_features'].items():
            print(f"{feature}: {value:.2f}")
        
        print(f"\nVisualisasi telah disimpan ke: {', '.join(temp_files)}")
        
        # Cleanup temp files
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()