from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
from PIL import Image
import pandas as pd

class ColorSegmentationProcessor:
    def __init__(self, image, n_colors=7):
        """
        Initializes the ColorSegmentationProcessor class.
        
        Args:
        image: Input image (numpy array or PIL image)
        n_colors: The number of colors to extract from the image (default is 7)
        """
        self.image = image
        self.n_colors = n_colors
        self.kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        self.palette = None
        self.rgb_palette = None
        self.processed_pixels = None
        self.color_percentages = None
        self.cluster_labels = None
        self.dominant_colors = None

    def preprocess_image(self):
        """
        Preprocesses the image by converting to LAB color space and reshaping for clustering.
        
        Returns:
        pixel_values: Reshaped pixel values ready for clustering
        """
        # Convert image to numpy array if needed
        if hasattr(self.image, 'convert'):
            img_array = np.array(self.image.convert('RGB'))
        elif hasattr(self.image, 'shape'):
            img_array = self.image
        else:
            # Handle other types of image objects
            img_array = np.array(self.image)
        
        # Ensure image is in correct format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Convert RGB to LAB color space
            lab_image = cv.cvtColor(img_array, cv.COLOR_RGB2LAB)
        else:
            raise ValueError("Image must be in RGB format with shape (height, width, 3)")
        
        # Reshape image to be a list of pixels
        pixel_values = lab_image.reshape((-1, 3))
        self.processed_pixels = pixel_values
        
        return pixel_values

    def extract_color_palette(self):
        """
        Extracts color palette using K-means clustering.
        
        Returns:
        rgb_palette: The RGB color palette as a numpy array
        """
        if self.processed_pixels is None:
            pixel_values = self.preprocess_image()
        else:
            pixel_values = self.processed_pixels
        
        # Fit K-means clustering
        self.kmeans.fit(pixel_values)
        self.palette = self.kmeans.cluster_centers_
        
        # Get cluster labels and calculate percentages
        self.cluster_labels = self.kmeans.labels_
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        total_pixels = len(self.cluster_labels)
        self.color_percentages = (counts / total_pixels) * 100
        
        # Convert LAB palette back to RGB
        lab_palette = np.uint8(self.palette)
        lab_palette_3d = lab_palette.reshape(1, -1, 3)
        rgb_palette_3d = cv.cvtColor(lab_palette_3d, cv.COLOR_LAB2RGB)
        self.rgb_palette = rgb_palette_3d.reshape(-1, 3)
        
        # Extract dominant colors (sort by percentage)
        self._extract_dominant_colors()
        
        return self.rgb_palette

    def _extract_dominant_colors(self):
        """
        Extracts dominant colors sorted by their percentage in the image.
        """
        if self.rgb_palette is not None and self.color_percentages is not None:
            # Create pairs of (color, percentage, index) and sort by percentage
            color_data = [(self.rgb_palette[i], self.color_percentages[i], i) 
                         for i in range(len(self.rgb_palette))]
            
            # Sort by percentage in descending order
            color_data.sort(key=lambda x: x[1], reverse=True)
            
            self.dominant_colors = {
                'colors': [item[0] for item in color_data],
                'percentages': [item[1] for item in color_data],
                'original_indices': [item[2] for item in color_data]
            }

    def classify_color_category(self, rgb_color):
        """
        Classifies a color into primary, secondary, tertiary, or other categories.
        
        Args:
        rgb_color: RGB color array [R, G, B]
        
        Returns:
        category: String indicating color category
        """
        r, g, b = rgb_color
        
        # Convert to HSV for better color classification
        rgb_normalized = np.array([[[r, g, b]]], dtype=np.uint8)
        hsv = cv.cvtColor(rgb_normalized, cv.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        # Define thresholds
        saturation_threshold = 50  # Low saturation = grayscale/neutral
        brightness_threshold_low = 50  # Very dark
        brightness_threshold_high = 200  # Very bright
        
        # Check for grayscale/neutral colors
        if s < saturation_threshold:
            if v < brightness_threshold_low:
                return "Lainnya (Hitam/Gelap)"
            elif v > brightness_threshold_high:
                return "Lainnya (Putih/Terang)"
            else:
                return "Lainnya (Abu-abu)"
        
        # Define hue ranges for color categories (in HSV, hue is 0-179 in OpenCV)
        # Primary colors: Red, Blue, Yellow (Green is also primary but less common in art)
        if (h >= 0 and h <= 10) or (h >= 170 and h <= 179):  # Red
            return "Primer (Merah)"
        elif h >= 100 and h <= 130:  # Blue
            return "Primer (Biru)"
        elif h >= 20 and h <= 35:  # Yellow
            return "Primer (Kuning)"
        elif h >= 40 and h <= 80:  # Green
            return "Primer (Hijau)"
        
        # Secondary colors: Orange, Purple, Green (mixing of primaries)
        elif h >= 11 and h <= 19:  # Orange (Red + Yellow)
            return "Sekunder (Oranye)"
        elif h >= 131 and h <= 169:  # Purple/Magenta (Red + Blue)
            return "Sekunder (Ungu)"
        elif h >= 81 and h <= 99:  # Cyan (Blue + Green, though green is primary)
            return "Sekunder (Cyan)"
        
        # Tertiary colors (mixing of primary and secondary)
        elif h >= 36 and h <= 39:  # Yellow-Green
            return "Tersier (Kuning-Hijau)"
        
        # Other colors that don't fit clearly into above categories
        else:
            return "Lainnya"

    def get_color_category_distribution(self):
        """
        Analyzes the distribution of color categories in the extracted palette.
        
        Returns:
        dict: Distribution of color categories with percentages
        """
        if self.dominant_colors is None:
            raise ValueError("No dominant colors extracted. Call extract_color_palette() first.")
        
        category_data = {}
        category_percentages = {}
        
        # Classify each dominant color
        for i, (color, percentage) in enumerate(zip(self.dominant_colors['colors'], 
                                                  self.dominant_colors['percentages'])):
            category = self.classify_color_category(color)
            
            if category not in category_data:
                category_data[category] = []
                category_percentages[category] = 0
            
            category_data[category].append({
                'color': color,
                'percentage': percentage,
                'hex': '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
            })
            category_percentages[category] += percentage
        
        # Sort categories by total percentage
        sorted_categories = sorted(category_percentages.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'categories': dict(sorted_categories),
            'detailed_data': category_data
        }

    def get_rgb_statistics(self):
        """
        Calculates RGB statistics for the extracted color palette.
        
        Returns:
        pandas.DataFrame: DataFrame containing RGB statistics
        """
        if self.rgb_palette is None:
            raise ValueError("No RGB palette extracted. Call extract_color_palette() first.")
        
        # Get original image as RGB array
        if hasattr(self.image, 'convert'):
            img_array = np.array(self.image.convert('RGB'))
        elif hasattr(self.image, 'shape'):
            img_array = self.image
        else:
            img_array = np.array(self.image)
        
        # Calculate statistics for the entire image
        img_flat = img_array.reshape(-1, 3)
        
        # Calculate overall image statistics
        overall_stats = {
            'Metrik': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
            'Red': [
                np.mean(img_flat[:, 0]),
                np.median(img_flat[:, 0]),
                np.std(img_flat[:, 0]),
                np.min(img_flat[:, 0]),
                np.max(img_flat[:, 0]),
                np.max(img_flat[:, 0]) - np.min(img_flat[:, 0])
            ],
            'Green': [
                np.mean(img_flat[:, 1]),
                np.median(img_flat[:, 1]),
                np.std(img_flat[:, 1]),
                np.min(img_flat[:, 1]),
                np.max(img_flat[:, 1]),
                np.max(img_flat[:, 1]) - np.min(img_flat[:, 1])
            ],
            'Blue': [
                np.mean(img_flat[:, 2]),
                np.median(img_flat[:, 2]),
                np.std(img_flat[:, 2]),
                np.min(img_flat[:, 2]),
                np.max(img_flat[:, 2]),
                np.max(img_flat[:, 2]) - np.min(img_flat[:, 2])
            ]
        }
        
        # Round values to 2 decimal places
        for key in ['Red', 'Green', 'Blue']:
            overall_stats[key] = [round(val, 2) for val in overall_stats[key]]
        
        overall_df = pd.DataFrame(overall_stats)
        
        # Calculate statistics for each color in the palette
        palette_stats = []
        hex_codes = self.get_hex_palette()
        
        for i, (color, percentage) in enumerate(zip(self.rgb_palette, self.color_percentages)):
            r, g, b = color
            palette_stats.append({
                'Warna': f'Warna {i+1}',
                'Hex': hex_codes[i],
                'R': int(r),
                'G': int(g),
                'B': int(b),
                'Persentase (%)': round(percentage, 2),
                'Brightness': round((0.299*r + 0.587*g + 0.114*b), 2),  # Perceived brightness
                'Kategori': self.classify_color_category(color)
            })
        
        palette_df = pd.DataFrame(palette_stats)
        
        return overall_df, palette_df

    def visualize_color_category_distribution(self):
        """
        Creates visualizations for color category distribution (pie chart and bar chart).
        
        Returns:
        tuple: (pie_fig, bar_fig) - Matplotlib figure objects
        """
        category_dist = self.get_color_category_distribution()
        categories = list(category_dist['categories'].keys())
        percentages = list(category_dist['categories'].values())
        
        # Define colors for each category
        category_colors = {
            'Primer (Merah)': '#FF0000',
            'Primer (Biru)': '#0000FF', 
            'Primer (Kuning)': '#FFFF00',
            'Primer (Hijau)': '#00FF00',
            'Sekunder (Oranye)': '#FFA500',
            'Sekunder (Ungu)': '#800080',
            'Sekunder (Cyan)': '#00FFFF',
            'Tersier (Kuning-Hijau)': '#ADFF2F',
            'Lainnya (Hitam/Gelap)': '#2F2F2F',
            'Lainnya (Putih/Terang)': '#F5F5F5',
            'Lainnya (Abu-abu)': '#808080',
            'Lainnya': '#D3D3D3'
        }
        
        colors_for_plot = [category_colors.get(cat, '#D3D3D3') for cat in categories]
        
        # Create pie chart
        pie_fig, pie_ax = plt.subplots(1, 1, figsize=(10, 8))
        
        wedges, texts, autotexts = pie_ax.pie(
            percentages,
            labels=categories,
            colors=colors_for_plot,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 9}
        )
        
        # Improve text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        pie_ax.set_title('Distribusi Kategori Warna', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Create bar chart
        bar_fig, bar_ax = plt.subplots(1, 1, figsize=(12, 6))
        
        bars = bar_ax.bar(range(len(categories)), percentages, color=colors_for_plot)
        
        # Add percentage labels on bars
        for i, (bar, percentage) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            bar_ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        bar_ax.set_xlabel('Kategori Warna', fontweight='bold')
        bar_ax.set_ylabel('Persentase (%)', fontweight='bold')
        bar_ax.set_title('Distribusi Persentase Kategori Warna', fontsize=14, fontweight='bold')
        bar_ax.set_xticks(range(len(categories)))
        bar_ax.set_xticklabels(categories, rotation=45, ha='right')
        bar_ax.set_ylim(0, max(percentages) * 1.1)
        
        # Add grid for better readability
        bar_ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        return pie_fig, bar_fig

    def get_hex_palette(self):
        """
        Converts the RGB palette to hexadecimal color codes.
        
        Returns:
        hex_codes: A list of hexadecimal color codes
        """
        if self.rgb_palette is None:
            raise ValueError("No RGB palette extracted. Call extract_color_palette() first.")
        
        hex_codes = []
        for rgb in self.rgb_palette:
            hex_code = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            hex_codes.append(hex_code)
        return hex_codes

    def get_dominant_hex_palette(self):
        """
        Converts the dominant colors to hexadecimal color codes.
        
        Returns:
        hex_codes: A list of hexadecimal color codes sorted by dominance
        """
        if self.dominant_colors is None:
            raise ValueError("No dominant colors extracted. Call extract_color_palette() first.")
        
        hex_codes = []
        for rgb in self.dominant_colors['colors']:
            hex_code = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            hex_codes.append(hex_code)
        return hex_codes

    def visualize_palette(self):
        """
        Creates a visualization of the extracted color palette.
        
        Returns:
        fig: Matplotlib figure object
        """
        if self.rgb_palette is None:
            raise ValueError("No RGB palette extracted. Call extract_color_palette() first.")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        
        # Create color bars
        palette_normalized = self.rgb_palette / 255.0
        colors = [palette_normalized[i] for i in range(len(palette_normalized))]
        
        # Create rectangles for each color
        for i, color in enumerate(colors):
            rect = plt.Rectangle((i, 0), 1, 1, facecolor=color)
            ax.add_patch(rect)
        
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(colors)) + 0.5)
        ax.set_xticklabels([f'Warna {i+1}' for i in range(len(colors))])
        ax.set_yticks([])
        ax.set_title('Palet Warna Citra yang Diambil')
        
        plt.tight_layout()
        return fig

    def visualize_dominant_palette(self):
        """
        Creates a visualization of the dominant color palette sorted by percentage.
        
        Returns:
        fig: Matplotlib figure object
        """
        if self.dominant_colors is None:
            raise ValueError("No dominant colors extracted. Call extract_color_palette() first.")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        
        # Create color bars with dominant colors
        dominant_colors_normalized = np.array(self.dominant_colors['colors']) / 255.0
        colors = [dominant_colors_normalized[i] for i in range(len(dominant_colors_normalized))]
        percentages = self.dominant_colors['percentages']
        
        # Create rectangles for each color with width proportional to percentage
        x_pos = 0
        total_width = 10
        
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            width = (percentage / 100) * total_width
            rect = plt.Rectangle((x_pos, 0), width, 1, facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            
            # Add percentage label
            if width > 0.5:  # Only show label if segment is wide enough
                ax.text(x_pos + width/2, 0.5, f'{percentage:.1f}%', 
                       ha='center', va='center', fontweight='bold', 
                       color='white' if np.mean(color) < 0.5 else 'black')
            
            x_pos += width
        
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Palet Warna Dominan (Diurutkan berdasarkan Persentase)', fontweight='bold')
        
        plt.tight_layout()
        return fig

    def visualize_color_distribution(self):
        """
        Creates a pie chart showing the distribution of colors in the image.
        
        Returns:
        fig: Matplotlib figure object
        """
        if self.rgb_palette is None or self.color_percentages is None:
            raise ValueError("No color palette or percentages extracted. Call extract_color_palette() first.")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Normalize RGB values for matplotlib colors
        palette_normalized = self.rgb_palette / 255.0
        colors = [palette_normalized[i] for i in range(len(palette_normalized))]
        
        # Create labels with percentages
        labels = [f'Warna {i+1}\n({self.color_percentages[i]:.1f}%)' 
                 for i in range(len(self.color_percentages))]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            self.color_percentages, 
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        # Improve text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Distribusi Warna dalam Citra', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig

    def get_dominant_color_info(self, top_n=3):
        """
        Returns information about the top N dominant colors.
        
        Args:
        top_n: Number of top dominant colors to return
        
        Returns:
        dict: Information about dominant colors
        """
        if self.dominant_colors is None:
            raise ValueError("No dominant colors extracted. Call extract_color_palette() first.")
        
        top_colors = []
        hex_codes = self.get_dominant_hex_palette()
        
        for i in range(min(top_n, len(self.dominant_colors['colors']))):
            color_info = {
                'rank': i + 1,
                'rgb': self.dominant_colors['colors'][i].tolist(),
                'hex': hex_codes[i],
                'percentage': self.dominant_colors['percentages'][i]
            }
            top_colors.append(color_info)
        
        return top_colors

    def segment_image(self):
        """
        Segments the image based on the extracted color palette.
        
        Returns:
        segmented_image: Image with pixels replaced by nearest palette colors
        """
        if self.palette is None:
            raise ValueError("No palette extracted. Call extract_color_palette() first.")
        
        if self.processed_pixels is None:
            self.preprocess_image()
        
        # Get cluster labels for each pixel
        if self.cluster_labels is None:
            labels = self.kmeans.predict(self.processed_pixels)
        else:
            labels = self.cluster_labels
        
        # Create segmented image in LAB space
        segmented_lab = self.palette[labels]
        
        # Reshape back to original image dimensions
        if hasattr(self.image, 'shape'):
            original_shape = self.image.shape
        elif hasattr(self.image, 'size'):
            # For PIL images
            original_shape = (self.image.size[1], self.image.size[0], 3)
        else:
            original_shape = np.array(self.image).shape
            
        segmented_lab = segmented_lab.reshape(original_shape)
        
        # Convert back to RGB
        segmented_rgb = cv.cvtColor(np.uint8(segmented_lab), cv.COLOR_LAB2RGB)
        
        return segmented_rgb

    def export_palette_data(self, format='csv'):
        """
        Exports palette data in CSV format.
        
        Args:
        format: Export format (only 'csv' is supported)
        
        Returns:
        data: Formatted palette data as CSV string
        """
        if self.rgb_palette is None:
            raise ValueError("No RGB palette extracted. Call extract_color_palette() first.")
        
        if format != 'csv':
            raise ValueError("Only CSV format is supported.")
        
        hex_codes = self.get_hex_palette()
        
        import io
        output = io.StringIO()
        
        if self.color_percentages is not None:
            output.write('Color_ID,R,G,B,Hex,Percentage,Dominance_Rank\n')
            for i, (rgb, hex_code, percentage) in enumerate(zip(self.rgb_palette, hex_codes, self.color_percentages)):
                # Find dominance rank
                dominance_rank = 'N/A'
                if self.dominant_colors is not None:
                    for j, orig_idx in enumerate(self.dominant_colors['original_indices']):
                        if orig_idx == i:
                            dominance_rank = j + 1
                            break
                output.write(f'{i+1},{rgb[0]},{rgb[1]},{rgb[2]},{hex_code},{percentage:.2f}%,{dominance_rank}\n')
        else:
            output.write('Color_ID,R,G,B,Hex\n')
            for i, (rgb, hex_code) in enumerate(zip(self.rgb_palette, hex_codes)):
                output.write(f'{i+1},{rgb[0]},{rgb[1]},{rgb[2]},{hex_code}\n')
        
        return output.getvalue()

def create_color_segmentation_ui(uploaded_image):
    """
    Creates Streamlit UI for color segmentation functionality.
    
    Args:
    uploaded_image: Uploaded image file from Streamlit
    """
    if uploaded_image is not None:
        try:
            # Convert uploaded file to PIL Image
            image = Image.open(uploaded_image)
            
            # Select number of colors
            num_colors = st.slider(
                "Pilih jumlah warna dalam palet citra (4-15):", 
                min_value=4, 
                max_value=15, 
                value=7, 
                step=1
            )
            st.write(f"Palet warna citra Anda saat ini memiliki {num_colors} warna")
            
            with st.spinner("Processing color segmentation..."):
                # Create processor instance
                processor = ColorSegmentationProcessor(image, num_colors)
                
                # Extract color palette
                rgb_palette = processor.extract_color_palette()
                
                # Create two columns for display
                col1, col2 = st.columns(2)
                
                with col1:
                    # Title for original image
                    st.subheader("Citra Asli")
                    # Display original image
                    st.image(image, use_container_width=True)
                
                with col2:
                    # Title for segmented image
                    st.subheader("Citra Tersegmentasi")
                    # Display segmented image
                    segmented_img = processor.segment_image()
                    st.image(segmented_img, use_container_width=True)
                
                # Visualize color palette
                st.subheader("Palet Warna Citra")
                palette_fig = processor.visualize_palette()
                st.pyplot(fig=palette_fig, use_container_width=True)
                
                # Visualize dominant color palette
                st.subheader("Palet Warna Dominan")
                dominant_palette_fig = processor.visualize_dominant_palette()
                st.pyplot(fig=dominant_palette_fig, use_container_width=True)
                
                # Display dominant color information
                st.subheader("Informasi Warna Dominan")
                dominant_info = processor.get_dominant_color_info(top_n=5)
                
                # Create columns for dominant colors
                cols = st.columns(min(5, len(dominant_info)))
                for i, color_info in enumerate(dominant_info):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="background-color: {color_info['hex']}; 
                                       width: 80px; height: 80px; 
                                       margin: 0 auto; 
                                       border-radius: 50%; 
                                       border: 2px solid #ddd;"></div>
                            <p><strong>#{i+1}</strong></p>
                            <p>{color_info['hex']}</p>
                            <p>{color_info['percentage']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Create two columns for distribution charts
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Add pie chart for color distribution
                    st.subheader("Distribusi Warna")
                    distribution_fig = processor.visualize_color_distribution()
                    st.pyplot(fig=distribution_fig, use_container_width=True)
                
                with dist_col2:
                    # Add pie chart for color category distribution
                    st.subheader("Distribusi Kategori Warna")
                    category_pie_fig, category_bar_fig = processor.visualize_color_category_distribution()
                    st.pyplot(fig=category_pie_fig, use_container_width=True)
                
                # Add bar chart for color category percentages
                st.subheader("Persentase Kategori Warna")
                st.pyplot(fig=category_bar_fig, use_container_width=True)
                
                # Add RGB Statistics Tables
                st.subheader("Statistik Fitur Warna RGB")
                
                # Get RGB statistics
                overall_stats_df, palette_stats_df = processor.get_rgb_statistics()
                
                # Create two columns for the statistics tables
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.write("**Statistik RGB Keseluruhan Citra:**")
                    st.dataframe(overall_stats_df, use_container_width=True)
                
                with stats_col2:
                    st.write("**Statistik RGB Palet Warna:**")
                    # Add color preview column to the dataframe display
                    st.dataframe(palette_stats_df, use_container_width=True)
                
                # Display detailed category information
                st.subheader("Detail Kategori Warna")
                category_dist = processor.get_color_category_distribution()
                
                for category, percentage in category_dist['categories'].items():
                    with st.expander(f"{category} ({percentage:.1f}%)"):
                        colors_in_category = category_dist['detailed_data'][category]
                        
                        # Create columns for colors in this category
                        if colors_in_category:
                            color_cols = st.columns(min(len(colors_in_category), 5))
                            for i, color_data in enumerate(colors_in_category):
                                if i < len(color_cols):
                                    with color_cols[i]:
                                        st.markdown(f"""
                                        <div style="text-align: center;">
                                            <div style="background-color: {color_data['hex']}; 
                                                       width: 60px; height: 60px; 
                                                       margin: 0 auto; 
                                                       border-radius: 50%; 
                                                       border: 2px solid #ddd;"></div>
                                            <p><small>{color_data['hex']}</small></p>
                                            <p><small>{color_data['percentage']:.1f}%</small></p>
                                        </div>
                                        """, unsafe_allow_html=True)
                
                # Export options (CSV only)
                st.subheader("Export Data")
                
                # Create columns for different export options
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    palette_csv = processor.export_palette_data('csv')
                    st.download_button(
                        label="Export Palet Warna (CSV)",
                        data=palette_csv,
                        file_name="color_palette.csv",
                        icon=":material/download:",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with export_col2:
                    # Export RGB statistics
                    stats_csv = palette_stats_df.to_csv(index=False)
                    st.download_button(
                        label="Export Statistik RGB (CSV)",
                        data=stats_csv,
                        file_name="rgb_statistics.csv",
                        icon=":material/download:",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Please make sure you uploaded a valid image file (JPG, JPEG, or PNG)")