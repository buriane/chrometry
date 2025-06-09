import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2 as cv
import pandas as pd
from typing import List, Dict, Tuple
import tempfile
import os
from datetime import datetime
import io
from PIL import Image

class PDFReportGenerator:
    def __init__(self):
        self.temp_files = []
    
    def create_analysis_report(
        self, 
        citra_asli: np.ndarray,
        citra_threshold: np.ndarray,
        citra_base: np.ndarray,
        citra_proses_rgb: np.ndarray,
        kontur: List[np.ndarray],
        faktor_skala: float,
        shape_2d: str,
        shape_3d: str,
        confidence: float,
        properties: Dict,
        parameter_terbaik: Tuple,
        color_data: Dict = None
    ) -> str:
        """
        Membuat laporan PDF lengkap dari hasil analisis citra.
        """
        # Buat file PDF sementara
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf.close()
        
        with PdfPages(temp_pdf.name) as pdf:
            # Halaman 1: Cover dan Ringkasan
            self._create_cover_page(pdf, shape_2d, shape_3d, confidence, properties, color_data)
            
            # Halaman 2: Visualisasi Deteksi Kontur
            self._create_contour_analysis_page(
                pdf, citra_asli, citra_threshold, citra_base, 
                citra_proses_rgb, kontur, faktor_skala
            )
            
            # Halaman 3: Detail Properti Geometri
            self._create_properties_page(pdf, properties, parameter_terbaik, kontur)
            
            # Halaman 4: Segmentasi Warna (jika ada data)
            if color_data:
                self._create_color_analysis_page(pdf, color_data)
                
                # Halaman 5: Detail Statistik Warna (jika ada data)
                self._create_color_statistics_page(pdf, color_data)
                
                # Halaman 6: Statistik RGB Keseluruhan Citra
                self._create_overall_rgb_statistics_page(pdf, color_data)
        
        return temp_pdf.name
    
    def _create_cover_page(self, pdf, shape_2d, shape_3d, confidence, properties, color_data):
        """Membuat halaman cover dengan ringkasan hasil."""
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')
        
        # Header
        ax.text(0.5, 0.95, 'LAPORAN ANALISIS CITRA DIGITAL', 
                ha='center', va='top', fontsize=20, fontweight='bold',
                transform=ax.transAxes)
        
        ax.text(0.5, 0.9, f'Tanggal: {datetime.now().strftime("%d %B %Y, %H:%M:%S")}', 
                ha='center', va='top', fontsize=12,
                transform=ax.transAxes)
        
        # Ringkasan Hasil Analisis Geometri
        y_pos = 0.8
        ax.text(0.1, y_pos, 'RINGKASAN HASIL ANALISIS GEOMETRI', 
                fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        y_pos -= 0.08
        contour_points = len(properties.get('contour_points', [])) if isinstance(properties.get('contour_points'), (list, np.ndarray)) and len(properties.get('contour_points', [])) > 0 else 0
        
        results = [
            f"Bentuk 2D Terdeteksi: {shape_2d}",
            f"Bentuk Ruang: {shape_3d}",
            f"Tingkat Kepercayaan: {confidence:.2%}",
            f"Luas: {properties['area']:.0f} px²",
            f"Keliling: {properties['perimeter']:.2f} px",
            f"Jumlah Titik Kontur: {contour_points}",
            f"Rasio Lebar/Tinggi: {properties['aspect_ratio']:.2f}",
            f"Circularity: {properties['circularity']:.3f}",
            f"Solidity: {properties['solidity']:.3f}",
            f"Extent: {properties['extent']:.3f}"
        ]
        
        for result in results:
            ax.text(0.1, y_pos, f"• {result}", fontsize=12, transform=ax.transAxes)
            y_pos -= 0.04
        
        # Ringkasan Hasil Analisis Warna
        if color_data:
            y_pos -= 0.04
            ax.text(0.1, y_pos, 'RINGKASAN HASIL ANALISIS WARNA', 
                    fontsize=16, fontweight='bold', transform=ax.transAxes)
            
            y_pos -= 0.06
            
            # Informasi jumlah warna
            num_colors = color_data.get('num_colors', 'N/A')
            ax.text(0.1, y_pos, f"• Jumlah Warna dalam Palet: {num_colors}", 
                   fontsize=12, transform=ax.transAxes)
            y_pos -= 0.04
            
            # Warna dominan
            if 'dominant_colors' in color_data and color_data['dominant_colors']:
                top_colors = color_data['dominant_colors'][:3]  # Top 3 colors
                ax.text(0.1, y_pos, "• 3 Warna Dominan Teratas:", 
                       fontsize=12, transform=ax.transAxes)
                y_pos -= 0.04
                
                for i, color_info in enumerate(top_colors):
                    color_text = f"  {i+1}. {color_info['hex']} ({color_info['percentage']:.1f}%)"
                    ax.text(0.1, y_pos, color_text, fontsize=11, transform=ax.transAxes)
                    y_pos -= 0.03
            
            # Kategori warna
            if 'category_distribution' in color_data:
                y_pos -= 0.02
                ax.text(0.1, y_pos, "• Kategori Warna Utama:", 
                       fontsize=12, transform=ax.transAxes)
                y_pos -= 0.04
                
                categories = color_data['category_distribution']['categories']
                top_categories = list(categories.items())[:3]  # Top 3 categories
                
                for category, percentage in top_categories:
                    cat_text = f"  - {category}: {percentage:.1f}%"
                    ax.text(0.1, y_pos, cat_text, fontsize=11, transform=ax.transAxes)
                    y_pos -= 0.03
        
        # Interpretasi
        y_pos -= 0.04
        ax.text(0.1, y_pos, 'INTERPRETASI HASIL', 
                fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        y_pos -= 0.06
        interpretations = self._get_shape_interpretation(shape_2d, properties)
        
        for interpretation in interpretations:
            ax.text(0.1, y_pos, f"• {interpretation}", fontsize=11, 
                   transform=ax.transAxes, wrap=True)
            y_pos -= 0.04
        
        # Interpretasi warna jika ada
        if color_data:
            color_interpretations = self._get_color_interpretation(color_data)
            for interpretation in color_interpretations:
                ax.text(0.1, y_pos, f"• {interpretation}", fontsize=11, 
                       transform=ax.transAxes, wrap=True)
                y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_contour_analysis_page(self, pdf, citra_asli, citra_threshold, 
                                    citra_base, citra_proses_rgb, kontur, faktor_skala):
        """Membuat halaman visualisasi analisis kontur."""
        fig = plt.figure(figsize=(8.27, 11.69))
        
        # Judul halaman
        fig.suptitle('VISUALISASI DETEKSI KONTUR', fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot 1: Citra Asli
        citra_asli_diubah = self._resize_image(citra_asli, faktor_skala)
        plt.subplot(3, 2, 1)
        plt.imshow(citra_asli_diubah, cmap='gray')
        plt.title('Citra Asli', fontweight='bold')
        plt.axis('off')
        
        # Subplot 2: Threshold
        plt.subplot(3, 2, 2)
        plt.imshow(citra_threshold, cmap='gray')
        plt.title('Hasil Threshold', fontweight='bold')
        plt.axis('off')
        
        # Subplot 3: Kontur (Garis Hijau)
        plt.subplot(3, 2, 3)
        plt.imshow(citra_proses_rgb)
        plt.title('Kontur Terdeteksi', fontweight='bold')
        plt.axis('off')
        
        if kontur:
            # Subplot 4: Titik Kontur
            citra_kontur_detail = citra_base.copy()
            citra_kontur_detail_rgb = cv.cvtColor(citra_kontur_detail, cv.COLOR_GRAY2RGB)
            for point in kontur[0]:
                x, y = point[0]
                cv.circle(citra_kontur_detail_rgb, (x, y), 2, (0, 0, 255), -1)
            plt.subplot(3, 2, 4)
            plt.imshow(citra_kontur_detail_rgb)
            plt.title(f'Titik Kontur ({len(kontur[0])} titik)', fontweight='bold')
            plt.axis('off')
            
            # Subplot 5: Convex Hull
            momen = cv.moments(kontur[0])
            if momen['m00'] != 0:
                pusat_x = int(momen['m10'] / momen['m00'])
                pusat_y = int(momen['m01'] / momen['m00'])
                citra_convex_hull = cv.cvtColor(citra_base.copy(), cv.COLOR_GRAY2RGB)
                
                hull = cv.convexHull(kontur[0])
                cv.polylines(citra_convex_hull, [hull], True, (255, 0, 0), 4)
                cv.circle(citra_convex_hull, (pusat_x, pusat_y), 5, (0, 255, 0), -1)
                
                plt.subplot(3, 2, 5)
                plt.imshow(citra_convex_hull)
                plt.title(f'Convex Hull ({len(hull)} titik)', fontweight='bold')
                plt.axis('off')
            
            # Subplot 6: Bounding Rectangle
            x, y, lebar, tinggi = cv.boundingRect(kontur[0])
            citra_dengan_kotak = citra_base.copy()
            cv.rectangle(citra_dengan_kotak, (x, y), (x + lebar, y + tinggi), (255), 2)
            plt.subplot(3, 2, 6)
            plt.imshow(citra_dengan_kotak, cmap='gray')
            plt.title('Bounding Rectangle', fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_properties_page(self, pdf, properties, parameter_terbaik, kontur):
        """Membuat halaman detail properti geometri."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.27, 11.69))
        fig.suptitle('DETAIL PROPERTI GEOMETRI', fontsize=16, fontweight='bold')
        
        # Tabel properti dasar
        ax1.axis('off')
        ax1.set_title('Properti Dasar', fontweight='bold', pad=20)
        
        basic_props = [
            ['Properti', 'Nilai'],
            ['Luas', f"{properties['area']:.2f} px²"],
            ['Keliling', f"{properties['perimeter']:.2f} px"],
            ['Lebar', f"{properties['width']} px"],
            ['Tinggi', f"{properties['height']} px"],
            ['Pusat X', f"{properties['center_x']} px"],
            ['Pusat Y', f"{properties['center_y']} px"],
            ['Rasio Aspek', f"{properties['aspect_ratio']:.3f}"]
        ]
        
        table1 = ax1.table(cellText=basic_props[1:], colLabels=basic_props[0],
                          cellLoc='center', loc='center', colWidths=[0.5, 0.5])
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)
        
        # Tabel properti lanjutan
        ax2.axis('off')
        ax2.set_title('Properti Lanjutan', fontweight='bold', pad=20)
        
        advanced_props = [
            ['Properti', 'Nilai', 'Interpretasi'],
            ['Circularity', f"{properties['circularity']:.3f}", self._interpret_circularity(properties['circularity'])],
            ['Solidity', f"{properties['solidity']:.3f}", self._interpret_solidity(properties['solidity'])],
            ['Extent', f"{properties['extent']:.3f}", self._interpret_extent(properties['extent'])]
        ]
        
        table2 = ax2.table(cellText=advanced_props[1:], colLabels=advanced_props[0],
                          cellLoc='center', loc='center', colWidths=[0.3, 0.2, 0.5])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 2)
        
        # Radar chart untuk properti bentuk
        ax3.set_title('Grafik Radar Properti Bentuk', fontweight='bold')
        categories = ['Circularity', 'Solidity', 'Extent']
        values = [properties['circularity'], properties['solidity'], properties['extent']]
        
        # Buat radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values_plot = values + [values[0]]  # Close the plot
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        ax3.plot(angles_plot, values_plot, 'b-', linewidth=2)
        ax3.fill(angles_plot, values_plot, alpha=0.25)
        ax3.set_xticks(angles)
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title('Grafik Radar Properti Bentuk', fontweight='bold', pad=20)
        
        # Parameter yang digunakan
        ax4.axis('off')
        ax4.set_title('Parameter Deteksi Kontur', fontweight='bold', pad=20)
        
        mode_adaptif, nilai_threshold, ukuran_kernel, invert_threshold = parameter_terbaik
        param_info = [
            ['Parameter', 'Nilai'],
            ['Mode Threshold', 'Adaptif' if mode_adaptif else 'Global'],
            ['Nilai Threshold', str(nilai_threshold)],
            ['Ukuran Kernel', f"{ukuran_kernel}x{ukuran_kernel}"],
            ['Inversi Threshold', 'Ya' if invert_threshold else 'Tidak'],
            ['Jumlah Titik Kontur', str(len(kontur[0]) if kontur else 0)]
        ]
        
        table3 = ax4.table(cellText=param_info[1:], colLabels=param_info[0],
                          cellLoc='center', loc='center', colWidths=[0.6, 0.4])
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 2)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_color_analysis_page(self, pdf, color_data):
        """Membuat halaman analisis segmentasi warna."""
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle('ANALISIS SEGMENTASI WARNA', fontsize=16, fontweight='bold', y=0.95)
        
        try:
            # Subplot 1: Citra Asli vs Citra Tersegmentasi
            plt.subplot(2, 2, 1)
            if 'original_image' in color_data:
                plt.imshow(color_data['original_image'])
                plt.title('Citra Asli', fontweight='bold')
                plt.axis('off')
            
            plt.subplot(2, 2, 2)
            if 'segmented_image' in color_data:
                plt.imshow(color_data['segmented_image'])
                plt.title('Citra Tersegmentasi', fontweight='bold')
                plt.axis('off')
            
            # Subplot 3: Palet Warna Dominan
            plt.subplot(2, 1, 2)
            if 'dominant_colors' in color_data and color_data['dominant_colors']:
                self._plot_color_palette(color_data['dominant_colors'], 'Palet Warna Dominan')
            
            plt.tight_layout()
            
        except Exception as e:
            # Fallback jika ada error
            plt.clf()
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Analisis Segmentasi Warna\n\nJumlah Warna: {color_data.get("num_colors", "N/A")}\n\nData tersedia untuk analisis lebih lanjut.', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_color_statistics_page(self, pdf, color_data):
        """Membuat halaman detail statistik warna."""
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle('STATISTIK DAN DISTRIBUSI WARNA', fontsize=16, fontweight='bold', y=0.95)
        
        try:
            # Subplot 1: Distribusi Kategori Warna (Pie Chart)
            if 'category_distribution' in color_data:
                plt.subplot(2, 2, 1)
                categories = list(color_data['category_distribution']['categories'].keys())
                percentages = list(color_data['category_distribution']['categories'].values())
                
                # Define colors for categories
                category_colors = self._get_category_colors()
                colors_for_plot = [category_colors.get(cat, '#D3D3D3') for cat in categories]
                
                plt.pie(percentages, labels=categories, colors=colors_for_plot, 
                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
                plt.title('Distribusi Kategori Warna', fontweight='bold', fontsize=10)
            
            # Subplot 2: Distribusi Warna Individual
            if 'dominant_colors' in color_data and color_data['dominant_colors']:
                plt.subplot(2, 2, 2)
                colors = [f"#{int(c['rgb'][0]):02x}{int(c['rgb'][1]):02x}{int(c['rgb'][2]):02x}" 
                         for c in color_data['dominant_colors']]
                percentages = [c['percentage'] for c in color_data['dominant_colors']]
                
                # Normalize colors for matplotlib
                rgb_colors = [[c['rgb'][0]/255, c['rgb'][1]/255, c['rgb'][2]/255] 
                             for c in color_data['dominant_colors']]
                
                plt.pie(percentages, colors=rgb_colors, autopct='%1.1f%%', 
                       startangle=90, textprops={'fontsize': 8})
                plt.title('Distribusi Warna Individual', fontweight='bold', fontsize=10)
            
            # Subplot 3: Tabel Statistik RGB
            plt.subplot(2, 1, 2)
            plt.axis('off')
            
            if 'rgb_statistics' in color_data:
                stats_data = color_data['rgb_statistics']
                
                # Create table for top colors
                table_data = []
                headers = ['Warna', 'Hex', 'R', 'G', 'B', 'Persentase', 'Kategori']
                
                for i, color_info in enumerate(color_data['dominant_colors'][:6]):  # Top 6 colors
                    rgb = color_info['rgb']
                    row = [
                        f'Warna {i+1}',
                        f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}",
                        f"{int(rgb[0])}",
                        f"{int(rgb[1])}",
                        f"{int(rgb[2])}",
                        f"{color_info['percentage']:.1f}%",
                        color_info.get('category', 'N/A')
                    ]
                    table_data.append(row)
                
                # Create table
                table = plt.table(cellText=table_data, colLabels=headers,
                                cellLoc='center', loc='center',
                                colWidths=[0.12, 0.15, 0.08, 0.08, 0.08, 0.12, 0.25])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
                
                plt.title('Detail Warna Dominan', fontweight='bold', y=0.8)
            
            plt.tight_layout()
            
        except Exception as e:
            # Fallback jika ada error
            plt.clf()
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Statistik Warna\n\nData statistik warna tersedia\nuntuk analisis lebih lanjut.\n\nError: {str(e)}', 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_overall_rgb_statistics_page(self, pdf, color_data):
        """Membuat halaman khusus untuk Statistik RGB Keseluruhan Citra dalam bentuk tabel lengkap."""
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        fig.suptitle('STATISTIK RGB KESELURUHAN CITRA', fontsize=20, fontweight='bold', y=0.95)
        
        ax.axis('off')
        
        try:
            if 'rgb_statistics' in color_data and 'overall' in color_data['rgb_statistics']:
                overall_stats = color_data['rgb_statistics']['overall']
                
                # Konversi dari dictionary format ke list format untuk tabel
                metrics = overall_stats.get('Metrik', ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'])
                red_values = overall_stats.get('Red', [0] * 6)
                green_values = overall_stats.get('Green', [0] * 6)
                blue_values = overall_stats.get('Blue', [0] * 6)
                
                # Prepare table data dengan semua 6 metrik
                table_data = []
                headers = ['Metrik Statistik', 'Red (R)', 'Green (G)', 'Blue (B)']
                
                # Pastikan semua 6 metrik ditampilkan
                metric_labels = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range']
                
                for i, metric in enumerate(metric_labels):
                    if i < len(red_values) and i < len(green_values) and i < len(blue_values):
                        row = [
                            metric,
                            f"{red_values[i]:.2f}",
                            f"{green_values[i]:.2f}",
                            f"{blue_values[i]:.2f}"
                        ]
                        table_data.append(row)
                
                # Create main table yang besar dan terpusat
                table = ax.table(cellText=table_data, colLabels=headers,
                                cellLoc='center', loc='center',
                                colWidths=[0.3, 0.23, 0.23, 0.23])
                
                # Styling tabel
                table.auto_set_font_size(False)
                table.set_fontsize(14)  # Font size lebih besar
                table.scale(1.2, 3)  # Scale lebih besar untuk visibilitas
                
                # Style header row dengan warna hijau
                for i in range(len(headers)):
                    cell = table[(0, i)]
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white', fontsize=16)
                    cell.set_height(0.08)
                
                # Style data rows dengan alternate colors
                for i in range(1, len(table_data) + 1):
                    # Warna berbeda untuk setiap baris
                    if i % 2 == 0:
                        row_color = '#f0f0f0'  # Abu-abu terang
                    else:
                        row_color = 'white'    # Putih
                    
                    for j in range(len(headers)):
                        cell = table[(i, j)]
                        cell.set_facecolor(row_color)
                        cell.set_height(0.06)
                        
                        # Bold untuk kolom metrik
                        if j == 0:
                            cell.set_text_props(weight='bold', fontsize=14)
                        else:
                            cell.set_text_props(fontsize=14)
                
                # Tambahkan border yang lebih tebal
                for i in range(len(table_data) + 1):
                    for j in range(len(headers)):
                        cell = table[(i, j)]
                        cell.set_edgecolor('black')
                        cell.set_linewidth(1.5)
                
                # Judul tabel
                ax.text(0.5, 0.85, 'Tabel Statistik RGB Keseluruhan Citra', 
                       ha='center', va='center', fontsize=18, fontweight='bold',
                       transform=ax.transAxes)
                
                # Keterangan di bawah tabel
                ax.text(0.5, 0.15, 
                       'Keterangan:\n' +
                       '• Mean: Nilai rata-rata intensitas warna\n' +
                       '• Median: Nilai tengah dari distribusi intensitas\n' +
                       '• Std Dev: Standar deviasi (variabilitas warna)\n' +
                       '• Min: Nilai intensitas minimum\n' +
                       '• Max: Nilai intensitas maksimum\n' +
                       '• Range: Rentang nilai (Max - Min)',
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
                
            else:
                # Fallback jika data tidak tersedia
                ax.text(0.5, 0.5, 'Data Statistik RGB Keseluruhan Citra\nTidak Tersedia', 
                        ha='center', va='center', fontsize=18, fontweight='bold',
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
            
        except Exception as e:
            # Error handling
            ax.text(0.5, 0.5, f'Error dalam pembuatan statistik RGB:\n\n{str(e)}', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_color_palette(self, dominant_colors, title):
        """Helper untuk plot palet warna."""
        # Create color bars dengan proporsi berdasarkan persentase
        x_pos = 0
        total_width = 10
        
        for color_info in dominant_colors:
            rgb = color_info['rgb']
            percentage = color_info['percentage']
            
            # Normalize RGB untuk matplotlib
            color_normalized = [rgb[0]/255, rgb[1]/255, rgb[2]/255]
            width = (percentage / 100) * total_width
            
            rect = plt.Rectangle((x_pos, 0), width, 1, 
                               facecolor=color_normalized, 
                               edgecolor='white', linewidth=1)
            plt.gca().add_patch(rect)
            
            # Add percentage label jika segment cukup lebar
            if width > 0.8:
                plt.text(x_pos + width/2, 0.5, f'{percentage:.1f}%', 
                        ha='center', va='center', fontweight='bold', 
                        color='white' if np.mean(color_normalized) < 0.5 else 'black',
                        fontsize=9)
            
            x_pos += width
        
        plt.xlim(0, total_width)
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(title, fontweight='bold')
    
    def _get_category_colors(self):
        """Warna untuk kategori warna."""
        return {
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
    
    def _resize_image(self, image, scale_factor):
        """Helper untuk mengubah ukuran gambar."""
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        return cv.resize(image, (new_width, new_height))
    
    def _get_shape_interpretation(self, shape_2d, properties):
        """Memberikan interpretasi berdasarkan bentuk yang terdeteksi."""
        interpretations = []
        
        if shape_2d == "Lingkaran":
            interpretations.append("Objek memiliki bentuk mendekati lingkaran sempurna")
            if properties['circularity'] > 0.8:
                interpretations.append("Nilai circularity tinggi menunjukkan bentuk sangat bulat")
        elif shape_2d == "Persegi":
            interpretations.append("Objek memiliki bentuk persegi atau persegi panjang")
            if properties['aspect_ratio'] < 1.2:
                interpretations.append("Rasio aspek mendekati 1:1 menunjukkan bentuk persegi")
        elif shape_2d == "Segitiga":
            interpretations.append("Objek memiliki bentuk segitiga")
            
        if properties['solidity'] > 0.9:
            interpretations.append("Bentuk objek sangat solid (sedikit cekungan)")
        elif properties['solidity'] < 0.7:
            interpretations.append("Bentuk objek memiliki beberapa cekungan atau tidak beraturan")
            
        return interpretations
    
    def _get_color_interpretation(self, color_data):
        """Memberikan interpretasi berdasarkan analisis warna."""
        interpretations = []
        
        if 'num_colors' in color_data:
            num_colors = color_data['num_colors']
            if num_colors <= 5:
                interpretations.append("Citra memiliki palet warna sederhana dengan variasi terbatas")
            elif num_colors <= 8:
                interpretations.append("Citra memiliki palet warna yang seimbang")
            else:
                interpretations.append("Citra memiliki palet warna yang kaya dan beragam")
        
        if 'dominant_colors' in color_data and color_data['dominant_colors']:
            top_color = color_data['dominant_colors'][0]
            if top_color['percentage'] > 50:
                interpretations.append(f"Warna dominan ({top_color['hex']}) mendominasi lebih dari setengah citra")
            elif top_color['percentage'] > 30:
                interpretations.append(f"Warna dominan ({top_color['hex']}) memiliki pengaruh kuat dalam citra")
        
        if 'category_distribution' in color_data:
            categories = color_data['category_distribution']['categories']
            if 'Primer' in str(categories):
                interpretations.append("Citra didominasi oleh warna-warna primer")
            if 'Lainnya' in str(categories):
                interpretations.append("Citra mengandung warna netral seperti hitam, putih, atau abu-abu")
        
        return interpretations
    
    def _interpret_circularity(self, value):
        """Interpretasi nilai circularity."""
        if value > 0.8:
            return "Sangat Bulat"
        elif value > 0.6:
            return "Cukup Bulat"
        elif value > 0.4:
            return "Agak Bulat"
        else:
            return "Tidak Bulat"
    
    def _interpret_solidity(self, value):
        """Interpretasi nilai solidity."""
        if value > 0.9:
            return "Sangat Solid"
        elif value > 0.8:
            return "Cukup Solid"
        elif value > 0.7:
            return "Agak Solid"
        else:
            return "Tidak Solid"
    
    def _interpret_extent(self, value):
        """Interpretasi nilai extent."""
        if value > 0.8:
            return "Mengisi Penuh"
        elif value > 0.6:
            return "Cukup Mengisi"
        elif value > 0.4:
            return "Sebagian Mengisi"
        else:
            return "Sedikit Mengisi"
    
    def cleanup_temp_files(self):
        """Membersihkan file-file sementara."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        self.temp_files.clear()