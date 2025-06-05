import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
from typing import List, Tuple, Dict

def muat_citra(lokasi_citra: str) -> np.ndarray:
    """
    Memuat citra dalam format grayscale dari lokasi yang ditentukan.
    
    Args:
        lokasi_citra (str): Path ke file citra.
        
    Returns:
        np.ndarray: citra dalam format grayscale.
        
    Raises:
        FileNotFoundError: Jika file citra tidak ditemukan.
        ValueError: Jika citra gagal dimuat.
    """
    if not os.path.exists(lokasi_citra):
        raise FileNotFoundError(f"File citra tidak ditemukan di: {lokasi_citra}")
    
    citra = cv.imread(lokasi_citra, cv.IMREAD_GRAYSCALE)
    if citra is None:
        raise ValueError(f"Gagal memuat citra dari: {lokasi_citra}")
    
    return citra

def ubah_ukuran_citra(citra: np.ndarray, faktor_skala: float) -> np.ndarray:
    """
    Mengubah ukuran citra berdasarkan faktor skala yang diberikan.
    
    Args:
        citra (np.ndarray): citra yang akan diubah ukurannya.
        faktor_skala (float): Faktor untuk mengubah dimensi citra.
        
    Returns:
        np.ndarray: citra yang sudah diubah ukurannya.
    """
    tinggi, lebar = citra.shape
    tinggi_baru = int(tinggi * faktor_skala)
    lebar_baru = int(lebar * faktor_skala)
    return cv.resize(citra, (lebar_baru, tinggi_baru))

def proses_kontur(citra: np.ndarray, 
                 mode_adaptif: bool = True,
                 nilai_threshold: int = 65, 
                 ukuran_kernel_dilasi: int = 3,
                 invert_threshold: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Memproses citra untuk mendeteksi dan mengekstrak kontur dengan parameter yang dapat disesuaikan.
    
    Args:
        citra (np.ndarray): citra dalam format grayscale.
        mode_adaptif (bool): Jika True, gunakan threshold adaptif, jika False gunakan threshold global.
        nilai_threshold (int): Nilai ambang batas untuk threshold global.
        ukuran_kernel_dilasi (int): Ukuran kernel untuk operasi dilasi.
        invert_threshold (bool): Jika True, inversi mode threshold.
        
    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: citra hasil threshold dan daftar kontur.
    """
    mean_value = np.mean(citra)
    is_light_object = mean_value > 127
    
    mode_threshold = cv.THRESH_BINARY if is_light_object else cv.THRESH_BINARY_INV
    if invert_threshold:
        mode_threshold = cv.THRESH_BINARY_INV if mode_threshold == cv.THRESH_BINARY else cv.THRESH_BINARY
    
    if mode_adaptif:
        citra_threshold = cv.adaptiveThreshold(
            citra, 
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            mode_threshold, 
            11, 
            2
        )
    else:
        _, citra_threshold = cv.threshold(citra, nilai_threshold, 255, mode_threshold)
    
    citra_threshold = cv.medianBlur(citra_threshold, 3)
    kernel = np.ones((ukuran_kernel_dilasi, ukuran_kernel_dilasi), np.uint8)
    citra_threshold = cv.morphologyEx(citra_threshold, cv.MORPH_OPEN, kernel)
    citra_threshold = cv.morphologyEx(citra_threshold, cv.MORPH_CLOSE, kernel)
    
    kontur, _ = cv.findContours(citra_threshold.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    
    kontur_terfilter = []
    if kontur:
        for cnt in kontur:
            area = cv.contourArea(cnt)
            if area > 500 and area < (citra.shape[0] * citra.shape[1] * 0.9):
                x, y, w, h = cv.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:
                    kontur_terfilter.append(cnt)
        
        kontur_terfilter = sorted(kontur_terfilter, key=cv.contourArea, reverse=True)
    
    return citra_threshold, kontur_terfilter[:1] if kontur_terfilter else []

def deteksi_bentuk_geometri(kontur: np.ndarray) -> Tuple[str, float, str]:
    """
    Mendeteksi bentuk geometri berdasarkan jumlah sudut dan properti kontur.
    Mengembalikan bentuk 2D, tingkat kepercayaan, dan bentuk ruang (jika ada).
    
    Args:
        kontur (np.ndarray): Kontur dari objek.
    
    Returns:
        Tuple[str, float, str]: (nama_bentuk_2D, tingkat_kepercayaan, nama_bentuk_ruang)
    """
    epsilon = 0.04 * cv.arcLength(kontur, True)
    approx = cv.approxPolyDP(kontur, epsilon, True)
    vertices = len(approx)
    
    area = cv.contourArea(kontur)
    perimeter = cv.arcLength(kontur, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
    
    x, y, w, h = cv.boundingRect(approx)
    aspect_ratio = float(w) / h if h != 0 else 1.0
    shape_2d = "Tidak Dikenal"
    confidence = 0.6
    shape_3d = "Tidak Diketahui"

    if circularity > 0.90:
        shape_2d = "Lingkaran"
        confidence = circularity
        shape_3d = "Silinder"
    elif vertices == 3:
        shape_2d = "Segitiga"
        confidence = 0.8
        shape_3d = "Prisma Segitiga"
    elif vertices == 4:
        if 0.95 <= aspect_ratio <= 1.05:
            shape_2d = "Persegi"
            confidence = 0.9
            shape_3d = "Kubus"
        else:
            shape_2d = "Persegi Panjang"
            confidence = 0.85
            shape_3d = "Balok"
    elif vertices == 5:
        shape_2d = "Pentagon"
        confidence = 0.8
        shape_3d = "Prisma Pentagon"
    elif vertices == 6:
        shape_2d = "Heksagon"
        confidence = 0.8
        shape_3d = "Prisma Heksagon"
    else:
        shape_2d = "Poligon"
        confidence = 0.6
        shape_3d = "Prisma Tidak Beraturan"

    return shape_2d, confidence, shape_3d

def analisis_properti_geometri(kontur: np.ndarray) -> Dict:
    """
    Menganalisis properti geometri secara detail.
    Returns:
        Dict: Dictionary berisi properti-properti geometri
    """
    try:
        # Properti dasar
        area = cv.contourArea(kontur)
        perimeter = cv.arcLength(kontur, True)
        
        # Momen
        M = cv.moments(kontur)
        cx = int(M['m10']/M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01']/M['m00']) if M['m00'] != 0 else 0
        
        # Properti tambahan
        hull = cv.convexHull(kontur)
        hull_area = cv.contourArea(hull)
        
        # Hitung properti dengan safety checks
        x, y, w, h = cv.boundingRect(kontur)
        aspect_ratio = float(w)/h if h != 0 else 1.0
        solidity = float(area)/hull_area if hull_area > 0 else 0
        extent = float(area)/(w*h) if w*h != 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'center_x': cx,
            'center_y': cy,
            'solidity': solidity,
            'circularity': circularity,
            'extent': extent,
            'aspect_ratio': aspect_ratio,
            'width': w,
            'height': h
        }
    except Exception as e:
        print(f"Error in analisis_properti_geometri: {str(e)}")
        return {
            'area': 0,
            'perimeter': 0,
            'center_x': 0,
            'center_y': 0,
            'solidity': 0,
            'circularity': 0,
            'extent': 0,
            'aspect_ratio': 1.0,
            'width': 0,
            'height': 0
        }

def visualisasi_hasil(
    citra_asli: np.ndarray,
    citra_threshold: np.ndarray,
    citra_base: np.ndarray,
    citra_proses_rgb: np.ndarray,
    kontur: List[np.ndarray],
    faktor_skala: float
) -> None:
    """
    Visualisasi hasil dengan tambahan analisis geometri.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot dasar
    plt.subplot(231)
    plt.imshow(ubah_ukuran_citra(citra_asli, faktor_skala), cmap='gray')
    plt.title('Citra Asli')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(citra_threshold, cmap='gray')
    plt.title('Threshold')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(citra_proses_rgb)
    plt.title('Kontur Terdeteksi')
    plt.axis('off')
    
    if kontur:
        # Analisis bentuk
        shape, confidence = deteksi_bentuk_geometri(kontur[0])
        properties = analisis_properti_geometri(kontur[0])
        
        # Plot properti geometri
        plt.subplot(234)
        categories = ['Circularity', 'Solidity', 'Extent']
        values = [properties['circularity'], properties['solidity'], properties['extent']]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax = plt.gca()
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title(f'Properti Bentuk: {shape}\nKepercayaan: {confidence:.2f}')
        
        # Plot histogram titik kontur
        plt.subplot(235)
        points = kontur[0].reshape(-1, 2)
        plt.hist2d(points[:, 0], points[:, 1], bins=30)
        plt.title('Distribusi Titik Kontur')
        plt.colorbar()
        
        # Plot properti ukuran
        plt.subplot(236)
        metrics = ['Area', 'Perimeter', 'Aspect Ratio']
        sizes = [
            properties['area']/1000,  # Skala ke ribu pixel
            properties['perimeter']/100,  # Skala ke ratus pixel
            properties['aspect_ratio']
        ]
        plt.bar(metrics, sizes)
        plt.title('Metrik Ukuran')
        plt.yscale('log')
        
        # Print informasi tambahan
        print(f"\nAnalisis Geometri:")
        print(f"================")
        print(f"Bentuk Terdeteksi: {shape}")
        print(f"Tingkat Kepercayaan: {confidence:.2f}")
        print(f"Luas: {properties['area']:.2f} piksel²")
        print(f"Keliling: {properties['perimeter']:.2f} piksel")
        print(f"Rasio Aspek: {properties['aspect_ratio']:.2f}")
        print(f"Pusat Massa: ({properties['center_x']}, {properties['center_y']})")
        
    plt.tight_layout()
    plt.show()

def analisis_kontur(kontur: np.ndarray) -> None:
    """
    Menganalisis dan mencetak properti kontur seperti luas dan keliling.
    
    Args:
        kontur (np.ndarray): Kontur yang akan dianalisis.
    """
    luas = cv.contourArea(kontur)
    keliling = cv.arcLength(kontur, True)
    print(f"Luas Kontur: {luas:.2f} piksel")
    print(f"Keliling Kontur: {keliling:.2f} piksel")

def deteksi_otomatis_parameter(citra: np.ndarray) -> Tuple[bool, int, int]:
    """
    Mendeteksi parameter yang optimal berdasarkan karakteristik citra.
    
    Args:
        citra (np.ndarray): Citra dalam format grayscale.
        
    Returns:
        Tuple[bool, int, int]: mode_adaptif, nilai_threshold, ukuran_kernel_dilasi
    """
    hist = cv.calcHist([citra], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    mean = np.mean(citra)
    std = np.std(citra)
    
    peaks = []
    for i in range(1, 255):
        if hist[i-1] < hist[i] and hist[i] > hist[i+1]:
            peaks.append((i, hist[i]))
    
    significant_peaks = [p for p in peaks if p[1] > 0.01]
    is_bimodal = len(significant_peaks) >= 2
    
    mode_adaptif = not is_bimodal and std > 30
    
    if is_bimodal and len(significant_peaks) >= 2:
        peaks_sorted = sorted(significant_peaks, key=lambda x: x[1], reverse=True)
        nilai_threshold = int((peaks_sorted[0][0] + peaks_sorted[1][0]) / 2)
    else:
        nilai_threshold = int(max(30, mean - std/2))
    
    tinggi, lebar = citra.shape
    ukuran_rata = (tinggi + lebar) / 2
    ukuran_kernel_dilasi = max(3, min(5, int(ukuran_rata / 300)))
    
    return mode_adaptif, nilai_threshold, ukuran_kernel_dilasi

def main():
    """
    Fungsi utama untuk melakukan deteksi kontur dan visualisasi.
    """
    try:
        FOLDER_citra = 'images'
        NAMA_citra = 'tesla.png'
        FAKTOR_SKALA = 4.0
        
        direktori_utama = os.getcwd()
        lokasi_citra = os.path.join(direktori_utama, FOLDER_citra, NAMA_citra)
        
        citra_asli = muat_citra(lokasi_citra)
        citra_diubah_ukuran = ubah_ukuran_citra(citra_asli, FAKTOR_SKALA)
        
        kombinasi_parameter = [
            (False, 65, 3, False),
            (False, 127, 3, False),
            (False, 200, 3, False),
            (False, 65, 3, True),
            (False, 127, 5, True),
            (True, 0, 3, False),
            (True, 0, 3, True)
        ]
        
        kontur_terbaik = []
        citra_threshold_terbaik = None
        jumlah_titik_terbaik = 0
        parameter_terbaik = None
        
        for mode, nilai, ukuran, invert in kombinasi_parameter:
            citra_threshold, kontur = proses_kontur(
                citra_diubah_ukuran,
                mode_adaptif=mode,
                nilai_threshold=nilai,
                ukuran_kernel_dilasi=ukuran,
                invert_threshold=invert
            )
            
            if kontur and len(kontur[0]) > jumlah_titik_terbaik:
                jumlah_titik_terbaik = len(kontur[0])
                kontur_terbaik = kontur
                citra_threshold_terbaik = citra_threshold
                parameter_terbaik = (mode, nilai, ukuran, invert)
                
                if jumlah_titik_terbaik > 500:
                    break
        
        if not kontur_terbaik:
            print("Tidak ada kontur yang terdeteksi dengan semua parameter yang dicoba.")
            return
            
        kontur = kontur_terbaik
        citra_threshold = citra_threshold_terbaik
        mode_adaptif, nilai_threshold, ukuran_kernel_dilasi, invert_threshold = parameter_terbaik
        
        citra_base = citra_diubah_ukuran.copy()
        citra_proses_rgb = cv.cvtColor(citra_diubah_ukuran, cv.COLOR_GRAY2RGB)
        cv.drawContours(citra_proses_rgb, kontur, -1, (0, 255, 0), 3)
        
        analisis_kontur(kontur[0])
        print(f"Jumlah titik dalam kontur: {len(kontur[0])}")
        print(f"Parameter terbaik yang digunakan:")
        print(f"- Mode threshold: {'Adaptif' if mode_adaptif else 'Global'}")
        print(f"- Nilai threshold: {nilai_threshold}")
        print(f"- Ukuran kernel: {ukuran_kernel_dilasi}x{ukuran_kernel_dilasi}")
        print(f"- Inverse threshold: {'Ya' if invert_threshold else 'Tidak'}")
        
        visualisasi_hasil(citra_asli, citra_threshold, citra_base, citra_proses_rgb, kontur, FAKTOR_SKALA)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error tak terduga: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()