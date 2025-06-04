import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

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

def visualisasi_hasil(
    citra_asli: np.ndarray,
    citra_threshold: np.ndarray,
    citra_base: np.ndarray,
    citra_proses_rgb: np.ndarray,
    kontur: List[np.ndarray],
    faktor_skala: float
) -> None:
    """
    Menampilkan hasil pemrosesan citra dalam layout 2x3 subplot.
    
    Args:
        citra_asli (np.ndarray): Citra asli dalam format grayscale.
        citra_threshold (np.ndarray): citra hasil threshold.
        citra_base (np.ndarray): citra grayscale dasar tanpa kontur berwarna.
        citra_proses_rgb (np.ndarray): citra RGB dengan kontur berwarna.
        kontur (List[np.ndarray]): Daftar kontur yang ditemukan.
        faktor_skala (float): Faktor skala yang digunakan untuk mengubah ukuran.
    """
    citra_asli_diubah = ubah_ukuran_citra(citra_asli, faktor_skala)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(citra_asli_diubah, cmap='gray')
    plt.title('Citra Asli')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(citra_threshold, cmap='gray')
    plt.title('Threshold')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(citra_proses_rgb)
    plt.title('Kontur (Garis Hijau)')
    plt.axis('off')
    
    if not kontur:
        plt.tight_layout()
        plt.show()
        return
    
    citra_kontur_detail = citra_base.copy()
    citra_kontur_detail_rgb = cv.cvtColor(citra_kontur_detail, cv.COLOR_GRAY2RGB)
    for point in kontur[0]:
        x, y = point[0]
        cv.circle(citra_kontur_detail_rgb, (x, y), 2, (0, 0, 255), -1)
    
    plt.subplot(234)
    plt.imshow(citra_kontur_detail_rgb)
    plt.title(f'Titik Kontur (total: {len(kontur[0])})')
    plt.axis('off')
    
    momen = cv.moments(kontur[0])
    if momen['m00'] != 0:
        pusat_x = int(momen['m10'] / momen['m00'])
        pusat_y = int(momen['m01'] / momen['m00'])
        citra_poligon = cv.cvtColor(citra_base.copy(), cv.COLOR_GRAY2RGB)
        keliling = cv.arcLength(kontur[0], True)
        epsilon = 0.005 * keliling
        poligon_aproksimasi = cv.approxPolyDP(kontur[0], epsilon, True)
        cv.polylines(citra_poligon, [poligon_aproksimasi], True, (0, 0, 255), 2)
        cv.circle(citra_poligon, (pusat_x, pusat_y), 5, (255, 0, 0), -1)
        plt.subplot(235)
        plt.imshow(citra_poligon)
        plt.title(f'Poligon Aproksimasi ({len(poligon_aproksimasi)} titik)')
        plt.axis('off')
    
    ex_hull = cv.convexHull(kontur[0])
    titik_hull = np.concatenate((ex_hull[:, 0, :], ex_hull[:1, 0, :]), axis=0)
    plt.subplot(235)
    plt.imshow(citra_base, cmap='gray')
    plt.plot(titik_hull[:, 0], titik_hull[:, 1], 'r-', label='Convex Hull')
    plt.title('Convex Hull')
    plt.legend()
    plt.axis('off')
    
    x, y, lebar, tinggi = cv.boundingRect(kontur[0])
    citra_dengan_kotak = citra_base.copy()
    cv.rectangle(citra_dengan_kotak, (x, y), (x + lebar, y + tinggi), (255), 2)
    plt.subplot(236)
    plt.imshow(citra_dengan_kotak, cmap='gray')
    plt.title('Kotak Pembatas')
    plt.axis('off')
    
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()