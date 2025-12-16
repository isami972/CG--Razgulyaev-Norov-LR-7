import sys
import subprocess
import importlib

def install_and_import(package):
    """Автоматически устанавливает и импортирует библиотеку"""
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Библиотека {package} не найдена. Устанавливаем...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Библиотека {package} успешно установлена!")

# Устанавливаем необходимые библиотеки
required_packages = ['Pillow', 'numpy']

print("=" * 60)
print("Лабораторная работа №7 - Фильтрация изображений")
print("Вариант 16: ФНЧ №3 и ФВЧ Робертса/Собела")
print("=" * 60)

for package in required_packages:
    try:
        install_and_import(package if package != 'Pillow' else 'PIL')
        print(f"✓ {package} - OK")
    except Exception as e:
        print(f"✗ Ошибка установки {package}: {e}")
        input("Нажмите Enter для выхода...")
        sys.exit(1)

print("=" * 60)
print("ВСЕ БИБЛИОТЕКИ УСТАНОВЛЕНЫ. ЗАПУСК ПРОГРАММЫ...")
print("=" * 60)

# ==================== ОСНОВНОЙ КОД ПРОГРАММЫ ====================
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import os


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Фильтрация изображений - Вариант 16")
        self.root.geometry("1200x800")
        
        # Определяем ФНЧ фильтры из таблицы (примерные)
        self.lpf_filters = {
            "ФНЧ №1": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Усредняющий
            "ФНЧ №2": [[1, 2, 1], [2, 4, 2], [1, 2, 1]],  # Более плавный
            "ФНЧ №3": [[1, 1, 1], [1, 2, 1], [1, 1, 1]],  # Третий из таблицы
        }
        
        # Variables
        self.original_image = None
        self.lpf_result = None
        self.hpf_roberts_result = None
        self.hpf_sobel_result = None
        self.final_lpf_result = None
        self.final_hpf_result = None
        self.selected_lpf = "ФНЧ №3"  # По умолчанию ФНЧ №3

        self.create_widgets()

    def create_widgets(self):
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Load button
        self.load_btn = tk.Button(control_frame, text="Загрузить изображение", 
                                  command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # LPF selection
        lpf_frame = tk.LabelFrame(control_frame, text="Выбор ФНЧ")
        lpf_frame.pack(side=tk.LEFT, padx=10)
        
        self.lpf_var = tk.StringVar(value="ФНЧ №3")
        lpf_combo = ttk.Combobox(lpf_frame, textvariable=self.lpf_var, 
                                values=list(self.lpf_filters.keys()),
                                state="readonly", width=10)
        lpf_combo.pack(padx=5, pady=2)
        
        self.lpf_btn = tk.Button(lpf_frame, text="Применить ФНЧ и вычесть", 
                                command=self.apply_low_pass_filter)
        self.lpf_btn.pack(padx=5, pady=2)

        # HPF button
        self.hpf_btn = tk.Button(control_frame, text="Применить ФВЧ Робертса и Собела", 
                                 command=self.apply_high_pass_filters)
        self.hpf_btn.pack(side=tk.LEFT, padx=5)

        # Save button
        self.save_btn = tk.Button(control_frame, text="Сохранить результаты", 
                                  command=self.save_results)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Info label
        info_label = tk.Label(control_frame, 
                             text="Вариант 16: 1) ФНЧ №3 → вычитание; 2) ФВЧ Робертса-Собела → разность",
                             fg="blue", font=("Arial", 10))
        info_label.pack(side=tk.LEFT, padx=20)

        # Image display frame
        images_frame = tk.Frame(self.root)
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Original image
        orig_frame = tk.LabelFrame(images_frame, text="Исходное изображение")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_label = tk.Label(orig_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # LPF result
        lpf_frame_display = tk.LabelFrame(images_frame, text="ФНЧ → Вычитание")
        lpf_frame_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.lpf_label = tk.Label(lpf_frame_display)
        self.lpf_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # HPF result
        hpf_frame_display = tk.LabelFrame(images_frame, text="Разность Робертса и Собела")
        hpf_frame_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.hpf_label = tk.Label(hpf_frame_display)
        self.hpf_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_bar = tk.Label(self.root, text="Готов к работе", bd=1, 
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.gif *.ppm *.tiff"),
                      ("Все файлы", "*.*")]
        )
        if file_path:
            try:
                self.original_image = Image.open(file_path).convert('RGB')
                self.display_image(self.original_image, self.original_label)
                self.status_bar.config(text=f"Загружено: {os.path.basename(file_path)}")
                messagebox.showinfo("Успех", "Изображение успешно загружено!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")

    def display_image(self, image, label_widget, max_size=(300, 300)):
        """Отображает изображение в Label"""
        if image is None:
            label_widget.config(image='', text="Нет изображения")
            return
        
        image_copy = image.copy()
        try:
            image_copy.thumbnail(max_size)
        except:
            pass
        
        photo = ImageTk.PhotoImage(image_copy)
        label_widget.config(image=photo)
        label_widget.image = photo

    def apply_low_pass_filter(self):
        """Применяет выбранный ФНЧ фильтр и вычитает результат из исходного"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        try:
            selected_filter = self.lpf_var.get()
            kernel = self.lpf_filters[selected_filter]
            
            self.status_bar.config(text=f"Применяем {selected_filter}...")
            
            # Применяем ФНЧ фильтр
            lpf_filtered = self.apply_custom_filter(self.original_image, kernel)
            self.lpf_result = lpf_filtered
            
            # Вычитание результата ФНЧ из исходного изображения
            original_array = np.array(self.original_image, dtype=np.float32)
            lpf_array = np.array(lpf_filtered, dtype=np.float32)
            
            # Вычитаем и нормализуем
            subtracted_array = original_array - lpf_array
            subtracted_array = np.clip(subtracted_array + 128, 0, 255)
            
            self.final_lpf_result = Image.fromarray(subtracted_array.astype(np.uint8))
            self.display_image(self.final_lpf_result, self.lpf_label)
            
            self.status_bar.config(text=f"{selected_filter} применен. Результат вычтен из исходного.")
            messagebox.showinfo("Успех", 
                               f"{selected_filter} применен и результат вычтен из исходного!\n"
                               f"Ядро фильтра: {kernel}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при ФНЧ-фильтрации: {str(e)}")
            self.status_bar.config(text="Ошибка при ФНЧ-фильтрации")

    def apply_custom_filter(self, image, kernel):
        """Применяет кастомный фильтр свёртки"""
        # Нормализуем ядро (сумма = 1)
        kernel_array = np.array(kernel, dtype=np.float32)
        kernel_sum = np.sum(kernel_array)
        if kernel_sum != 0:
            kernel_array /= kernel_sum
        
        # Конвертируем в numpy
        img_array = np.array(image, dtype=np.float32)
        height, width, channels = img_array.shape
        
        # Размер ядра
        k_height, k_width = kernel_array.shape
        pad_h, pad_w = k_height // 2, k_width // 2
        
        # Создаем паддинг
        padded = np.pad(img_array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 
                       mode='edge')
        
        result = np.zeros_like(img_array)
        
        # Применяем свёртку
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    region = padded[y:y+k_height, x:x+k_width, c]
                    result[y, x, c] = np.sum(region * kernel_array)
        
        # Нормализуем результат
        result = np.clip(result, 0, 255)
        return Image.fromarray(result.astype(np.uint8))

    def apply_high_pass_filters(self):
        """ФВЧ Робертса и ФВЧ Собела с вычислением разности"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        try:
            self.status_bar.config(text="Применяем ФВЧ Робертса и Собела...")
            
            # Конвертируем в градации серого для градиентных фильтров
            gray_image = self.original_image.convert('L')
            
            # ФВЧ Робертса
            roberts_result = self.apply_roberts_filter(gray_image)
            self.hpf_roberts_result = roberts_result
            
            # ФВЧ Собела
            sobel_result = self.apply_sobel_filter(gray_image)
            self.hpf_sobel_result = sobel_result
            
            # Находим разность результатов
            roberts_array = np.array(roberts_result, dtype=np.float32)
            sobel_array = np.array(sobel_result, dtype=np.float32)
            
            difference_array = np.abs(roberts_array - sobel_array)
            difference_array = np.clip(difference_array, 0, 255)
            
            self.final_hpf_result = Image.fromarray(difference_array.astype(np.uint8))
            self.display_image(self.final_hpf_result, self.hpf_label)
            
            self.status_bar.config(text="ФВЧ Робертса и Собела применены. Разность найдена.")
            messagebox.showinfo("Успех", 
                               "ФВЧ Робертса и Собела применены!\n"
                               "Разность результатов вычислена.")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при ФВЧ-фильтрации: {str(e)}")
            self.status_bar.config(text="Ошибка при ФВЧ-фильтрации")

    def apply_roberts_filter(self, image):
        """Применяет оператор Робертса (перекрёстный градиент)"""
        image_array = np.array(image, dtype=np.float32)
        height, width = image_array.shape
        
        result = np.zeros_like(image_array)
        
        # Ядра Робертса
        for y in range(height - 1):
            for x in range(width - 1):
                gx = image_array[y, x] - image_array[y + 1, x + 1]
                gy = image_array[y, x + 1] - image_array[y + 1, x]
                magnitude = np.sqrt(gx**2 + gy**2)
                result[y, x] = np.clip(magnitude, 0, 255)
        
        return Image.fromarray(result.astype(np.uint8))

    def apply_sobel_filter(self, image):
        """Применяет оператор Собела"""
        image_array = np.array(image, dtype=np.float32)
        height, width = image_array.shape
        
        result = np.zeros_like(image_array)
        
        # Ядра Собела
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        kernel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                region = image_array[y-1:y+2, x-1:x+2]
                gx = np.sum(region * kernel_x)
                gy = np.sum(region * kernel_y)
                magnitude = np.sqrt(gx**2 + gy**2)
                result[y, x] = np.clip(magnitude, 0, 255)
        
        return Image.fromarray(result.astype(np.uint8))

    def save_results(self):
        """Сохраняет все результаты в PNG и PPM"""
        results = []
        if self.final_lpf_result:
            results.append(("lpf_subtraction", self.final_lpf_result))
        if self.final_hpf_result:
            results.append(("hpf_difference", self.final_hpf_result))
        if self.lpf_result:
            results.append(("lpf_filtered", self.lpf_result))
        if self.hpf_roberts_result:
            results.append(("roberts_only", self.hpf_roberts_result))
        if self.hpf_sobel_result:
            results.append(("sobel_only", self.hpf_sobel_result))
        
        if not results:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения!")
            return
        
        folder_path = filedialog.askdirectory(title="Выберите папку для сохранения")
        if not folder_path:
            return
        
        saved_files = []
        for name, image in results:
            try:
                # Сохраняем в PNG
                png_path = os.path.join(folder_path, f"{name}.png")
                image.save(png_path, "PNG")
                saved_files.append(f"{name}.png")
                
                # Сохраняем в PPM
                ppm_path = os.path.join(folder_path, f"{name}.ppm")
                image.save(ppm_path, "PPM")
                saved_files.append(f"{name}.ppm")
                
            except Exception as e:
                continue
        
        if saved_files:
            message = f"Сохранено файлов: {len(saved_files)}\n"
            message += "\n".join(saved_files)
            messagebox.showinfo("Успех", message)
            self.status_bar.config(text=f"Сохранено {len(saved_files)} файлов в {folder_path}")
        else:
            messagebox.showwarning("Предупреждение", "Не удалось сохранить файлы")


def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
