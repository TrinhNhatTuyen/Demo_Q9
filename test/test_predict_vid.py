from ultralytics import YOLO
import cv2, os, time
from threading import Thread

def predict_no_thread():
    # a = fire_model.predict(source='bus.jpg', device=0)
    # Đường dẫn đến thư mục chứa các hình ảnh
    folder_path = "hinhtest"

    # Danh sách các file jpg trong thư mục
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]

    # Tính tổng thời gian
    start_time = time.time()
    # Khởi tạo model YOLO
    fire_model = YOLO('set_hsv_warmup_epoch_3.pt')
    # Lặp qua từng file và đo thời gian dự đoán
    for file in files:
        start_t = time.time()
        results = fire_model.predict(source=file, device=0, save=False)
        print(f"\nPredict Time: {time.time() - start_t:.2f} seconds")
    print(f"\nTotal Time: {time.time() - start_time:.2f} seconds")

def predict_image(image_path):
    fire_model1 = YOLO('set_hsv_warmup_epoch_3.pt')
    results = fire_model1.predict(source=image_path, device=0, save=False)

def main():
    folder_path = "hinhtest"
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]
    files.reverse()
    num_threads = len(files)
    threads = []
   
    for i in range(num_threads):
        image_path = [files[i]]
        thread = Thread(target=predict_image, args=(image_path))
        threads.append(thread)
    
    start_time = time.time()
    
    for thread in threads:
        thread.start()
        break

    for thread in threads:
        thread.join()
        break

    total_time = time.time() - start_time
    print(f"\nTotal Time: {total_time:.2f} seconds")

predict_no_thread()
main()


# Starting threads that each have their own model instance
# Thread(target=predict_image, args=("hinhtest/a (1).jpg",)).start()
# Thread(target=predict_image, args=("hinhtest/a (1).jpg",)).start()