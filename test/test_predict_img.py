from ultralytics import YOLO
import cv2, os, time, multiprocessing
from threading import Thread
from queue import Queue

from functools import partial
from multiprocessing import Pool


# fire_model = YOLO('set_hsv_warmup_epoch_3.pt')

# r = fire_model.predict(source=['hinhtest/a (1).jpg',
#                                'hinhtest/a (2).jpg',
#                                'hinhtest/a (3).jpg',
#                                'hinhtest/a (4).jpg'], device=0, save=False)
# print()
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

# predict_no_thread()
# main()

def predict_video(i):
    video_path = f'hinhtest/test{i+1}.mp4'
    scale = 0.5
    t_oldframe = 0
    fire_model = YOLO('set_hsv_warmup_epoch_3.pt')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở video file.")
        return

    while True:
        ret, frame = cap.read()
        
        timer =time.time()
        fps = 1/(timer-t_oldframe)
        t_oldframe = timer
        
        if not ret:
            print("Hết video hoặc có lỗi khi đọc frame.")
            break
        
        results = fire_model.predict(source=frame, device=0, save=False)
        annotated_frame = results[0].plot()
        
        cv2.putText(annotated_frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
        cv2.imshow(video_path, 
                   cv2.resize(annotated_frame, (int((annotated_frame.shape[1])*scale),int((annotated_frame.shape[0])*scale))))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def stream_video(i):
    video_path = f'hinhtest/test{i+1}.mp4'
    scale = 0.5
    t_oldframe = 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở video file.")
        return

    while True:
        ret, frame = cap.read()
        
        timer =time.time()
        fps = 1/(timer-t_oldframe)
        t_oldframe = timer
        
        if not ret:
            print("Hết video hoặc có lỗi khi đọc frame.")
            break
        
        cv2.putText(frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
        cv2.imshow(video_path, 
                   cv2.resize(frame, (int((frame.shape[1])*scale),int((frame.shape[0])*scale))))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    

def main1():
    threads = []
    num_threads = 3
    for i in range(num_threads):
        
        thread = Thread(target=stream_video, args=([i]))
        threads.append(thread)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
    cv2.destroyAllWindows()
main1()



#-------------------------------------------------------------------------------------------------


list_input, list_result = [], []
def stream_video(i):
    video_path = f'hinhtest/test{i+1}.mp4'
    scale = 0.5
    t_oldframe = 0
    cap = cv2.VideoCapture(video_path)
    list_frame = []
    if not cap.isOpened():
        print("Không thể mở video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video hoặc có lỗi khi đọc frame.")
            break
        list_frame.append(frame)
    return list_frame

def prediction_worker(fire_model, num_threads):
    while True:
        if all(x is not None for x in list_input):
            results = fire_model.predict(source=list_input, device=0, save=False)
            for i in range(len(list_result)):
                list_result[i] = results[i]
            for i in range(len(list_input)):
                list_input[i] = None

def main3():
    fire_model = YOLO('set_hsv_warmup_epoch_3.pt')
    list_frame = stream_video(0)
    a = 0
    start_time = time.time()
    for i in list_frame:
        print(a)
        a+=1
        fire_model.predict(source=i, device=0, save=False)
    total_time2 = time.time() - start_time
    
    start_time = time.time()
    for i in range(0, len(list_frame), 3):
        print(i+3)
        batch = list_frame[i:i+3]
        fire_model.predict(source=batch, device=0, save=False)
    total_time1 = time.time() - start_time
    
    
    
    print(f"\nTotal Time: {total_time1:.2f} seconds")
    print(f"\nTotal Time: {total_time2:.2f} seconds")

# if __name__ == "__main__":
#     # Số lượng tiến trình bạn muốn tạo
#     num_processes = 3

#     # Danh sách các đường dẫn video
#     video_paths = [i for i in range(num_processes)]

#     # Tạo một pool của các tiến trình và sử dụng hàm predict_video với functools.partial
#     with Pool(num_processes) as pool:
#         pool.map(partial(predict_video), video_paths)


# main3()
# main4()