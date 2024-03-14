import os, shutil, cv2, pyodbc, base64, time
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.spatial.distance import cosine, euclidean
from keras_vggface.utils import preprocess_input
from facereg_model import loadVggFaceModel
from scipy.spatial.distance import euclidean
from PIL import Image
import numpy as np

def image_path_to_array(image_path):
    # Đọc hình ảnh từ đường dẫn và chuyển đổi thành mảng NumPy
    img = Image.open(image_path)
    img_array = np.array(img)/255.0
    return img_array

def calculate_euclidean_distance(image_path1, image_path2):
    thresh = 127
    # Chuyển đổi hai hình ảnh thành mảng NumPy
    img_array1 = image_path_to_array(image_path1)
    img_array2 = image_path_to_array(image_path2)

    # Tính khoảng cách Euclidean giữa hai mảng
    distance = euclidean(img_array1.ravel(), img_array2.ravel())
    
    return distance

# # Sử dụng hàm để tính khoảng cách giữa hai hình ảnh
# image_path1 = "hinhtrain/5015/1/2023-09-19_13h13m55s.jpg"
# image_path2 = "hinhtrain/5015/1/2023-09-19_13h26m27s.jpg"

# distance = calculate_euclidean_distance(image_path1, image_path2)
# print(f"Khoảng cách Euclidean giữa hai hình ảnh: {distance}")


def download_hinhtrain():
    """Xóa "hinhtrain" và tải lại tất cả
    """
    # Clear folder "hinhtrain" 
    if os.path.exists("hinhtrain"):
        shutil.rmtree("hinhtrain")
        
    try:
        conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=112.78.15.3;"
                      "Database=VinaAIAPP;"
                      "uid=ngoi;"
                      "pwd=admin123;")
        cursor = conn.cursor()
        cursor.execute("SELECT ImageID, Base64, ImagePath FROM FaceRegData")
        for row in cursor.fetchall():
            image_id, base64_data, image_path = row
            if os.path.exists(image_path):
                continue
            if base64_data:
                try:
                    anh_base64 = np.frombuffer(base64.b64decode(base64_data), dtype=np.uint8)
                    anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
                    
                    if not os.path.exists(os.path.dirname(image_path)):
                        os.makedirs(os.path.dirname(image_path))
                        
                    cv2.imwrite(image_path, anh_base64)

                    print(f"Đã lưu hình ảnh từ ImageID {image_id} vào {image_path}")

                except Exception as e:
                    print(f"Lỗi khi xử lý ImageID {image_id}: {str(e)}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Lỗi khi kết nối đến cơ sở dữ liệu: {str(e)}")

def load_hinhtrain(root_folder='hinhtrain'):
    """ Trả về dict có cấu trúc như thư mục "hinhtrain", chứa 3D array của các ảnh nhận diện
    """
    image_data = {}  # Dictionary để lưu trữ dữ liệu ảnh
    vgg_model = loadVggFaceModel()
    # Duyệt qua các thư mục HomeID
    for homeid in os.listdir(root_folder):
        home_path = os.path.join(root_folder, homeid)
        if os.path.isdir(home_path):
            
            folder_face_id = {}
            
            # Duyệt qua các thư mục FaceID
            for face_id in os.listdir(home_path):
                face_path = os.path.join(home_path, face_id)
                if os.path.isdir(face_path):
                    
                    folder_images = {}
                    
                    for image_name in os.listdir(face_path):
                        image_path = os.path.join(face_path, image_name)

                        # Kiểm tra xem image_path có phải là tệp hình ảnh và tồn tại không
                        if os.path.isfile(image_path) and image_name.endswith(('.jpg', '.jpeg', '.png')):
                            # Sử dụng OpenCV để đọc hình ảnh và chuyển đổi thành mảng NumPy
                            image = cv2.imread(image_path)
                            
                            img_pixels = Image.fromarray(image)
                            img_pixels = np.expand_dims(img_pixels, axis = 0)
                            samples = np.asarray(img_pixels, 'float32')
                            # prepare the face for the model, e.g. center pixels
                                ####
                            samples = preprocess_input(samples, version=2)
                            samples_fn = np.asarray(img_pixels, 'float32')
                            samples_fn = preprocess_input(samples_fn, version=2)
                            representation = vgg_model.predict(samples_fn)
                            
                            if image is not None:
                                # Đưa về 1D array và chuẩn hóa các giá trị pixel về khoảng 0-1
                                folder_images[image_name] = representation

                    # Thêm dictionary của thư mục vào dictionary tổng
                    folder_face_id[face_id] = folder_images
                
            image_data[homeid] = folder_face_id
    
    return image_data


#---------------------------------------------------------------------------------------------
# start_time = time.time()
# download_hinhtrain()
# image_data = load_hinhtrain()

# elapsed_time = time.time() - start_time
# print("Thời gian chạy hàm: {:.2f} giây".format(elapsed_time))
def compare_imgs(image_data, face2compare, homeid):
    """ So khớp khuôn mặt

    Args:
        image_data (dict): Danh sách các array trong dữ liệu nhận diện
        face2compare (array): captured_representation = model.predict(samples_fn)
        homeid (str): homeid của cam hiện tại
    """
    name = None
    image = None
    minratio=1
    for face_id in image_data[homeid]:
        for img in image_data[homeid][face_id]:
            if img=='2023-09-19_13h13m55s.jpg':
                continue
            print(image_data[homeid][face_id][img].shape)
            similarity = cosine(image_data[homeid][face_id][img].reshape(-1), face2compare.reshape(-1))
            # print(euclidean(employees[CC], captured_representation))
            if(similarity)<minratio: 
                minratio=similarity
                name = face_id
                image = img
                
                
    eucli = euclidean(image_data[homeid][name][image].reshape(-1), face2compare.reshape(-1))
    return(minratio < 0.38 and eucli <90)

# a = cv2.imread('hinhtrain/5015/1/2023-09-19_13h13m55s.jpg').astype(np.float32) / 255.0
# result = compare_imgs(image_data, face2compare=a, homeid='5015')
# print(result)
print('Done!')


# for homeid in image_data:
#     for face_id in image_data[homeid]:
#         for img in image_data[homeid][face_id]:
#             print(image_data[homeid][face_id][img].shape)        
