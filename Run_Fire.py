import cv2, torch, requests, base64, json, datetime, time, pyodbc, threading, firebase_admin
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
# from keras.models import load_model
# from keras.applications.imagenet_utils import preprocess_input
# from keras_vggface.utils import preprocess_input
# from scipy.spatial.distance import cosine, euclidean
from overstepframe import FreshestFrame
# from inside_the_box import inside_the_box
# from prepare_data import download_hinhtrain, load_hinhtrain
# from remote_lock import get_accesstoken, lock, unlock
# from face_detect import detect_face
# from facereg_model import loadVggFaceModel
# from padding_image import padding
# from shapely.geometry import Polygon
from firebase_admin import credentials, messaging

# detector = cv2.dnn.readNetFromCaffe("pre_model/deploy.prototxt","pre_model/res10_300x300_ssd_iter_140000.caffemodel")
# vgg_model = loadVggFaceModel()
# from firebase_admin import credentials, messaging
#------------------------------------ LOAD MODEL ------------------------------------
from ultralytics import YOLO
# pose_model = YOLO('YOLOv8n-pose.pt')
pose_model = YOLO('YOLOv8n.pt')
fire_model = YOLO('set_hsv_warmup_epoch_3.pt')
cred = credentials.Certificate('ngocvinaai-firebase-adminsdk-57cev-b988d1a956.json')
firebase_admin.initialize_app(cred)

#------------------------------------ PARAMETERS ------------------------------------
json_filename = "fire_param.json"

with open(json_filename, "r") as json_file:
    params = json.load(json_file)

firesmoke_conf = params['firesmoke_conf'] # Ngưỡng phát hiện khói và lửa
humanpose_conf = params['humanpose_conf'] # Ngưỡng phát hiện người
queue_len = params['queue_len']       # Độ dài hàng đợi
n = params['n']               # Số frame phát hiện khói lửa tối thiểu (để push thông báo)
scale = params['scale']
send = params['send']
diff_ratio = params['diff_ratio']
overlap_ratio = params['overlap_ratio']


# firesmoke_conf = 0.3 # Ngưỡng phát hiện khói và lửa
# humanpose_conf = 0.6 # Ngưỡng phát hiện người
# queue_len = 20       # Độ dài hàng đợi
# n = 2               # Số frame phát hiện khói lửa tối thiểu (để push thông báo)

# device = 0 if torch.cuda.is_available() else 'cpu'
device = 'cpu'

ip = '125.253.117.120'
# ip = '112.78.15.5'
port = '5001'
# Scale khung hình để xem
# scale = 0.5
    
#------------------------------------------------------------------------------------------------------------
def remove_duplicates_and_none(input_list):
    result = []
    
    for sublist in input_list:
        unique_sublist = []
        for element in sublist:
            if (element is None) or (element in unique_sublist):
                continue
            else:
                unique_sublist.append(element)
        
        result.append(unique_sublist)
    
    return result

def result_queue(q,value):
    if q.full():
        q.get()  # Lấy giá trị đầu tiên khi Queue đầy
    q.put(value)  # Thêm giá trị mới vào Queue
    if q.queue.count(1)>=n and q.queue.count(2)==0:  # Kiểm tra nếu có ít nhất n True trong Queue
        if value==1:
            return True
        else:
            return False
    else:
        return False

def get_camera_id(rtsp):
    api_url = 'http://'+ip+':'+port+'/api/notification/get-camera-id'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'rtsp': rtsp,

    }
    response = requests.post(api_url, json=data)
    return response.json()['message']

def save_ntf_img(frame, camera_id, title, body, notification_type, formatted_time, send):
    api_url = 'http://'+ip+':'+port+'/api/notification/save'
    # Chuyển đổi dữ liệu ảnh thành chuỗi base64
    _, image_data = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(image_data).decode("utf-8")
    
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'camera_id': camera_id,
        'notification_type': notification_type,
        'title': title,
        'body': body,
        'base64': base64_image,
        'formatted_time': formatted_time,
        'send': send,
    }
    response = requests.post(api_url, json=data)
    print(response.json())

def get_fcm_to_send(camera_id):
    api_url = 'http://'+ip+':'+port+'/api/notification/get-fcm-to-send'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'camera_id': camera_id,
    }
    response = requests.post(api_url, json=data)
    return json.loads(response.text)

def fcm_dat123():
    api_url = 'http://'+ip+':'+port+'/api/ntf/fcm_dat123'
    response = requests.get(api_url)
    return json.loads(response.text)

def push_alert(fcm_list, title, body, rtsp, camera_id, lock_id=''):
    
    android_config = messaging.AndroidConfig(
        priority='high',
        notification=messaging.AndroidNotification(
            sound='res_sound45', 
            priority='max'
            ),
        data={'rtsp': rtsp,
              'camera_id': str(camera_id),
              'lock_id': str(lock_id)}
    )
    apns_config = messaging.APNSConfig(
        payload=messaging.APNSPayload(
            aps=messaging.Aps(
                alert=messaging.ApsAlert(
                    title=title,
                    body=body
                ),
                sound='res_sound45.wav'
            ),
            custom_data={'rtsp': rtsp,
                         'camera_id': str(camera_id),
                         'lock_id': str(lock_id)}
        )
    )
    
    message = messaging.MulticastMessage( 
                            notification = messaging.Notification( title=title, body=body), 
                            android=android_config,
                            apns=apns_config,
                            tokens=fcm_list
                            )
    
    # Gửi thông báo đến thiết bị cụ thể
    response = messaging.send_multicast(message)
    print(f"Failure Count: {response.failure_count}")
    return response.failure_count

def get_camera_data():
    
    # Lấy ra CameraID, HomeID, CameraName, RTSP, LockpickingArea, ClimbingArea
    api_url = 'http://'+ip+':'+port+'/api/pose/get-camera-data'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
    }
    response = requests.post(api_url, json=data)
    cam_list = json.loads(response.text)
    
    """
    *  Lặp qua từng Dict trong "cam_list" và lấy danh sách FCM cần gửi thông báo, thêm vào trường FCM của dict đó
    *  Phải kiểm tra có cùng HomeID với cam trước đó k, nếu trùng k cần gọi lại "api/notification/get-fcm-to-send"
    """
    
    homeid_fcm_list = [] # List gồm các HomeID và các FCM tương ứng với HomeID đó
    
    # Lặp qua từng dict trong cam_list
    for cam in cam_list:
        call_api = True
        # Lặp qua từng dict trong homeid_fcm
        for i in homeid_fcm_list:
            # Nếu HomeID của cam nằm trong dict đã lấy FCM thì dùng lại, k cần gọi API
            if cam['HomeID']==i['HomeID']:
                cam['FCM'] = i['FCM']
                call_api = False
                break
            
        if call_api:
            cam['FCM'] = get_fcm_to_send(cam['CameraID'])
            # Thêm HomeID và các FCM tương ứng để biết đã lấy FCM của căn này rồi
            homeid_fcm_list.append({
                                    'HomeID': cam['HomeID'],
                                    'FCM': cam['FCM'],
                                    })
            
    return cam_list

def drawbox(frame, points):
    # Draw Box
    for p1, p2 in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]:
        pt1 = tuple(points[p1])
        pt2 = tuple(points[p2])
        cv2.line(frame, pt1, pt2, (0, 0, 255), 5)
    return frame

def base64_to_array(anh_base64):
        try:
            img_arr = np.frombuffer(base64.b64decode(anh_base64), dtype=np.uint8)
            img_arr = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
        except:
            return "Không chuyển được ảnh base64 sang array"
        return img_arr

def post_ntf_knownperson_detected(faceid, cameraname, fcm_list, formatted_time):
    connection = pyodbc.connect("Driver={SQL Server};"
                                "Server=112.78.15.3;"
                                "Database=VinaAIAPP;"
                                "uid=ngoi;"
                                "pwd=admin123;")
    cursor = connection.cursor()
    
    cursor.execute("SELECT FaceName FROM FaceRegData WHERE FaceID = ?", (faceid,))
    facename = cursor.fetchone().FaceName

    body = f"Đã phát hiện {facename} ở camera {cameraname} lúc {formatted_time}"

    push_alert(fcm_list=fcm_list, title="Phát hiện người quen", body=body)

def overlap(box1, box2, ratio=overlap_ratio):
    """
    Hàm tính tỉ lệ   Vùng box1 trùng với box2   so với   Kích thước box1

    Parameters:
    - box1, box2: Danh sách chứa thông số của bounding box [left, top, right, bottom].

    Returns:
    - overlap: Giá trị Intersection over Union.
    """
    if box1==0 or box1==2:
        return True
    # Tính toán diện tích của hai bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Tính toán diện tích của phần giao nhau
    intersection_left = max(box1[0], box2[0])
    intersection_top = max(box1[1], box2[1])
    intersection_right = min(box1[2], box2[2])
    intersection_bottom = min(box1[3], box2[3])

    intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)

    # Tính toán diện tích của phần hợp nhất (Union)
    # union_area = area_box1 + area_box2 - intersection_area
    # Tính giá trị IoU
    # iou = intersection_area / union_area

    overlap_ratio = intersection_area/area_box1
    return overlap_ratio>=ratio

def calculate_diff(box, frame1, frame2, ratio=diff_ratio):
    """
    Hàm tính phần trăm sự khác nhau giữa hai bounding boxes trong hai frame.

    Parameters:
    - box1, box2: Danh sách chứa thông số của bounding box [left, top, right, bottom].
    - frame1, frame2: Hai frame chứa hình ảnh.

    Returns:
    - difference_percentage: Phần trăm sự khác nhau.
    """
    
    thresh = 127
    # Lấy phần ảnh trong bounding box từ hai frame
    box_frame1 = frame1[box[1]:box[3], box[0]:box[2]]
    box_frame2 = frame2[box[1]:box[3], box[0]:box[2]]
    
    box_frame1 = cv2.threshold(box_frame1, thresh, 255, cv2.THRESH_BINARY)[1]
    box_frame2 = cv2.threshold(box_frame2, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # Tính toán sự khác nhau giữa hai phần ảnh
    diff_image = cv2.absdiff(box_frame1, box_frame2)
    diff_image = diff_image.astype(np.uint8)

    # Tính toán phần trăm sự khác nhau
    diff = np.count_nonzero(diff_image) / diff_image.size

    return diff>ratio

def diff_boxes(previous_frame, current_frame):
    kernel_size = (21,21)
    
    previous_frame = cv2.cvtColor(previous_frame,cv2.COLOR_BGR2GRAY)
    previous_frame = cv2.GaussianBlur(previous_frame,kernel_size,0)
    
    current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
    current_frame = cv2.GaussianBlur(current_frame,kernel_size,0)
    
    
    diff = cv2.absdiff(previous_frame,current_frame)
    thresh = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations = 2)
    
    """
        'cnts': Danh sách các đường viền được tìm thấy.
        'res': Ảnh kết quả sau khi đã tìm các đường viền. (k dùng)
    """
    cnts,res = cv2.findContours(thresh.copy(),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    diff_boxs = [] 
    for contour in cnts:
        if cv2.contourArea(contour) < 3600:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        left = x
        top = y
        right = x + w
        bottom = y + h
        
        diff_boxs.append([left, top, right, bottom])
    
    return diff_boxs

def main():
    # download_hinhtrain()
    # known_persons = load_hinhtrain()
    cam_data = get_camera_data()
    
    indexs = params["cam_index"]  
    cam_data = [cam_data[i] for i in indexs]
        
    # cam_data = [
    #     # cam_data[1], 
    #     cam_data[9], 
    #     cam_data[10], 
    #     # cam_data[11],
    #     ] #ff
    #------------------------------------ Các thông tin của Camera ------------------------------------
    url, lockpicking_area, climbing_area, fcm_list, camera_id, camera_name, homeid, lockid, related_camera_id, task = [], [], [], [], [], [], [], [], [], []
    # Bỏ qua các cam chưa nhập LockpickingArea & ClimbingArea và không có LockID
    for cam in cam_data:
        # if not ((cam['LockpickingArea'] is None) and (cam['ClimbingArea'] is None) and (cam['LockID'] is None)):
        url.append(cam['RTSP'])
        lockpicking_area.append(cam['LockpickingArea'])
        climbing_area.append(cam['ClimbingArea'])
        fcm_list.append(cam['FCM'])
        camera_id.append(cam['CameraID'])
        related_camera_id.append(cam['RelatedCameraID'])
        camera_name.append(cam['CameraName'])
        homeid.append(cam['HomeID'])
        lockid.append(cam['LockID'])
        # Phân biệt cam chạy FaceID hay Pose
        task.append('Pose' if cam['LockID'] is None else 'FaceID')
            

    #------------------------------------ FRESHEST FRAME ------------------------------------
    fresh, frame, cnt, first_frame, second_frame, t_oldframe, None_frame, q_knownperson, q_lockpicking, q_climbing, lastest_detected_face = [], [], [], [], [], [], [], [], [], [], []
    previous_frame, get_previous_frame, q_fire, q_smoke, q_fire_box, q_smoke_box = [], [], [], [], [], []
    for i in range(len(url)):
        fresh.append(FreshestFrame(cv2.VideoCapture(url[i])))

        frame.append(object())
        cnt.append(0)
        first_frame.append(None)
        second_frame.append(None)
        previous_frame.append(None)
        # get_previous_frame.append(None)
        t_oldframe.append(None)
        None_frame.append(0)
        lastest_detected_face.append(None)
        q_lockpicking.append(Queue(maxsize=queue_len))
        q_climbing.append(Queue(maxsize=queue_len))
        q_knownperson.append(Queue(maxsize=10))
        q_fire.append(Queue(maxsize=queue_len))
        q_smoke.append(Queue(maxsize=queue_len))
        # q_fire_box.append(Queue(maxsize=queue_len))
        # q_smoke_box.append(Queue(maxsize=queue_len))
    
    #======================= Params =======================#
    label_mapping = {
        0: "fire",
        1: "smoke",
        }
    # cap = cv2.VideoCapture('a.mp4')
    #===========================================================================================================#
    try:
        while True:
            # Release FreshestFrame objects every 10 minutes
            t = datetime.datetime.now()
            if (t.minute % 50 == 0) and t.second<2:
                raise Exception("Restarting...")
            
            for CC in range(len(url)):
                cnt[CC],frame[CC] = fresh[CC].read(seqnumber=cnt[CC]+1, timeout=5)
                # cnt[CC],frame[CC] = cap.read()

                # Nếu đọc được frame là None
                if frame[CC] is None:
                    None_frame[CC]+=1
                    
                # Nếu đọc được 5 frame là None
                if None_frame[CC]>5:
                    print("Cannot read frame from camera!")
                    raise Exception()

                # dùng để tính toán FPS
                timer =time.time()
                if t_oldframe[CC] is None:
                    t_oldframe[CC] = timer
                
                try:
                    frame[CC] = cv2.resize(frame[CC], (1920,1080))
                except:
                    pass
                
                if previous_frame[CC] is None:
                    previous_frame[CC] = frame[CC].copy()
                    continue
                
                # Lưu 2 background đầu tiên -> continue
                # for i in previous_frame[CC]:
                #     if i is None:
                #         i = frame[CC].copy()
                #         continue
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Từ frame thứ 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                current_time = datetime.datetime.now()
                
                fps = 1/(timer-t_oldframe[CC])
                t_oldframe[CC] = timer
                
                # first_frame[CC] = cv2.cvtColor(first_frame[CC], cv2.COLOR_BGR2GRAY)
                # first_frame[CC] = cv2.threshold(first_frame[CC], thresh, 255, cv2.THRESH_BINARY)[1]

                # second_frame[CC] = frame[CC].copy()
                annotated_frame = frame[CC].copy()
                #=================================================================================================#
                
                # Nếu KHÔNG có chuyển động thì KHÔNG PREDICT
                frame_diff = diff_boxes(previous_frame[CC], frame[CC].copy())
                if len(frame_diff)==0:
                    
                    result_queue(q_fire[CC], 0)
                    result_queue(q_smoke[CC], 0)
                    
                    # Hiện FPS
                    cv2.putText(annotated_frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)

                    cv2.imshow('Fire - ' + camera_name[CC], 
                                cv2.resize(annotated_frame, (int((annotated_frame.shape[1])*scale),int((annotated_frame.shape[0])*scale))))
                    continue
                else:
                    previous_frame[CC] = frame[CC].copy()

                #=================================================================================================#
                
                
                max_prob_fire = None
                max_prob_smoke = None
                max_prob_person = None
                no_person = True
                formatted_time = current_time.strftime("%d-%m-%Y %Hh%M'%S\"")
                formatted_time_ntf = ' lúc ' + current_time.strftime("%Hh%M'%S\" %d-%m-%Y")
                
                copy_frame = frame[CC].copy()
                # Nhận diện con người
                results_human = pose_model.predict(source=copy_frame, conf=humanpose_conf, device=device, save=False, classes=[0])
                
                annotated_frame = results_human[0].plot(boxes=False)
                
                # Nếu có người
                if len(results_human[0].boxes.data)>0:
                    no_person = False
                    result_queue(q_fire[CC], 2)
                    result_queue(q_smoke[CC], 2)
                else:    
                    # Nhận diện lửa khói
                    results_fire = fire_model.predict(source=copy_frame, conf=firesmoke_conf, device=device, save=False)
                    
                    # Vẽ box cho khói-lửa
                    for result in results_fire[0].boxes.data:
                        draw_box_flag = False
                        
                        # Thiết lập font chữ
                        font_path = 'arial.ttf'
                        font_size = 40
                        font_color = (255, 255, 255)  # Màu trắng (B, G, R)
                        font = ImageFont.truetype(font_path, font_size)                                                         

                        # Vẽ khung chữ nhật
                        left = int(result[0])
                        top = int(result[1])
                        right = int(result[2])
                        bottom = int(result[3])
                        prob = result[4]*100
                        label = label_mapping.get(int(result[5]), "unknown")
                        
                        if label=='fire':
                            background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                            text = "Lửa ({:.2f}%)".format(prob)
                            if max_prob_fire==None or max_prob_fire<prob:
                                for i in frame_diff:
                                    if overlap(i, [left,top,right,bottom]):
                                        draw_box_flag = True
                                        max_prob_fire=prob
                                        fire_box = [left,top,right,bottom]
                                        cv2.rectangle(annotated_frame, (i[0], i[1]), (i[2], i[3]), (93, 242, 245), thickness=2)
                                        break
                                # max_prob_fire=prob
                                # fire_box = [left,top,right,bottom]

                        elif label=='smoke':
                            background_color = (224, 144, 139)
                            text = "Khói ({:.2f}%)".format(prob)
                            if max_prob_smoke==None or max_prob_smoke<prob:
                                for i in frame_diff:
                                    if overlap(i, [left,top,right,bottom]):
                                        draw_box_flag = True
                                        max_prob_smoke=prob
                                        smoke_box = [left,top,right,bottom]
                                        cv2.rectangle(annotated_frame, (i[0], i[1]), (i[2], i[3]), (93, 242, 245), thickness=2)
                                        break
                                # max_prob_smoke=prob
                                # smoke_box = [left,top,right,bottom]
                        
                        if draw_box_flag:
                            cv2.rectangle(annotated_frame, (left,top), (right,bottom), background_color, thickness=3, lineType=cv2.LINE_AA)
                                
                            # Tạo một ảnh PIL từ hình ảnh Numpy
                            pil_image = Image.fromarray(annotated_frame)

                            # Tạo đối tượng vẽ trên ảnh PIL
                            draw = ImageDraw.Draw(pil_image)
                            
                            # Vẽ nền cho text 
                            text_width, text_height = draw.textsize(text, font=font)
                            t_left = int(result[0])-2
                            t_top = int(result[1])-text_height-3
                            rectangle_position = (t_left, t_top, t_left + text_width, t_top + text_height)
                            draw.rectangle(rectangle_position, fill=background_color)
                            
                            text_position = (t_left, t_top)
                            # Vẽ văn bản màu đỏ
                            draw.text(text_position, text, font=font, fill=font_color)
                            # Chuyển đổi ảnh PIL thành ảnh Numpy
                            annotated_frame = np.array(pil_image)
                
                #-----------------------------------------------------------------------
                
                
                #-----------------------------------------------------------------------
                
                # Nếu có lửa mà không có người
                if max_prob_fire is not None and no_person:

                    if result_queue(q_fire[CC], 1):         
                        # text = 'Có cháy' + " ({:.2f}%)".format(max_prob_fire)
                        text = 'Có cháy'
                        title = "Cảnh báo cháy"
                        # print(text)
                        if send==1:
                            push_alert(fcm_list=fcm_list[CC], title=title, body=text, rtsp=url[CC], camera_id=camera_id[CC])# + formatted_time_ntf)
                        if send==0:
                            push_alert(fcm_list=fcm_dat123(), title=title, body=text, rtsp=url[CC], camera_id=camera_id[CC])
                        save_ntf_img(annotated_frame, 
                                    camera_id=get_camera_id(url[CC]), 
                                    title=title, 
                                    body=text,
                                    notification_type='Fire',
                                    formatted_time=formatted_time,
                                    send=send) # Lưu lại thông tin cảnh báo

                # Nếu không có lửa
                elif max_prob_fire is None and no_person:
                    result_queue(q_fire[CC], 0)

                #---------------------------------------------------------------------------------------
                # Nếu có khói mà không có người
                if max_prob_smoke is not None and no_person:

                    if result_queue(q_smoke[CC], 1):
                        # text = 'Có khói' + " ({:.2f}%)".format(max_prob_smoke)
                        text = 'Có khói'
                        title = "Cảnh báo cháy"
                        # print(text)
                        if send==1:
                            push_alert(fcm_list=fcm_list[CC], title=title, body=text, rtsp=url[CC], camera_id=camera_id[CC])# + formatted_time_ntf)
                        if send==0:
                            push_alert(fcm_list=fcm_dat123(), title=title, body=text, rtsp=url[CC], camera_id=camera_id[CC])
                        save_ntf_img(annotated_frame, 
                                    camera_id=get_camera_id(url[CC]), 
                                    title=title, 
                                    body=text,
                                    notification_type='Fire',
                                    formatted_time=formatted_time,
                                    send=send) # Lưu lại thông tin cảnh báo
                            
                    
                # Nếu không có khói
                elif max_prob_smoke is None and no_person:
                    result_queue(q_smoke[CC], 0)
                                
                #===========================================================================================#

                # Hiện FPS
                cv2.putText(annotated_frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)

                cv2.imshow('Fire - ' + camera_name[CC], 
                            cv2.resize(annotated_frame, (int((annotated_frame.shape[1])*scale),int((annotated_frame.shape[0])*scale))))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(e)
        for CC in range(len(url)):
            fresh[CC].release()
            
    for CC in range(len(url)):
        fresh[CC].release()
        # cap.release()
    cv2.destroyAllWindows()
    
# main()
while True:
    try:
        main()
    except:
        print("Lỗi, Khởi động lại !!!")
        pass
