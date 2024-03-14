import cv2, torch, requests, base64, json, datetime, time, pyodbc, threading, firebase_admin
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from overstepframe import FreshestFrame
from firebase_admin import credentials, messaging
from ultralytics.yolo.utils.callbacks.tensorboard import callbacks
#------------------------------------ LOAD MODEL ------------------------------------
from ultralytics import YOLO
# pose_model = YOLO('YOLOv8n-pose.pt')
# pose_model = YOLO('YOLOv8n.pt')
# fire_model = YOLO('set_hsv_warmup_epoch_3.pt')
cred = credentials.Certificate('ngocvinaai-firebase-adminsdk-57cev-b988d1a956.json')
firebase_admin.initialize_app(cred)

#------------------------------------ PARAMETERS ------------------------------------

firesmoke_conf = 0.3 # Ngưỡng phát hiện khói và lửa
humanpose_conf = 0.6 # Ngưỡng phát hiện người
queue_len = 20       # Độ dài hàng đợi
n = 15               # Số frame phát hiện khói lửa tối thiểu (để push thông báo)

device = 0 if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

ip = '125.253.117.120'
# ip = '112.78.15.5'
port = '5001'
# Scale khung hình để xem
scale = 0.5
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

def save_ntf_img(frame, camera_id, title, body, notification_type, formatted_time):
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

def overlap(box1, box2, ratio=0.1):
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

def calculate_diff(box, frame1, frame2, ratio=0.2):
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

def run_1cam(url, lockpicking_area, climbing_area, 
            fcm_list, camera_id, related_camera_id, 
            camera_name, homeid, lockid, label_mapping,
            frame, cnt, first_frame, second_frame, 
            previousframe, t_oldframe, None_frame,
            lastest_detected_face, q_lockpicking, q_climbing, q_knownperson,
            q_fire, q_smoke, q_fire_box, q_smoke_box):

    #===========================================================================================================#
    pose_model = YOLO('YOLOv8n.pt')
    fire_model = YOLO('set_hsv_warmup_epoch_3.pt')
    fresh = FreshestFrame(cv2.VideoCapture(url))
    
    t_oldframe = 0
    while True:
        # Release FreshestFrame objects every 10 minutes
        t = datetime.datetime.now()
        if (t.minute % 50 == 0) and t.second<2:
            raise Exception("Restarting...")
        
        cnt, frame = fresh.read(seqnumber=cnt+1, timeout=5)
        while frame is None:
            cnt, frame = fresh.read(seqnumber=cnt+1, timeout=5)
        # frame = cv2.flip(frame, 0)

        if not cnt:
            break
        # frame = cv2.resize(frame, (1920,1080))
        if not cnt:
            print(f"Timeout, can't read new frame of cam {CC}!")
            raise Exception()
        
        if None_frame>5:
            print("Cannot read frame from camera!")
            raise Exception()
        
        # dùng để tính toán FPS
        timer = time.time()
        # if t_oldframe is None:
        #     t_oldframe = timer
        
        # gọi lỗi nếu k đọc được frame từ camera
        if first_frame is None:
            first_frame = frame.copy()
            None_frame+=1
            continue
        
        try:
            frame = cv2.resize(frame, (1920,1080))
        except:
            pass
        
        current_time = datetime.datetime.now()
        
        fps = 1/(timer-t_oldframe)
        t_oldframe = timer
        
        # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # first_frame = cv2.threshold(first_frame, thresh, 255, cv2.THRESH_BINARY)[1]

        # second_frame = frame.copy()
        annotated_frame = frame.copy()
        

        #=================================================================================================#
        
        
        max_prob_fire = None
        max_prob_smoke = None
        max_prob_person = None
        no_person = True
        formatted_time = current_time.strftime("%d-%m-%Y %Hh%M'%S\"")
        formatted_time_ntf = ' lúc ' + current_time.strftime("%Hh%M'%S\" %d-%m-%Y")
        
        copy_frame = frame.copy()
        
        # Nhận diện con người
        human_results = pose_model.predict(source=copy_frame, conf=humanpose_conf, device=device, save=False, classes=[0])
        
        if previousframe is None:
            previousframe = frame.copy()
        
        # annotated_frame = human_results[0].plot(boxes=False)
        
        # Nếu có người
        if len(human_results[0].boxes.data)>0:
            no_person = False
            result_queue(q_fire, 2)
            result_queue(q_smoke, 2)
        else:    
            # Nhận diện lửa khói
            fire_results = fire_model.predict(source=copy_frame, conf=firesmoke_conf, device=device, save=False)
            
            # Vẽ box cho khói-lửa
            for result in fire_results[0].boxes.data:
                # Thiết lập font chữ
                font_path = 'arial.ttf'
                font_size = 40
                # catdog_overlap = None
                font_color = (255, 255, 255)  # Màu trắng (B, G, R)
                font = ImageFont.truetype(font_path, font_size)                                                         

                # Vẽ khung chữ nhật
                left = int(result[0])
                top = int(result[1])
                right = int(result[2])
                bottom = int(result[3])
                prob = result[4]*100
                label = label_mapping.get(int(result[5]), "unknown")

                """calculate_diff để tính toán xem vùng trong box 
                có khác biệt đáng kể gì so với 1 khoảng tgian trước đó k"""
                
                if label=='fire':
                    background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                    text = "Lửa ({:.2f}%)".format(prob)
                    if max_prob_fire==None or max_prob_fire<prob:
                        max_prob_fire=prob
                        fire_box = [left,top,right,bottom]

                elif label=='smoke':
                    background_color = (224, 144, 139)
                    text = "Khói ({:.2f}%)".format(prob)
                    if max_prob_smoke==None or max_prob_smoke<prob:
                        max_prob_smoke=prob
                        smoke_box = [left,top,right,bottom]
                
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
            
        # Nếu có lửa mà không có người
        if max_prob_fire is not None and no_person:
            # Nếu k có obj nào trong hàng đợi
            if len([i for i, value in enumerate(q_fire_box.queue) if isinstance(value, list)])==0:
                if q_fire_box.full():
                    q_fire_box.get()
                q_fire_box.put(fire_box)
                result_queue(q_fire, 1)
            # Hàng đợi có obj -> Kiểm tra overlap
            else:
                overlap_flag = False
                for i in [index for index, value in enumerate(q_fire_box.queue) if isinstance(value, list)]:
                    dequeue_fire_box = q_fire_box.queue
                    # Nếu Overlap
                    if overlap(dequeue_fire_box[i], fire_box) and calculate_diff(fire_box, copy_frame, previousframe):
                        # Đưa fire_box vào q_fire_box
                        if q_fire_box.full():
                            q_fire_box.get()
                        q_fire_box.put(fire_box)
                        
                        # Kiểm tra hàng đợi, đủ thì cảnh báo
                        if result_queue(q_fire, 1):
                            
                            # text = 'Có cháy' + " ({:.2f}%)".format(max_prob_fire)
                            text = 'Có cháy'
                            title = "Cảnh báo cháy"
                            # push_alert(fcm_list=fcm_list, title=title, body=text)# + formatted_time_ntf)
                            # push_alert(fcm_list=fcm_dat123(), title=title, body=text)
                            save_ntf_img(annotated_frame, 
                                        camera_id=get_camera_id(url), 
                                        title=title, 
                                        body=text,
                                        notification_type='Fire',
                                        formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                        overlap_flag = True
                        break

                # Nếu KHÔNG Overlap
                if not overlap_flag:
                    result_queue(q_fire, 0)
                    if q_fire_box.full():
                        q_fire_box.get()
                    q_fire_box.put(fire_box)            # Vẫn put "fire_box" vào
                        
        # Nếu không có lửa
        elif max_prob_fire is None:
            result_queue(q_fire, 0)
            if q_fire_box.full():
                q_fire_box.get()
            q_fire_box.put(0)
        #---------------------------------------------------------------------------------------
        # Nếu có khói mà không có người
        if max_prob_smoke is not None and no_person:
            # Nếu k có obj nào trong hàng đợi
            if len([i for i, value in enumerate(q_smoke_box.queue) if isinstance(value, list)])==0:
                if q_smoke_box.full():
                    q_smoke_box.get()
                q_smoke_box.put(smoke_box)
                result_queue(q_smoke, 1)
            # Hàng đợi có obj -> Kiểm tra overlap
            else:
                overlap_flag = False
                for i in [index for index, value in enumerate(q_smoke_box.queue) if isinstance(value, list)]:
                    dequeue_smoke_box = q_smoke_box.queue
                    # Nếu Overlap
                    if overlap(dequeue_smoke_box[i], smoke_box) and calculate_diff(smoke_box, copy_frame, previousframe):
                        # Đưa smoke_box vào q_smoke_box
                        if q_smoke_box.full():
                            q_smoke_box.get()
                        q_smoke_box.put(smoke_box)
                        
                        # Kiểm tra hàng đợi, đủ thì cảnh báo
                        if result_queue(q_smoke, 1):
                            
                            # text = 'Có khói' + " ({:.2f}%)".format(max_prob_smoke)
                            text = 'Có khói'
                            title = "Cảnh báo cháy"
                            # push_alert(fcm_list=fcm_list, title=title, body=text)# + formatted_time_ntf)
                            # push_alert(fcm_list=fcm_dat123(), title=title, body=text)
                            save_ntf_img(annotated_frame, 
                                        camera_id=get_camera_id(url), 
                                        title=title, 
                                        body=text,
                                        notification_type='Fire',
                                        formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                        overlap_flag = True
                        break

                # Nếu KHÔNG Overlap
                if not overlap_flag:
                    result_queue(q_smoke, 0)
                    if q_smoke_box.full():
                        q_smoke_box.get()
                    q_smoke_box.put(smoke_box)            # Vẫn put "smoke_box" vào
            
        # Nếu không có khói
        elif len(human_results[0].boxes.data)>0:
            result_queue(q_smoke, 0)
            if q_smoke_box.full():
                q_smoke_box.get()
            q_smoke_box.put(0)
                        
        #===========================================================================================#
        
        if len(human_results[0].boxes.data)==0 and (t.second%10)==0:
            previousframe = frame.copy()
            
        # Hiện FPS
        cv2.putText(annotated_frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)

        cv2.imshow(camera_name, 
                    cv2.resize(annotated_frame, (int((annotated_frame.shape[1])*scale),int((annotated_frame.shape[0])*scale))))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # except Exception as e:
    #     print(e)
    #     for CC in range(len(url)):
    #         fresh.release()
            
    for CC in range(len(url)):
        fresh.release()
        # cap.release()
    cv2.destroyAllWindows()

def main():
    cam_data = get_camera_data()
    # cam_data = [cam_data[9]] #ff
    # cam_data = [cam_data[8], cam_data[9],cam_data[10]] #ff
    cam_data = [
        # cam_data[1], 
        cam_data[9], 
        cam_data[10], 
        # cam_data[11],
        ] #ff
    #------------------------------------ Các thông tin của Camera ------------------------------------
    url, lockpicking_area, climbing_area, fcm_list, camera_id, camera_name, homeid, lockid, related_camera_id, task = [], [], [], [], [], [], [], [], [], []
    for cam in cam_data:
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
    previousframe, q_fire, q_smoke, q_fire_box, q_smoke_box = [], [], [], [], []
    for i in range(len(url)):
        # fresh.append(FreshestFrame(cv2.VideoCapture(url[i])))
        # fresh.append(FreshestFrame(cv2.VideoCapture("rtsp://admin:1qazxsw2@192.168.6.64:5580/cam/realmonitor?channel=1&subtype=0&unicast=true")))
        frame.append(object())
        cnt.append(0)
        first_frame.append(None)
        second_frame.append(None)
        previousframe.append(None)
        t_oldframe.append(None)
        None_frame.append(0)
        lastest_detected_face.append(None)
        q_lockpicking.append(Queue(maxsize=queue_len))
        q_climbing.append(Queue(maxsize=queue_len))
        q_knownperson.append(Queue(maxsize=10))
        q_fire.append(Queue(maxsize=queue_len))
        q_smoke.append(Queue(maxsize=queue_len))
        q_fire_box.append(Queue(maxsize=queue_len))
        q_smoke_box.append(Queue(maxsize=queue_len))
    
    #======================= Params =======================#
    label_mapping = {
        0: "fire",
        1: "smoke",
        }
    
    for CC in range(len(url)):
        p = threading.Thread(target=run_1cam, args=(url[CC], lockpicking_area[CC], climbing_area[CC], 
                                    fcm_list[CC], camera_id[CC], related_camera_id[CC], 
                                    camera_name[CC], homeid[CC], lockid[CC], label_mapping,
                                    frame[CC], cnt[CC], first_frame[CC], second_frame[CC], 
                                    previousframe[CC], t_oldframe[CC], None_frame[CC],
                                    lastest_detected_face[CC], q_lockpicking[CC], q_climbing[CC], q_knownperson[CC],
                                    q_fire[CC], q_smoke[CC], q_fire_box[CC], q_smoke_box[CC]))
        p.start()

main()
# while True:
#     try:
#         main()
#     except:
#         print("Lỗi, Khởi động lại !!!")
#         pass
