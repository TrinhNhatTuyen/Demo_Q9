import cv2, os
import numpy as np

def inside_the_box(point_test, points, img_w=1920, img_h=1080):
    """
    Checks if a point lies inside a specified rectangular box.

    Args:
        point_test (array): Coordinates of the point to be tested, provided as pixel coordinates ON THE IMAGE.
        points (dict): Coordinates of the four corners of the box, provided as pixel coordinates ON THE IMAGE.
        img_w (int, optional): Width of the image. Defaults to 1920.
        img_h (int, optional): Height of the image. Defaults to 1080.

    Returns:
        bool: True if the point is inside the box, False if it's on or outside the box.
    """
    
    A_Oxy = np.array([points['A'][0], img_h - points['A'][1]])
    B_Oxy = np.array([points['B'][0], img_h - points['B'][1]])
    C_Oxy = np.array([points['C'][0], img_h - points['C'][1]])
    D_Oxy = np.array([points['D'][0], img_h - points['D'][1]])
    # Point needs to be checked if it's in the box or not
    P_Oxy = np.array([point_test[0] , img_h - point_test[1]])
    for A, B in [[A_Oxy,B_Oxy],[B_Oxy,C_Oxy],[C_Oxy,D_Oxy],[D_Oxy,A_Oxy]]:
        AB = B - A
        AP = P_Oxy - A
        result = np.cross(AB, AP)
        if result >= 0:
            return False        
    return True

def draw_box_on_image(image_path, points, output_path=None, scale=1.0):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)

    # Vẽ hình chữ nhật trên ảnh
    for p1, p2 in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]:
        pt1 = tuple(points[p1])
        pt2 = tuple(points[p2])
        cv2.line(image, pt1, pt2, (0, 0, 255), 5)  # Màu xanh lá cây, độ rộng đường viền là 2 pixel

    # Scale ảnh nếu cần
    if scale != 1.0:
        height, width = image.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        image = cv2.resize(image, (new_width, new_height))

    # Hiển thị ảnh và kiểm tra phím nhấn
    while True:
        cv2.imshow('Image with Rectangle', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # cv2.imshow('Image with Rectangle', image)
        
    # Lưu ảnh với hình chữ nhật đã vẽ (nếu cần)
    if output_path:
        cv2.imwrite(output_path, image)

    cv2.destroyAllWindows()

# points = {
#     'A': [922, 160],
#     'B': [1000, 160],
#     'C': [1000, 233],
#     'D': [922, 233],
# }
# Cũ
# points = {
#     'A': [1230, 600],
#     'B': [1320, 710],
#     'C': [1230, 880],
#     'D': [1145, 770],
# }
# {"L0": {"A": [630, 315], "B": [1390, 315], "C": [1390, 485], "D": [630, 485]}}
# points = {
#     'A': [630, 315],
#     'B': [1390, 315],
#     'C': [1390, 485],
#     'D': [630, 485],
# }

# directory = "test/hinh"
# draw_box_on_image(os.path.join(directory, 'Cam trong.jpg'), points, scale=0.5)

# Lặp qua tất cả các tệp trong thư mục
# for filename in os.listdir(directory):
#     draw_box_on_image(os.path.join(directory, filename), points, scale=0.5)  


# point_test1 = np.array([500, 380])
# point_test2 = np.array([700, 80])
# point_test3 = np.array([600, 130])

# print(inside_the_box(np.array([800, 580]), points))
# print(inside_the_box(point_test2, points))
# print(inside_the_box(point_test3, points))