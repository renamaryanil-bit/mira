import cv2
import numpy as np
import math
#rena added
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


VIDEO_PATH = "ellipsoid_bucket.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    exit(1)

def get_ellipse_rim_points(cx, cy, A, B, angle_deg):
    angle = np.deg2rad(angle_deg)
    a = A / 2
    b = B / 2
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    ux = cos_t
    uy = sin_t
    vx = -sin_t
    vy = cos_t
    right = (cx + a * ux, cy + a * uy)
    left  = (cx - a * ux, cy - a * uy)
    top   = (cx + b * vx, cy + b * vy)
    bottom= (cx - b * vx, cy - b * vy)
    return np.array([top, left, bottom, right], dtype=np.float32)

def get_camera_matrix(w, h):
    fx = 800
    fy = 800
    cx = w / 2
    cy = h / 2
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.6
EDGE_MARGIN = 40
RIM_BAND_RATIO = 0.12

#rena added
class BucketPosePublisher(Node):

    def __init__(self):
        super().__init__('bucket_pose_publisher')

        self.publisher_ = self.create_publisher(
            PoseStamped,
            'ellipsoid_pose',
            10
        )

#rena added
rclpy.init()
node = BucketPosePublisher()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(blur, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    MIN_AREA = MIN_AREA_RATIO * h * w
    MAX_AREA = MAX_AREA_RATIO * h * w

    best_score = 0
    best_ellipse = None

    for cnt in contours:
        if len(cnt) < 40:
            continue

        cnt_area = cv2.contourArea(cnt)
        if cnt_area < MIN_AREA or cnt_area > MAX_AREA:
            continue

        try:
            (cx, cy), (A, B), angle = cv2.fitEllipse(cnt)
        except:
            continue

        if cx < EDGE_MARGIN or cy < EDGE_MARGIN or cx > w-EDGE_MARGIN or cy > h-EDGE_MARGIN:
            continue

        a = max(A, B) / 2
        b = min(A, B) / 2
        if a <= 0 or b <= 0:
            continue

        rim_points = []
        inner_limit = 1.0 - RIM_BAND_RATIO

        for p in cnt:
            px, py = p[0]
            dxn = (px - cx) / a
            dyn = (py - cy) / b
            r = math.sqrt(dxn*dxn + dyn*dyn)
            if inner_limit < r <= 1.05:
                rim_points.append([px, py])

        if len(rim_points) < 30:
            continue

        rim_points = np.array(rim_points).reshape(-1, 1, 2)

        try:
            (cx2, cy2), (A2, B2), angle2 = cv2.fitEllipse(rim_points)
        except:
            continue

        if max(A2, B2) / min(A2, B2) > 1.6:
            continue

        score = cv2.contourArea(rim_points)
        if score > best_score:
            best_score = score
            best_ellipse = (cx2, cy2, A2, B2, angle2)

    if best_ellipse is not None:
        cx2, cy2, A2, B2, angle2 = best_ellipse

        cv2.ellipse(
            frame,
            ((cx2, cy2), (A2, B2), angle2),
            (0, 255, 0),
            2
        )

        image_points = get_ellipse_rim_points(cx2, cy2, A2, B2, angle2)

        for (px, py) in image_points:
            cv2.circle(frame, (int(px), int(py)), 6, (0, 255, 255), -1)

        object_points = np.array([
            (0, -1, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (1, 0, 0)
        ], dtype=np.float32)

        camera_matrix = get_camera_matrix(w, h)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if success:
            tx, ty, tz = tvec.flatten()
            #print(f"tvec: tx={tx:.2f}, ty={ty:.2f}, tz={tz:.2f}")
            #rena
            node.get_logger().info(
                f"Publishing pose: x={tx:.2f}, y={ty:.2f}, z={tz:.2f}"
            )
            #rena added
            rot_matrix, _ = cv2.Rodrigues(rvec)
            r = R.from_matrix(rot_matrix)
            quat = r.as_quat()

            pose_msg = PoseStamped()

            pose_msg.header.stamp = node.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_link"

            pose_msg.pose.position.x = float(tx)
            pose_msg.pose.position.y = float(ty)
            pose_msg.pose.position.z = float(tz)

            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])

            node.publisher_.publish(pose_msg)


    #rena added
    rclpy.spin_once(node, timeout_sec=0.0)

    cv2.imshow("Bucket solvePnP", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#rena added
node.destroy_node()
rclpy.shutdown()
