import rclpy
from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class YoloNode(Node):
    """ Use Yolo to identify scene objects """

    def __init__(self):
        super().__init__("pose")
        self.bridge = CvBridge()
        self.model = YOLO("yolo11n-pose.pt")
        self.create_subscription(Image, 'image', self.yolo_callback, 10)
        self.pub = self.create_publisher(Image, 'new_image', 10)

    def yolo_callback(self, image):
        """Draw a circle on the subscribed image and republish it to new_image."""
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        results = self.model(cv_image)
        frame = results[0].plot()
        new_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(new_msg)

def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()
