'''
패킷 분할 없이 한번에 보내는 대신에 해상도는 원본의 1/2로 줄음(320 x 240)
'''



import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import socket

class UdpImageSender(Node):
    def __init__(self):
        super().__init__('udp_image_sender')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = ('192.168.0.4', 5005)  #서버 주소로 변환

        self.subscription = self.create_subscription(
            Image,
            '/ascamera/camera_publisher/rgb0/image',
            self.callback,
            10
        )
        self.get_logger().info(f"Sending to {self.server_addr}")

    def callback(self, msg):
        # YUYV -> BGR 변환
        yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 2))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)

        # 필요시 리사이즈 (대역폭 절약, 패킷 크기 여유 확보)
        bgr = cv2.resize(bgr, (320, 240))

        ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            return

        data = encoded.tobytes()
        if len(data) > 65000:
            self.get_logger().warn(f"Frame too large ({len(data)} bytes), skipping")
            return

        self.sock.sendto(data, self.server_addr)

def main():
    rclpy.init()
    node = UdpImageSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()