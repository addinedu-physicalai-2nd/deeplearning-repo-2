
'''
패킷 분할을 통해 원본 해상도로 이미지 송신
'''



import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import socket

MAX_PACKET_SIZE = 60000  # 여유 두고 60KB로 자름
frame_id_counter = 0

class UdpImageSender(Node):
    def __init__(self):
        super().__init__('udp_image_sender')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = ('192.168.0.4', 5005)

        self.subscription = self.create_subscription(
            Image,
            '/ascamera/camera_publisher/rgb0/image',
            self.callback,
            10
        )
        self.get_logger().info(f"Sending to {self.server_addr}")
    def callback(self, msg):
        global frame_id_counter

        yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 2))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
        # 리사이즈 제거! 640x480 원본 그대로

        ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            return

        data = encoded.tobytes()
        frame_id = frame_id_counter
        frame_id_counter = (frame_id_counter + 1) % 65536

        chunks = [data[i:i+MAX_PACKET_SIZE] for i in range(0, len(data), MAX_PACKET_SIZE)]
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            # 헤더: frame_id(2B) + chunk_idx(1B) + total_chunks(1B)
            header = struct.pack('!HBB', frame_id, idx, total_chunks)
            self.sock.sendto(header + chunk, self.server_addr)


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