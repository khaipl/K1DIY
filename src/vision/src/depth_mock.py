import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthMockNode(Node):
    def __init__(self):
        super().__init__('depth_mock_node')
        # Listen to your live laptop camera
        self.subscription = self.create_subscription(
            Image,
            '/booster_camera_bridge/image_left_raw',
            self.image_callback,
            10)
        
        # Publish to the K1's expected depth topic
        self.publisher_ = self.create_publisher(
            Image,
            '/booster_camera_bridge/StereoNetNode/stereonet_depth',
            10)
        
        self.bridge = CvBridge()
        self.get_logger().info('Depth Mock Node active! Faking stereo depth for vision_node...')

    def image_callback(self, msg):
        # Create a blank 16-bit depth image (simulating a flat wall 1000mm away)
        depth_image = np.full((msg.height, msg.width), 1000, dtype=np.uint16)
        
        # Convert to ROS message with MONO16 encoding (what the C++ node expects)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="mono16")
        
        # CRITICAL: Copy the exact timestamp from the webcam so the C++ synchronizer accepts it
        depth_msg.header = msg.header 
        
        self.publisher_.publish(depth_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthMockNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()