
import rclpy
from rclpy.node import Node


class Tracker(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.get_logger().info("Tracker initialised")

    def publish_frame(self):
            self.publisher.publish(msg)

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    node = Tracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
