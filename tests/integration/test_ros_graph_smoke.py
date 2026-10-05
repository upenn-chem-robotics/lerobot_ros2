"""Hardware-free ROS 2 transport smoke test for the final container image."""

from __future__ import annotations

import threading
import time
import uuid

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


def test_ros_publish_subscribe_round_trip() -> None:
    """Prove discovery, serialization, delivery, spinning, and clean shutdown."""
    rclpy.init(args=None)
    suffix = uuid.uuid4().hex
    publisher_node = rclpy.create_node(f"lerobot_ros2_smoke_pub_{suffix}")
    subscriber_node = rclpy.create_node(f"lerobot_ros2_smoke_sub_{suffix}")
    topic = f"/lerobot_ros2/smoke/run_{suffix}"
    payload = f"lerobot-ros2-smoke-{suffix}"
    received = threading.Event()
    received_payloads: list[str] = []

    def callback(message: String) -> None:
        received_payloads.append(message.data)
        if message.data == payload:
            received.set()

    subscription = subscriber_node.create_subscription(String, topic, callback, 10)
    publisher = publisher_node.create_publisher(String, topic, 10)
    executor = SingleThreadedExecutor()
    executor.add_node(publisher_node)
    executor.add_node(subscriber_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        deadline = time.monotonic() + 15.0
        message = String(data=payload)
        while time.monotonic() < deadline and not received.is_set():
            publisher.publish(message)
            received.wait(0.1)
        assert received.is_set(), f"timed out waiting for {topic}; received={received_payloads!r}"
    finally:
        executor.shutdown(timeout_sec=5.0)
        spin_thread.join(timeout=5.0)
        publisher_node.destroy_publisher(publisher)
        subscriber_node.destroy_subscription(subscription)
        publisher_node.destroy_node()
        subscriber_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
