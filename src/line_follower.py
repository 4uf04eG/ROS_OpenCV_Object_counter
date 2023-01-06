import rospy
from geometry_msgs.msg import Twist


class LineFollower:
    def __init__(self, detector):
        self.detector = detector

        self.velocity = Twist()
        self.velocity.linear.x = 0.12

        self.last_error = 0

        self.velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        rospy.on_shutdown(self.stop)

    def follow_road(self):
        loop_rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            center = self.detector.center

            if center is None:
                continue

            self.rotate(center)
            loop_rate.sleep()

    def rotate(self, center: float):
        error = center - 500
        Kp = 0.002
        Kd = 0.006

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        self.velocity.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        self.velocity_publisher.publish(self.velocity)

    def stop(self):
        self.velocity_publisher.publish(Twist())
