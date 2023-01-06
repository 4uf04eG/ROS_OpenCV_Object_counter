import cv2
import rospy

import detector
import line_follower

if __name__ == "__main__":
    try:
        rospy.init_node('opencv_counter', anonymous=True)
        detector = detector.Detector()
        line_follower = line_follower.LineFollower(detector)
        line_follower.follow_road()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
