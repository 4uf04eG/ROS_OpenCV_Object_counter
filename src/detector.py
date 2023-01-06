import cv2
import cv_bridge
import numpy
import rospy
from sensor_msgs.msg import Image

import object_tracker


# noinspection PyTupleAssignmentBalance
class Detector:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.object_tracker = object_tracker.ObjectTracker(70)

        self.center = None

        self.image_subscriber = rospy.Subscriber("camera/image", Image, self.image_callback)

    def image_callback(self, image: Image):
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        road_image = self.image_projection(cv_image)
        target_image = self.process_image(road_image, cv_image)

        cv2.putText(cv_image, f"Found circles: {self.object_tracker.total_objects_num}",
                    (cv_image.shape[1] - 400, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        cv2.imshow("Target objects image", cv_image)
        cv2.imshow("Road image", target_image)
        cv2.waitKey(1)

    def process_image(self, road_image: Image, source_image: Image) -> Image:
        hsv_target = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
        target_mask = self.select_target_objects(hsv_target)
        target_mask = cv2.erode(target_mask, None, iterations=2)
        target_mask = cv2.dilate(target_mask, None, iterations=2)

        hsv_road = cv2.cvtColor(road_image, cv2.COLOR_BGR2HSV)
        left_line_mask = self.select_left_line(hsv_road)
        # left_line_mask = cv2.dilate(left_line_mask, None, iterations=2)
        left_line_mask = cv2.blur(left_line_mask, (9, 9), 3)

        right_line_mask = self.select_right_line(hsv_road)
        # right_line_mask = cv2.dilate(right_line_mask, None, iterations=2)
        right_line_mask = cv2.blur(right_line_mask, (9, 9), 3)

        self.detect_target_objects(source_image, cv2.blur(target_mask, (9, 9), 3))
        self.detect_road(road_image, left_line_mask, right_line_mask)

        return road_image

    @staticmethod
    def image_projection(image: Image) -> Image:
        height, width, _ = image.shape

        pts_src = numpy.array([[
            [width / 2 - width / 6, height / 2 + height / 13],  # Top left
            [width / 2 + width / 6, height / 2 + height / 13],  # Top right
            [width - width / 5, height],                        # Bottom right
            [width / 5, height],                                # Bottom left
        ]], numpy.int32)
        pts_dst = numpy.array([[200, 0], [800, 0], [800, 600], [200, 600]])

        h, _ = cv2.findHomography(pts_src, pts_dst)

        return cv2.warpPerspective(image, h, (1000, 600))

    @staticmethod
    def select_target_objects(image: Image) -> Image:
        lower_purple = numpy.array([140, 10, 0])
        upper_purple = numpy.array([160, 255, 255])

        return cv2.inRange(image, lower_purple, upper_purple)

    @staticmethod
    def select_left_line(image: Image) -> Image:
        lower_purple = numpy.array([22, 93, 0])
        upper_purple = numpy.array([45, 255, 255])

        return cv2.inRange(image, lower_purple, upper_purple)

    @staticmethod
    def select_right_line(image: Image) -> Image:
        lower_purple = numpy.array([0, 0, 180])
        upper_purple = numpy.array([25, 36, 255])

        return cv2.inRange(image, lower_purple, upper_purple)

    def detect_target_objects(self, image: Image, mask: Image):
        detected_circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 0.9, 50,
                                            param1=100, param2=55, minRadius=0, maxRadius=500)
        self.object_tracker.update_objects(detected_circles)

        for (object_id, centroid) in self.object_tracker.on_screen.items():
            text = "ID {}".format(object_id)
            cv2.putText(image, text, (int(centroid[0]) - 15, int(centroid[1] + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), thickness=3)

    def detect_road(self, image: Image, mask_left: numpy.array, mask_right: numpy.array):
        fraction_left = numpy.count_nonzero(mask_left)
        fraction_right = numpy.count_nonzero(mask_right)

        left = None
        right = None

        if fraction_left > 3000:
            left = self.detect_line_segment(mask_left, 'left')
        if fraction_right > 3000:
            right = self.detect_line_segment(mask_right, 'right')

        self.display_road(image, left, right)

    @staticmethod
    def detect_line_segment(image: numpy.array, left_or_right: str) -> numpy.array:
        histogram = numpy.sum(image, axis=0)
        center = numpy.int(histogram.shape[0] / 2)

        if left_or_right == 'left':
            lane_base = numpy.argmax(histogram[:center])
        elif left_or_right == 'right':
            lane_base = numpy.argmax(histogram[center:]) + center
        else:
            return

        windows_num = 40  # Number of sliding windows
        window_height = numpy.int(image.shape[0] / windows_num)

        nonzero = image.nonzero()
        nonzero_y = numpy.array(nonzero[0])
        nonzero_x = numpy.array(nonzero[1])

        x_current = lane_base
        window_width = 100
        pixel_threshold = 10  # Minimum number of pixels
        lane_indices = []

        for window in range(windows_num):
            # Identifying window boundaries in x and y
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_x_low = x_current - window_width
            win_x_high = x_current + window_width

            # Identifying the nonzero pixels in x and y within the window
            good_lane_indices = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                 (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)).nonzero()[0]
            lane_indices.append(good_lane_indices)

            if len(good_lane_indices) > pixel_threshold:
                x_current = numpy.int(numpy.mean(nonzero_x[good_lane_indices]))

        lane_indices = numpy.concatenate(lane_indices)
        x = nonzero_x[lane_indices]
        y = nonzero_y[lane_indices]

        if len(x) > 0 and len(y) > 0:
            lane_fit = numpy.polyfit(y, x, 2)
        else:
            return

        plot_y = numpy.linspace(0, image.shape[0] - 1, image.shape[0])

        return lane_fit[0] * plot_y ** 2 + lane_fit[1] * plot_y + lane_fit[2]

    def display_road(self, image: Image, left: numpy.array, right: numpy.array):
        plot_y = numpy.linspace(0, image.shape[0] - 1, image.shape[0])

        if left is not None:
            left_line_pts = numpy.array([numpy.transpose(numpy.vstack([left, plot_y]))])
        if right is not None:
            right_line_pts = numpy.array([numpy.flipud(numpy.transpose(numpy.vstack([right, plot_y])))])

        if left is not None and right is not None:
            center = numpy.mean([left, right], axis=0)
            pts = numpy.hstack((left_line_pts, right_line_pts))
            center_line = numpy.array([numpy.transpose(numpy.vstack([center, plot_y]))])
            cv2.fillPoly(image, numpy.int_([pts]), (0, 255, 0))
        elif left is not None and right is None:  # Only left road found
            center = numpy.add(left, 450)
            center_line = numpy.array([numpy.transpose(numpy.vstack([center, plot_y]))])
        elif left is None and right is not None:  # Only right road found
            center = numpy.subtract(right, 450)
            center_line = numpy.array([numpy.transpose(numpy.vstack([center, plot_y]))])
        else:
            return

        self.center = center.item(500)
        cv2.polylines(image, numpy.int_([center_line]), isClosed=False, color=(0, 0, 255), thickness=12)
