import numpy
from scipy.spatial import distance as dist


class ObjectTracker:
    def __init__(self, disappearance_delay):
        self.next_id = 0
        self.total_objects_num = 0  # Keeps overall number of found objects

        self.on_screen = dict()
        self.disappeared = dict()

        self.delay = disappearance_delay

    def add_object(self, centroid):
        self.on_screen[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        self.total_objects_num += 1

    def remove_object(self, object_id: int):
        del self.on_screen[object_id]
        del self.disappeared[object_id]

    def update_objects(self, on_screen_objects):
        if on_screen_objects is None:
            for current_id in list(self.disappeared.keys()):
                # Incrementing number of frames since object disappeared
                self.disappeared[current_id] += 1

                # Object lost. Stopping tracking it
                if self.disappeared[current_id] > self.delay:
                    self.remove_object(current_id)

            return

        on_screen_centroids = []

        for (x, y, r) in numpy.array(on_screen_objects[0, :]):
            on_screen_centroids.append((x, y))

        if len(self.on_screen) == 0:
            for centroid in on_screen_centroids:
                self.add_object(centroid)

            return

        self.update_existing_objects(on_screen_centroids)

    def update_existing_objects(self, on_screen_centroids: numpy.array):
        all_ids = list(self.on_screen.keys())
        all_centroids = numpy.array(list(self.on_screen.values()))

        # Distances between each current centroid and already found one
        distances = dist.cdist(all_centroids, on_screen_centroids)
        min_rows = distances.min(axis=1).argsort()
        min_columns = distances.argmin(axis=1)[min_rows]

        used_rows = set()
        used_columns = set()

        for (row, column) in zip(min_rows, min_columns):
            if row in used_rows or column in used_columns:
                continue

            object_id = all_ids[row]
            self.on_screen[object_id] = on_screen_centroids[column]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_columns.add(column)

        unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
        unused_columns = set(range(0, distances.shape[1])).difference(used_columns)

        # If number of input objects lower than number of objects on screen,
        # checking which ones disappeared
        if distances.shape[0] >= distances.shape[1]:
            for row in unused_rows:
                object_id = all_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.delay:
                    self.remove_object(object_id)
        else:
            for column in unused_columns:
                self.add_object(on_screen_centroids[column])
