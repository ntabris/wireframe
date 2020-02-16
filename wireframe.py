import cv2
import numpy as np
from math import sin, cos, pi


COLORS = dict(
    black=(0, 0, 0),
    blue=(1, 0, 0),
    green=(0, 1, 0),
    red=(0, 0, 1),
    yellow=(1, 1, 0)
)


class Point:
    def __init__(self, x, y, height=1):
        self.x = x
        self.y = y
        self.height = height
        self._init_camera()

    def _init_camera(self):
        self.is_visible = False
        self.vis_horiz = 0.0
        self.vis_height = 0.0
        self.vis_dist = 0.0

    def update_camera(self, camera):
        self.is_visible, self.vis_horiz = camera.map_to_view(self.x, self.y)
        self.vis_dist = camera.distance(self.x, self.y)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, height={self.height}, is_visible={self.is_visible}, vis_horiz={self.vis_horiz}, vis_dist={self.vis_dist})"


class World:
    def __init__(self):
        self.points = []
        self.connections = []

    def add_wall(self, list_of_points, height=1, closed=False):
        i = len(self.points)

        list_of_points = [Point(*point_coords, height=height) for point_coords in list_of_points]

        self.points.extend(list_of_points)
        new_connections = list(range(i, len(self.points)))
        if closed:
            new_connections.append(i)
        self.connections.append(new_connections)

    def update_camera(self, camera):
        for point in self.points:
            point.update_camera(camera)


class Camera:
    def __init__(self, focal_distance, focal_width, focal_height):
        self.focal_distance = focal_distance
        self.focal_width = focal_width
        self.focal_height = focal_height

    def set(self, x, y, angle):
        self.camera_x, self.camera_y, self.camera_angle = x, y, angle
        self._make_field_line()

    def get_point(self):
        return (self.camera_x, self.camera_y)

    def map_point_to_screen(self, point, screen_height, screen_width):
        height_dist_scale = self.focal_distance/point.vis_dist
        full_height = point.height * self.focal_height

        height = full_height * height_dist_scale

        bottom_offset = (screen_height - (self.focal_height * height_dist_scale)) / 2
        top_offset = screen_height - bottom_offset - height

        x = point.vis_horiz * screen_width
        y = top_offset
        y_ = screen_height - bottom_offset

        return x, y, y_


    @staticmethod
    def do_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Determines whether two lines intersect.

        Line A: (x1, y1) to (x2, y2)
        Line B: (x3, y3) to (x4, y4)

        Returns:
            (True, x, y) for the point of intersection, or (False, 0, 0)
            if the lines don't intersect.
        """
        t = (x1 - x3) * (y3 - y4) - ((y1 - y3) * (x3 - x4))
        t_div = (x1 - x2) * (y3 - y4) - ((y1 - y2) * (x3 - x4))

        if t_div != 0:
            t /= t_div
            p_x, p_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)

            return True, p_x, p_y

        return False, 0, 0

    def _make_field_line(self):
        x_f, y_f = self.angle_to_vector(self.camera_angle, self.focal_distance)
        x_f += self.camera_x
        y_f += self.camera_y

        x_, y_ = self.angle_to_vector(self.camera_angle + pi/2, self.focal_width)

        x1 = x_f - x_
        y1 = y_f - y_

        x2 = x_f + x_
        y2 = y_f + y_

        self.field_line = (x1, y1, x2, y2)

    def map_to_view(self, x, y):
        # Does the line from (x, y) to eye cross the visual field line?
        are_intersecting, inter_x, inter_y = self.do_lines_intersect(
            *self.field_line,
            x, y, self.camera_x, self.camera_y
        )

        # Make sure that the eye isn't between the object and the point
        # of intersection with the visual field line, since this means
        # the object is behind the eye.

        min_x, max_x = min(inter_x, x), max(inter_x, x)
        min_y, max_y = min(inter_y, y), max(inter_y, y)

        intersect_on_ray = not (min_x <= self.camera_x <= max_x
                                and min_y <= self.camera_y <= max_y)

        if not (are_intersecting and intersect_on_ray):
            return False, 0

        # From the point of intersection on the visual field line,
        # find the [0, 1] relative position on the visual field line.

        x1, y1, x2, y2 = self.field_line

        if x2 - x1 != 0:
            field_pos = (inter_x - x1) / (x2 - x1)
        else:
            field_pos = (inter_y - y1) / (y2 - y1)

        return 0 <= field_pos <= 1, field_pos

    def distance(self, x, y):
        return ((x - self.camera_x) ** 2 + (y - self.camera_y) ** 2) ** .5

    @classmethod
    def angle_to_vector(cls, angle, d=1):
        return cos(angle) * d, sin(angle) * d


class Image:
    def __init__(self, w, h):
        self.h = h
        self.w = w
        self.clear()

    def clear(self):
        self.img = np.full((self.h, self.w, 3), 1, dtype="float")

    def draw_circle(self, x, y, color="red", size=3):
        x, y = int(x), int(y)
        self.img = cv2.circle(self.img, (x, y), 1, COLORS[color], 2)

    def draw_line(self, points, color="black"):
        x0, y0, x1, y1 = points
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        self.img = cv2.line(self.img, (x0, y0), (x1, y1), COLORS[color])

    def draw_eye(self, eye, angle):
        self.draw_circle(*eye, "blue")

    def show(self):
        cv2.imshow('image', self.img)
        k = cv2.waitKey(0) & 127

        if k == 0:
            return "up"
        elif k == 1:
            return "down"
        elif k == 2:
            return "left"
        elif k == 3:
            return "right"
        elif k == ord("q"):
            return "quit"


def make_image(img, camera, world):

    img.clear()

    visual_field_points = camera.field_line

    world.update_camera(camera)

    # Draw front view

    visual_field_points = []
    for i, point in enumerate(world.points):
        x, y, y_ = camera.map_point_to_screen(point, screen_height, screen_width)

        visual_field_points.append((point.is_visible, x, y, y_))

    # draw vertical lines for wall points
    for i, (is_vis, x, y, y_) in enumerate(visual_field_points):
        if is_vis:
            img.draw_line((x, y, x, y_))

    # draw top and bottom of connected wall points
    for connection_list in world.connections:
        for from_idx, to_idx in zip(connection_list[:-1], connection_list[1:]):
            if world.points[from_idx].is_visible or world.points[to_idx].is_visible:
                _, x0, y0, y0_ = visual_field_points[from_idx]
                _, x1, y1, y1_ = visual_field_points[to_idx]

                img.draw_line((x0, y0, x1, y1))
                img.draw_line((x0, y0_, x1, y1_))

    return img.show()


if __name__ == "__main__":

    screen_width = 500
    screen_height = 500
    focal_dist = 40
    focal_width = 35
    focal_height = screen_height * 3 / 5

    camera_x = 140
    camera_y = 160
    degrees = -95

    img = Image(screen_width, screen_height)
    camera = Camera(focal_dist, focal_width, focal_height)
    world = World()

    world.add_wall([
        (20, 120),
        (60, 120),
        (60, 60),
        (100, 60),
        (100, 20),
    ])

    world.add_wall([
        (160, 20),
        (160, 60),
        (200, 60),
        (200, 120),
        (240, 120),
    ])

    world.add_wall([
        (20, 160),
        (60, 160),
        (60, 200),
        (60, 240),
        (100, 280),
        (140, 280),
    ])

    world.add_wall([
        (100, 100),
        (105, 100),
        (105, 95),
        (100, 95),
    ], closed=True, height=.25)

    world.add_wall([
        (160, 100),
        (165, 100),
        (165, 95),
        (160, 95)
    ], closed=True, height=.25)

    action = ""
    while action != "quit":
        if action == "left":
            degrees -= 10
        elif action == "right":
            degrees += 10
        elif action in ("up", "down"):
            angle = degrees/360*2*pi
            forward_x, forward_y = Camera.angle_to_vector(angle, d=4)
            if action == "up":
                camera_x += forward_x
                camera_y += forward_y
            else:
                camera_x -= forward_x
                camera_y -= forward_y

        camera.set(camera_x, camera_y, degrees/360*2*pi)
        action = make_image(img, camera, world)
