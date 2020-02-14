# from PIL import Image, ImageDraw
import cv2
import numpy as np
from math import sin, cos, pi
from random import randint


COLORS = dict(black=(0, 0, 0), blue=(1, 0, 0), green=(0, 1, 0), red=(0, 0, 1), yellow=(1, 1, 0))


def do_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    t = (x1 - x3) * (y3 - y4) - ((y1 - y3) * (x3 - x4))
    t_div = (x1 - x2) * (y3 - y4) - ((y1 - y2) * (x3 - x4))
    
    
    if t_div != 0:
        t /= t_div
        p_x, p_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)

        return t, p_x, p_y
    
    return -1, 0, 0

def proj_to_screen_coords(val_x, dist, focal_dist, wall_height, screen_height, screen_width):
    
    height = abs(wall_height * focal_dist/dist)

    bottom_offset = (screen_height - abs(300 * focal_dist/dist)) / 2
    top_offset = screen_height - bottom_offset - height
    
    x = screen_width + val_x * screen_width
    y = top_offset
    y_ = screen_height - bottom_offset

    return x, y, y_



class Eye:
    def __init__(self, focal_distance, focal_width):
        self.focal_distance = focal_distance
        self.focal_width = focal_width

    def set(self, x, y, angle):
        self.eye_x, self.eye_y, self.eye_angle = x, y, angle
        self._make_field_line()
    
    def get_point(self):
        return (self.eye_x, self.eye_y)
    
    def _make_field_line(self):
        x_f, y_f = self.angle_to_vector(self.eye_angle, self.focal_distance)
        x_f += self.eye_x
        y_f += self.eye_y

        x_, y_ = self.angle_to_vector(self.eye_angle + pi/2, self.focal_width)
    
        x1 = x_f - x_
        y1 = y_f - y_
    
        x2 = x_f + x_
        y2 = y_f + y_

        self.field_line = (x1, y1, x2, y2)
    
    def map_to_view(self, x, y):
        # Does the line from (x, y) to eye cross the visual field line?
        are_intersecting, inter_x, inter_y = do_lines_intersect(
            *self.field_line,
            x, y, self.eye_x, self.eye_y
        )
        
        # Make sure that the eye isn't between the object and the point
        # of intersection with the visual field line, since this means
        # the object is behind the eye.

        min_x, max_x = min(inter_x, x), max(inter_x, x)
        min_y, max_y = min(inter_y, y), max(inter_y, y)
        
        are_intersecting = not (min_x <= self.eye_x <= max_x and min_y <= self.eye_y <= max_y)
        
        if not are_intersecting:
            return False, 0

        # From the point of intersection on the visual field line,
        # find the [0, 1] relative position on the visual field line.
        
        x1, y1, x2, y2 = self.field_line
        
        if x2 - x1 != 0:
            field_pos = (inter_x - x1) / (x2 - x1)
        else:
            field_pos = (inter_y - y1) / (y2 - y1)

        return are_intersecting and 0 <= field_pos <= 1, field_pos
    
    def distance(self, x, y):
        return ((x - self.eye_x) ** 2 + (y - self.eye_y) ** 2) ** .5
    
    @classmethod
    def angle_to_vector(cls, angle, d = 1):
        return cos(angle) * d, sin(angle) * d


class Image:
    def __init__(self, w, h):
        self.img = np.full((h, w * 2, 3), 1, dtype="float")
    
    def draw_circle(self, x, y, color="red", size=3):
        x, y = int(x), int(y)
        self.img = cv2.circle(self.img, (x, y), 1, COLORS[color], 2)
#         draw.ellipse((x-size, y-size, x+size, y+size), fill=color)

    def draw_line(self, points, color="black"):
        x0, y0, x1, y1 = points
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        self.img = cv2.line(self.img, (x0, y0), (x1, y1), COLORS[color])
#         draw.line(points, fill=color)
        
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
    

def make_image(eye, degrees, idx):
    screen_width = 500
    screen_height = 500
    
    focal_dist = 40
    focal_width = 35

#     im = Image.new('RGB', (screen_width * 2, screen_height), "white")
#     draw = ImageDraw.Draw(im)
    img = Image(screen_width, screen_height)
    eye__ = Eye(focal_dist, focal_width)
    eye__.set(*eye, degrees/360*2*pi)
    # Draw topdown view

    img.draw_eye(eye__.get_point(), eye__.eye_angle)
    visual_field_points = eye__.field_line
    
    # draw projective plane
    img.draw_line(visual_field_points, "green")
    
    wall_points = []
    wall_connections = []
    
    def add_wall(list_of_points, closed=False):
        i = len(wall_points)
        wall_points.extend(list_of_points)
        connections = list(range(i, len(wall_points)))
        if closed:
            connections.append(i)
        wall_connections.append(connections)
    
    add_wall([
        (20, 120),
        (60, 120),
        (60, 60),
        (100, 60),
        (100, 20),
    ])

    add_wall([
        (160, 20),
        (160, 60),
        (200, 60),
        (200, 120),
        (240, 120),
    ])

    add_wall([
        (20, 160),
        (60, 160),
        (60, 200),
        (60, 240),
        (100, 280),
        (140, 280),
    ])
    
    short_wall_idx = len(wall_points)
    
    add_wall([        
        (100, 100),
        (105, 100),
        (105, 95),
        (100, 95),
    ], closed=True)
        
    add_wall([
        (160, 100),
        (165, 100),
        (165, 95),
        (160, 95)
    ], closed=True)

    vals = []

    for wall_point in wall_points:
        is_vis, val_x = eye__.map_to_view(*wall_point)
        if is_vis:
            img.draw_circle(*wall_point, "black" if is_vis else "yellow")
        
        dist = eye__.distance(*wall_point)
        
        vals.append((is_vis, val_x, dist))

    for connection_list in wall_connections:
        for from_idx, to_idx in zip(connection_list[:-1], connection_list[1:]):
            wall_from = wall_points[from_idx]
            wall_to = wall_points[to_idx]

            img.draw_line((*wall_from, *wall_to))

    # Draw front view
    
    img.draw_line((screen_width, 0, screen_width, screen_height))

    visual_field_points = []
    for i, (is_vis, val_x, dist) in enumerate(vals):
    
        wall_height = 300 if i < short_wall_idx else 80
        
        x, y, y_ = proj_to_screen_coords(
            val_x,
            dist,
            focal_dist,
            wall_height,
            screen_height,
            screen_width
        )
        
        visual_field_points.append((is_vis, x, y, y_))

    # draw vertical lines for wall points
    for i, (is_vis, x, y, y_) in enumerate(visual_field_points):
        if is_vis:
            img.draw_line((x, y, x, y_))

    # draw top and bottom of connected wall points
    for connection_list in wall_connections:
        for from_idx, to_idx in zip(connection_list[:-1], connection_list[1:]):
            if vals[from_idx][0] or vals[to_idx][0]:
                _, x0, y0, y0_ = visual_field_points[from_idx]
                _, x1, y1, y1_ = visual_field_points[to_idx]

                img.draw_line((x0, y0, x1, y1))
                img.draw_line((x0, y0_, x1, y1_))

#     im.save(f'img{idx}.jpg')
    return img.show()
    

if __name__ == "__main__":

    import sys

    

    eye_x, eye_y = (140, 160)
    degrees = -95
    
    action = ""
    while action != "quit":
        if action == "left":
            degrees -= 10
        elif action == "right":
            degrees += 10
        elif action in ("up", "down"):
            angle = degrees/360*2*pi
            forward_x, forward_y = Eye.angle_to_vector(angle, d=4)
            if action == "up":
                eye_x += forward_x
                eye_y += forward_y
            else:
                eye_x -= forward_x
                eye_y -= forward_y

        action = make_image((eye_x, eye_y), degrees, eye_y)