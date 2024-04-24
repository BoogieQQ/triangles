import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


class PictureProcess:

    def __init__(self):
        self.borders = [(np.array([0, 70, 50]), np.array([20, 230, 150])), (np.array([0, 0, 0]), np.array([100, 255, 255]))]

        template = cv2.imread('template.jpg')
        template = cv2.resize(template, (200, 200), interpolation=cv2.INTER_LINEAR)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, self.template = cv2.threshold(template, 120, 255, cv2.THRESH_BINARY_INV)

    def rotate(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def find_components(self, img, x_center, y_center, tr=8, max_tr=200):
        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
        areas = []
        counter = {(False, False): 0,
                   (False, True): 0,
                   (True, False): 0,
                   (True, True): 0,
                   }
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
            areas.append(area)
            if tr < area < max_tr:
                x, y = centroid[i]
                counter[(x < x_center, y < (y_center - 10))] += 1
        counter[(True, True)] += counter[(False, True)]
        del counter[(False, True)]
        for key in counter.keys():
            if counter[key] > 7:
                counter[key] = 0
        return counter, areas


    def process(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        tmp_ans = {}
        final_stats = []
        for grads in [-90, -60, -30, 0, 180, 30, 60, 90]:
            for size in [(140, 140), (120, 120)]:
                for k, (lower, upper) in enumerate(self.borders):
                    resized_template = cv2.resize(self.template, size, interpolation=cv2.INTER_LINEAR)
                    rot_template = self.rotate(resized_template, grads)
                    w, h = rot_template.shape[::-1]
                    mask = cv2.inRange(hsv, lower, upper)

                    res = cv2.matchTemplate(mask, rot_template, cv2.TM_CCOEFF_NORMED)
                    # Specify a threshold
                    threshold = 0.45

                    # Store the coordinates of matched area in a numpy array
                    loc = np.where(res >= threshold)

                    img_rect = img.copy()
                    # Draw a rectangle around the matched region.
                    for pt in zip(*loc[::-1]):
                        f = True
                        for coords in tmp_ans.keys():
                            if max(abs(coords[0] - pt[0]), abs(coords[1] - pt[1])) < 100:
                                f = False
                                tmp_ans[coords].append((pt[0], pt[1], w, h, k))
                                break
                        if f:
                            tmp_ans[(pt[0], pt[1], w, h, k)] = [(pt[0], pt[1], w, h, k)]
                            cv2.rectangle(img_rect, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        ans = []
        for key in tmp_ans.keys():
            ans.append(np.mean(tmp_ans[key], axis=0).astype(int))
        img_rect = img
        final_ans = []
        final_output = np.zeros_like(img_rect)

        for pt in ans:
            triangle = img_rect[pt[1]:pt[1] + pt[2], pt[0]:pt[0] + pt[3]]
            hsv = cv2.cvtColor(triangle, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.borders[pt[4]][0], self.borders[pt[4]][1])

            totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(
                                                        mask, 4, cv2.CV_32S)

            output = np.zeros(mask.shape, dtype="uint8")
            areas = []
            for i in range(1, totalLabels):
                areas.append(values[i, cv2.CC_STAT_AREA])
            areas = np.array(areas)
            q = np.max(areas)

            for i, area in enumerate(areas):
                if area == q:
                    componentMask = (label_ids == (i + 1)).astype("uint8")
                    output = cv2.bitwise_or(output, componentMask)
            output = binary_fill_holes(output).astype("uint8")

            triangle = triangle * output[:, :, None]

            # save triangle for adding to final output
            ans_triangle = triangle.copy()
            contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ans_triangle, contours, 0, (0, 255, 0), 2)

            M = cv2.moments(contours[0])
            x_center1, y_center1 = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            cv2.circle(ans_triangle, (x_center1, y_center1), 5, (0, 255, 0), -1)

            # crap triangle
            x, y, pic_w, pic_h = cv2.boundingRect(contours[0])
            triangle = triangle[y:y + pic_h, x:x + pic_w]
            output = output[y:y + pic_h, x:x + pic_w]
            contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            M = cv2.moments(contours[0])
            x_center, y_center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            kernel = np.ones((2, 2), np.uint8)

            yellow = cv2.dilate(cv2.inRange(triangle, (0, 62, 62), (25, 255, 255), 10),
                                kernel, 3)
            y, _ = self.find_components(yellow, x_center, y_center, 10)

            blue = cv2.dilate(cv2.inRange(triangle, (60, 0, 0), (255, 100, 100)),
                              kernel, 3)
            b, _ = self.find_components(blue, x_center, y_center, 12)

            red = cv2.dilate(cv2.inRange(triangle, (10, 10, 80), (30, 30, 255)),
                             kernel, 3)
            r, _ = self.find_components(red, x_center, y_center, 10, 101)

            green = cv2.dilate(cv2.inRange(triangle, (0, 60, 0), (60, 255, 60)), kernel, 3)
            g, _ = self.find_components(green, x_center, y_center, 12)

            gray = cv2.cvtColor(triangle, cv2.COLOR_BGR2GRAY) * 255

            _, black = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
            black = cv2.dilate(cv2.subtract(cv2.subtract(black, red), blue),
                               kernel, 3)
            black *= output
            bl, _ = self.find_components(black, x_center, y_center, 12, 100)

            sensitivity = 100
            lower_white = np.array([0, 0, 255 - sensitivity])
            upper_white = np.array([255, sensitivity, 255])
            white = cv2.dilate(cv2.inRange(cv2.cvtColor(triangle, cv2.COLOR_BGR2HSV), lower_white, upper_white),
                               kernel, 3)

            w, white_areas = self.find_components(white, x_center, y_center, 5)

            area = output.mean()
            if abs(pic_h - pic_w) > 35 or area < 0.38 or area > 0.75:
                continue

            final_output[pt[1]:pt[1] + pt[2], pt[0]:pt[0] + pt[3]] += ans_triangle

            res = [pt[0] + x_center1, pt[1] + y_center1]
            for key in y.keys():
                res.append(min(max(y[key], b[key], r[key], w[key], bl[key], g[key]), 5))

            final_ans.append((*pt, *res))

        final_ans = sorted(final_ans, key=lambda x: (x[5], x[6]))
        for pt in final_ans:
            cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (0, 0, 255), 2)
            final_stats.append(str(pt[5:])[1:-1])
        return final_output, len(final_ans), final_stats
