import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_centers(k, points):
    centers = []
    centers.append(points[0])

    for i in range(1, k):
        max_dist = -99999999
        new_center = None

        for point in points:
            min_dist = 99999999
            for center in centers:
                dist = distance(point, center)
                min_dist = min(min_dist, dist)

            if min_dist > max_dist:
                max_dist = min_dist
                new_center = point

        centers.append(new_center)

    print(centers)


k = int(input())
n = int(input())
points = []

for i in range(0, n):
    x, y = map(int, input().split())
    points.append((x, y))

find_centers(k, points)


