import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import triangle.plot

"""
Mesh generation from binary images by Yalım Doğan

This program reads the given image to detect its contours and using its hierarchy to
generate corresponding mesh which can contain multiple polygons and holes.

Program uses the following Python libraries:

- Triangle for triangulations
- Shapely for hole detection
- Openmesh for output

"""

# This method searches for holes inside contours, by exposing their parent-child relationship.
# Rule is: If number of parents is odd, the polygon with this contour as its boundary has a hole
# The above rules holds for 1px wide polygons, as openf returns 1 internal and 1 external contours in that case
def determine_holes(contours, hierarchy, curLevel=0, contIndex=0):

    hole = []
    status = hierarchy[contIndex]
    if curLevel % 2 == 1:
        print("Contour {} has a hole in it".format(contIndex))
        hole = generatePolygon(contours, hierarchy, contIndex)
    if status[2] != -1:  # Has child(ren)
        hole += determine_holes(contours, hierarchy, curLevel + 1, status[2])
    if status[0] != -1:
        hole += determine_holes(contours, hierarchy, curLevel, status[0])
    return hole


# Generate the polygon that contains a hole, so that we can determine a single point in it that
# will be passed to Triangle as hole.

# I have used Shapely module here,as it can find a point that is guaranteed to be
# inside the polygon even if it is concave easily.
def generatePolygon(contours, hierarchy, contIndex):
    from shapely.geometry import Polygon  # Is a geographic information system library

    children = np.where(hierarchy[:, 3] == contIndex)[0]
    print("Children of {} are {}".format(contIndex, children))

    polyVertices = contours[contIndex]
    holeVertices = []

    for c in range(len(children)):
        holeVertices = holeVertices + contours[children[c]]
    if holeVertices == []:
        poly = Polygon(shell=polyVertices)
    else:
        poly = Polygon(shell=polyVertices, holes=[holeVertices])
    print("Generated the polygon, returning holes")

    hole = poly.representative_point()

    return [[hole.x, hole.y]]


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Create mesh obj from given binary/grayscale image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--image", help="Image to create mesh of", default="test.jpg"
    )
    parser.add_argument(
        "-ir",
        "--refineImage",
        help="Refine the image using erosion, dilation",
        const=True,
        default=False,
        nargs="?",
    )
    parser.add_argument(
        "--triangleCommands",
        help="Commands for Triangle (check Triangle documentation)",
        default="p",
    )
    args = parser.parse_args()

    # Load the given binary image
    img = cv2.imread(args.image)
    if img is None:
        print("Image {} not found".format(args.image))
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, 0)

    # Apply dilation, erosion if necessary
    if args.refineImage:
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closing_kernel, thresh)
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_kernel, thresh)

    # SIMPLE APPROX returns too many points for construction
    image, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )

    # Contour approximation decreases the amount of vertices that would be required to represent it.
    # It gives a shape that is close to a convex hull.THIS DOESN'T GIVE GOOD RESULTS WITH CONCAVE SHAPES
    # epsilon = 0.05 * cv2.arcLength(contours[0], True)
    # contours[0] = cv2.approxPolyDP(contours[0],epsilon,True)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Centroid of the mesh, but not necessarily on the mesh itself. Therefore not useful
    # mom = cv2.moments(contours[0])
    # print("{} - {}".format(mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]))

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contours", image)
    cv2.waitKey(0)

    v = {"vertices": [], "segments": []}
    overhead = 0  # Used for maintaining the inner connection of contours
    contours_new = []
    for i in range(len(contours)):

        # Correct the y coordinates of the contours
        contours_new.append(
            list(map(lambda x: [x[0][0], img.shape[0] - x[0][1]], contours[i]))
        )
        contourSegments = []
        for s in range(len(contours_new[i])):
            contourSegments.append(
                [s + overhead, ((s + 1) % (len(contours_new[i]))) + overhead]
            )
        overhead += len(contours_new[i])
        v["vertices"].append(contours_new[i])
        v["segments"].append(contourSegments)

    v["vertices"] = v["vertices"][0]
    v["segments"] = v["segments"][0]

    # Determine and instert the holes
    holes = determine_holes(contours_new, hierarchy[0])
    if len(holes) > 0:
        v["holes"] = holes

    # "p" is a mandatory flag, as it says Triangle that the given vertice-segment info forms a PSLG
    # "q" enforces 20 degrees as minimum for all corners in triangles, which means more triangles
    # "Y" forbids usage of Steiner points
    segments = triangle.triangulate(v, 'p')

    try:
        triangle.plot(
            plt.axes(),
            vertices=segments["vertices"],
            triangles=segments["triangles"],
            holes=segments["holes"],
        )
    except KeyError:  # Some PSLG's (poly files) contain no holes, therefore visualization cannot be done with it
        triangle.plot(
            plt.axes(), vertices=segments["vertices"], triangles=segments["triangles"]
        )

    print("Generated triangulation {}".format(segments["triangles"]))

    # Export in obj format using openmesh
    import openmesh

    mesh = openmesh.TriMesh()

    # We need to keep the created vertex handle objects so we can use them to create faces
    openvertices = []
    for vertex in segments["vertices"]:
        openvertices.append(mesh.add_vertex([vertex[0], vertex[1], 0]))

    for triangles in segments["triangles"]:
        mesh.add_face([openvertices[triangle] for triangle in triangles])

    import os
    import ntpath

    if not os.path.exists("obj_files"):
        os.makedirs("obj_files")

    openmesh.write_mesh(
        "obj_files/" + ntpath.basename(args.image).split(".")[0] + ".obj", mesh
    )

    plt.show()
