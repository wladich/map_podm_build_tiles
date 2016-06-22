#!/bin/env python
# -*- coding: utf-8 -*-

import sys
import pyproj
import json
from shapely import geometry

proj_wgs84 = pyproj.Proj('+init=epsg:4326')
proj_gmerc = pyproj.Proj('+init=epsg:3785')


def read_plt(filename):
    points = []
    with open(filename) as f:
        lines = f.readlines()[6:]
    for line in lines:
        lat, lon = map(float, line.split(',')[:2])
        points.append((lon, lat))
    return points


def linestring_from_plt(filename):
    points = [pyproj.transform(proj_wgs84, proj_gmerc, lon, lat) for lon, lat in read_plt(filename)]
    linear_ring = geometry.LinearRing(points)
    if not linear_ring.is_valid:
        linear_ring = geometry.LinearRing(points[1:])
    if not linear_ring.is_valid:
        raise Exception('Invalid ring geometry in file "%s".' % filename)
    return linear_ring

args = sys.argv[1:]
rings = [linestring_from_plt(fn) for fn in args]

polygons = []
for ring in rings:
    is_inner = False
    for other_ring in rings:
        if ring != other_ring and geometry.Polygon(other_ring).contains(geometry.Polygon(ring)):
            is_inner = True
            break
    if is_inner:
        continue
    inner_rings = []
    for other_ring in rings:
        if ring != other_ring and geometry.Polygon(ring).contains(geometry.Polygon(other_ring)):
            inner_rings.append(other_ring)
    polygon = geometry.Polygon(ring, inner_rings)
    if not polygon.is_valid:
        raise Exception('Invalid polygon constructed (maybe inner rings intersect shell or each other)')
    polygons.append(polygon)

coords_json = []
if polygons and not geometry.MultiPolygon(polygons):
    raise Exception('Invalid multipolygon (maybe polygons overlap)')

for polygon in polygons:
    coords_json.append([list(polygon.exterior.coords)] + [list(ring.coords) for ring in polygon.interiors])
print json.dumps(coords_json)
