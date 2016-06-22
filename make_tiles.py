# -*- coding: utf-8 -*-
import sys
import os
import argparse
import pyproj
import subprocess
import multiprocessing
import multiprocessing.pool
import traceback
import signal
import json
import shapely.geometry as geometry
import tempfile
import math
from PIL import Image, ImageFilter
from array import array
import png
import imagequant
from itertools import izip
from cStringIO import StringIO
import time
import pysqlite2.dbapi2 as sqlite

proj_wgs84 = pyproj.Proj('+init=epsg:4326')
proj_gmerc = pyproj.Proj('+init=epsg:3785')
max_gmerc_coord = pyproj.transform(proj_wgs84, proj_gmerc, 180, 0)[0]

config = None
vmaps_extents = None
border = None
tile_writer = None

### misc utils

class ChildException(Exception):
    pass

def mpimap_wrapper((func, args, kwargs)):
    result = {'error': None}
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        result['value'] = func(*args, **kwargs)
    except Exception:
        err_cls, err, tb = sys.exc_info()
        result['error'] = err
        result['traceback'] = ''.join(traceback.format_tb(tb))
    return result


def mpstarimap(func, job, **kwargs):
    job = ((func, args, kwargs) for args in job)
    pool = multiprocessing.Pool()
    for result in pool.imap(mpimap_wrapper, job):
        error = result['error']
        if error:
            raise ChildException('%r\n%s' % (error, result['traceback']))
        yield result['value']


def mpimap(func, job, **kwargs):
    job = ((x,) for x in job)
    return mpstarimap(func, job, **kwargs)

#### tile and geometry utils


def tile_size_in_gmerc_meters(level):
    return max_gmerc_coord * 2 / 2 ** level


def tile_from_gmerc_meters(x, y, level):
    tile_size = tile_size_in_gmerc_meters(level)
    tx = (x + max_gmerc_coord) / tile_size
    ty = (-y + max_gmerc_coord) / tile_size
    return int(tx), int(ty)


def link_polygons(p1, p2):
    '''Finds two closest points of two polygons and joins pokygons,
       adding link between those points.
       Used to simulate island polygons.
       Naive implementation, does not check for intersections, use only
       for simple areas.'''
    p1 = p1.exterior.coords
    p2 = p2.exterior.coords
    min_dist = 1e10
    min_dist_nodes = None
    for i, (x1, y1) in enumerate(p1):
        for j, (x2, y2) in enumerate(p2):
            dx = x2 - x1
            dy = y2 - y1
            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                min_dist_nodes = i, j
    i, j = min_dist_nodes
    p = p1[:i+1] + p2[j:] + p2[:j + 1] + p1[i:]
    p = geometry.Polygon(p)
    return p


class Extents(object):
    def __init__(self, extents, gmerc=False):
        if gmerc:
            extents = extents[:]
            self._extents_gmerc = extents
            self._extents_wgs84 = self._transform_extents(
                proj_gmerc, proj_wgs84, extents)
        else:
            self._extents_wgs84 = extents
            self._extents_gmerc = self._transform_extents(
                proj_wgs84, proj_gmerc, extents)

    def _transform_extents(self, s_srs, t_srs, extents):
        x1, y1, x2, y2 = extents
        [[x1, x2], [y1, y2]] = pyproj.transform(s_srs, t_srs, [x1, x2], [y1, y2])
        return [x1, y1, x2, y2]

    @classmethod
    def from_wgs84(cls, extents):
        return cls(extents)

    @classmethod
    def from_gmerc(cls, extents):
        return cls(extents, gmerc=True)

    @classmethod
    def from_tile_index(cls, tile_x, tile_y, level):
        tile_size = tile_size_in_gmerc_meters(level)
        x = tile_x * tile_size - max_gmerc_coord
        y = -tile_y * tile_size + max_gmerc_coord
        extents = x, y - tile_size, x + tile_size, y
        return cls.from_gmerc(extents)

    @property
    def as_wgs84(self):
        return self._extents_wgs84

    @property
    def as_gmerc(self):
        return self._extents_gmerc

    @classmethod
    def from_combination(cls, extents_list):
        extents_list = [e.as_wgs84 for e in extents_list]
        extents_list = zip(*extents_list)
        total_extents = [
            min(extents_list[0]),
            min(extents_list[1]),
            max(extents_list[2]),
            max(extents_list[3])
            ]
        return cls(total_extents)

    def intersects(self, other):
        left1, bottom1, right1, top1 = self.as_gmerc
        left2, bottom2, right2, top2 = other.as_gmerc
        return left1 < right2 and right1 > left2 and bottom1 < top2 and top1 > bottom2

    def expand_by_meters(self, margin):
        minx, miny, maxx, maxy = self.as_gmerc
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin
        return self.from_gmerc([minx, miny, maxx, maxy])

    def get_size_gmerc(self):
        extents = self.as_gmerc
        return extents[2] - extents[0], extents[3] - extents[1]


#### load data


def read_vmap_extents(vmap_filename):
    in_data = False
    minx = miny = 1e100
    maxx = maxy = -1e100
    for line_n, line in enumerate(open(vmap_filename), 1):
        if line.startswith('VMAP '):
            continue
        try:
            command, data = line.split('\t', 1)
        except ValueError as e:
            raise Exception('Faild to parse line %d in %s: %s' % (line_n, vmap_filename, e))
        command = command.strip()
        if command == 'DATA':
            in_data = True
        elif command == '':
            if not in_data:
                continue
        else:
            in_data = False
            continue
        data = data.split()
        coords = [map(int, xy.split(',')) for xy in data]
        x, y = zip(*coords)
        minx = min(minx, *x)
        miny = min(miny, *y)
        maxx = max(maxx, *x)
        maxy = max(maxy, *y)
    return Extents.from_wgs84([minx / 1000000., miny / 1000000., maxx / 1000000., maxy / 1000000.])

def get_all_vmaps_extents():
    vmaps_dir = config.vmaps_dir
    vmaps_extents = {}
    vmap_paths = [os.path.join(vmaps_dir, fn) for fn in os.listdir(vmaps_dir) if fn.endswith('.vmap')]
    for vmap_path, vmap_extents in zip(vmap_paths, mpimap(read_vmap_extents, vmap_paths)):
        vmaps_extents[vmap_path] = vmap_extents
    return vmaps_extents

def get_border():
    '''Load border from json format and transform it to wgs84'''
    if config.border_filename:
        areas = json.load(open(config.border_filename))
        for area in areas:
            for polygon in area:
                x, y = zip(*polygon)
                x, y = pyproj.transform(proj_gmerc, proj_wgs84, x, y)
                polygon[:] = zip(x, y)
        border = geometry.MultiPolygon([(area[0], area[1:]) for area in areas])
        return border
    else:
        return None

#### Save data
def save_png_rgba(im, fd):
    has_alpha = im.mode[-1] == 'A'
    ar = array('B', im.tostring())
    pngw = png.Writer(size=im.size, alpha=has_alpha, compression=1)
    pngw.write_array(fd, ar)

def save_png_with_palette(im, fd):
    quantized = imagequant.quantize_image(im, colors=256, speed=8)
    palette = list(quantized['palette'])
    palette = zip(palette[::4], palette[1::4], palette[2::4], palette[3::4])
    if all(c[3]==255 for c in palette):
        palette = [c[:3] for c in palette]
    pngw = png.Writer(size=im.size, palette=palette)
    pngw.write_array(fd, quantized['image'])

def serialize_image(im):
    s = StringIO()
    save_png_rgba(im, s)
    return s.getvalue()

def unserialize_image(s):
    s = StringIO(s)
    im = Image.open(s)
    return im

def is_image_empty(im):
    if im.mode[-1] == 'A':
        _, alpha_max = im.split()[-1].getextrema()
        return alpha_max == 0
    else:
        return False

class MBTilesWriter(object):
    SCHEME = '''
        CREATE TABLE tiles(zoom_level integer, tile_column integer, tile_row integer, tile_data blob);
        CREATE UNIQUE INDEX idx_tiles ON tiles(zoom_level, tile_column, tile_row);
        CREATE INDEX idx_z ON tiles(zoom_level);
    '''

    PRAGMAS = '''
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = 0;
    '''
    def __init__(self, path, image_encoder):
        if os.path.exists(path):
            os.remove(path)
        self.encoder = image_encoder
        self.conn = sqlite.connect(path)
        self.conn.executescript(self.SCHEME)
        self.conn.executescript(self.PRAGMAS)
        self.lock = multiprocessing.RLock()

    def write(self, im, tile_x, tile_y, level):
        if not is_image_empty(im):
            tile_y = 2 ** level - tile_y - 1
            s = StringIO()
            self.encoder(im, s)
            s = buffer(s.getvalue())
            with self.lock:
                self.conn.execute('''
                    INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?,?,?,?)''',
                    (level, tile_x, tile_y, s))
                self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.execute('PRAGMA journal_mode = off')
        self.conn.close()

    def __del__(self):
        self.close()

class FilesWriter(object):
    def __init__(self, path, image_encoder):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.path = path
        self.encoder = image_encoder

    def write(self, im, tile_x, tile_y, level):
        if not is_image_empty(im):
            filename = '%s_%s_%s.png' % (level, tile_y, tile_x)
            filename = os.path.join(self.path, filename)
            with open(filename, 'w') as f:
                self.encoder(im, f)

#### main

def list_vmaps_intersecting_extent(extent):
    return [vmap_name
            for vmap_name, vmap_extents in vmaps_extents.items()
            if extent.intersects(vmap_extents)]

def get_extents_with_border_intersection(extents):
    cropped_box = geometry.box(*extents.as_wgs84)
    if border:
        cropped_box = cropped_box.intersection(border)
        if isinstance(cropped_box, geometry.MultiPolygon):
            cropped_box = reduce(link_polygons, cropped_box)
        if not isinstance(cropped_box, geometry.Polygon):
        # the tile is totaly outside border
            return None
    coords = list(cropped_box.exterior.coords)
    return coords


def join_vmaps_for_extents(extents, out_name):
    vmaps = list_vmaps_intersecting_extent(extents)
    if not vmaps:
        return False
    render_border = get_extents_with_border_intersection(extents)
    if not render_border:
        return False

    arg_geom = sum(render_border, ())
    arg_geom = ','.join(map(str, arg_geom))
    cmd = [
        'mapsoft_vmap', '-v',
        ] + vmaps + [
        '-o',  out_name,
        '--set_brd', arg_geom
        ]
    subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True

def render_extents(extents, pixel_size, vmap_name, out_name):
    dx, dy = extents.get_size_gmerc()
    x0, y0, _, _ = extents.as_gmerc
    arg_geom = '%sx%s+%s+%s' % (dx, dy, x0, y0)
    _, y1, _, y2 = extents.as_wgs84
    corrected_rscale = config.rscale / math.cos(math.radians((y1 + y2) / 2))

    # there were some problems with rounding
    pixel_size += 0.49
    dpi = corrected_rscale * pixel_size * 2.54 / (dx * 100)
    if config.low_quality:
        arg_qual = ['--contours', '0', '--label_style', '0']
    else:
        arg_qual = []

    cmd = [
        'vmap_render',
        '-d', str(dpi),
        '--datum', 'sphere',
        '--proj', 'google',
        '--geom', arg_geom,
        '--transp_margins', '1',
        '--rscale=%s' % corrected_rscale] + arg_qual + [
        vmap_name, out_name]
    subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_rendered_metatile(tile_x, tile_y, tile_level):
    metatile_extents = Extents.from_tile_index(tile_x, tile_y, tile_level)
    margin = tile_size_in_gmerc_meters(config.max_level)
    metatile_extents = metatile_extents.expand_by_meters(margin)

    with tempfile.NamedTemporaryFile(suffix='.vmap') as vmap_file:
        if join_vmaps_for_extents(metatile_extents, vmap_file.name):
            meta_delta = config.max_level - config.metatile_level
            metatile_size_pixels = 256 * (2 ** meta_delta + 2)
            if config.low_quality:
                metatile_size_pixels /= 2
            with tempfile.NamedTemporaryFile(suffix='.png') as image_file:
                render_extents(metatile_extents, metatile_size_pixels, vmap_file.name, image_file.name)
                im = Image.open(image_file)
                assert im.size == (metatile_size_pixels, metatile_size_pixels), im.size
                if config.low_quality:
                    # it was rendered half the size
                    w, h = im.size
                    im = im.resize((w * 2, h * 2))
                width, height = im.size
                im = im.crop([256, 256, width - 256, height - 256])
                return im
        else:
            return None

def slice_metatile(im, metatile_x, metatile_y, dest_level):
    meta_delta = dest_level - config.metatile_level
    meta_q = 2**meta_delta
    assert im.size == (256 * (meta_q),) * 2
    tile_x0 = metatile_x * meta_q
    tile_y0 = metatile_y * meta_q
    for d_tile_y in xrange(meta_q):
        y0 = d_tile_y * 256
        for d_tile_x in xrange(meta_q):
            x0 = d_tile_x * 256
            im2 = im.crop([x0, y0, x0+256, y0+256])
            tile_writer.write(im2, tile_x0 + d_tile_x, tile_y0 + d_tile_y, dest_level)

def process_metatile(tile_x, tile_y):
    # 1. render
    # 2. crop
    # 3. cut tiles
    # 4. shrink and cut overviews
    # 5. return last overview
    im = get_rendered_metatile(tile_x, tile_y, config.metatile_level)
    if im is not None:
        for level in xrange(config.max_level, config.metatile_level, -1):
            slice_metatile(im, tile_x, tile_y, level)
            resampler = Image.ANTIALIAS if not config.low_quality else Image.NEAREST
            im = im.resize((im.size[0] / 2, )*2, resampler)
        tile_writer.write(im, tile_x, tile_y, config.metatile_level)
        return serialize_image(im)
    else:
        return None

def list_tiles(level):
    total_extents = Extents.from_combination(vmaps_extents.values())
    minx, miny, maxx, maxy = total_extents.as_gmerc
    tile_minx, tile_maxy = tile_from_gmerc_meters(minx, miny, level)
    tile_maxx, tile_miny = tile_from_gmerc_meters(maxx, maxy, level)
    tiles = []
    for tx in xrange(tile_minx, tile_maxx + 1):
        for ty in xrange(tile_miny, tile_maxy + 1):
            tiles.append((tx, ty))
    return tiles

def make_tiles_from_metalevel_to_maxlevel():
    metatiles = list_tiles(config.metatile_level)
    saved_tiles = {}
    n = 0
    for tile_index, rendered_metatile in izip(metatiles, mpstarimap(process_metatile, metatiles)):
        if rendered_metatile is not None:
            saved_tiles[tile_index] = rendered_metatile
        n += 1
        print ('\r%.1f%%' % (n * 100./ len(metatiles))),
        sys.stdout.flush()
    print
    return saved_tiles

highlight_color = 0xdb, 0x5a, 0x00

def build_overviews(saved_tiles, source_level):
    next_saved_tiles = {}
    dest_level = source_level - 1
    if dest_level < 0:
        return
    for tile_x, tile_y in list_tiles(dest_level):
        im_dest = None
        for dx in [0, 1]:
            for dy in [0, 1]:
                src_tile_x = tile_x * 2 + dx
                src_tile_y = tile_y * 2 + dy
                src_tile_index = (src_tile_x, src_tile_y)
                if src_tile_index in saved_tiles:
                    if im_dest is None:
                        im_dest = Image.new('RGBA', (512, 512))
                    src_image = unserialize_image(saved_tiles[src_tile_index])
                    if config.highlight_level == dest_level:
                        src_image = src_image.convert('RGBA')
                        mask = src_image.split()[-1]
                        mask = mask.point(lambda x: 0 if x < 255 else 255)
                        im_dest.paste(highlight_color, (dx * 256, dy * 256), mask)
                    else:
                        im_dest.paste(src_image, (dx * 256, dy * 256))
        if im_dest is not None:
            if (config.highlight_level is not None and dest_level <= config.highlight_level):
                im_dest = im_dest.resize((256, 256), Image.NEAREST)
                im_dest = im_dest.filter(ImageFilter.MaxFilter())
            else:
                resampler = Image.NEAREST if config.low_quality else Image.ANTIALIAS
                im_dest = im_dest.resize((256, 256), resampler)
            tile_writer.write(im_dest, tile_x, tile_y, dest_level)
            next_saved_tiles[(tile_x, tile_y)] = serialize_image(im_dest)
    build_overviews(next_saved_tiles, dest_level)

def parge_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low-quality', action='store_true')
    parser.add_argument('--no-size-optimize', action='store_true')
    parser.add_argument('--border', metavar='FILE', dest='border_filename')
    parser.add_argument('--vmap', metavar='DIR', dest='vmaps_dir', required=True)
    parser.add_argument('--out', metavar='PATH', dest='out_path', required=True,
                        help='Filename of mbtiles container or tiles dir')
    parser.add_argument('--rscale', type=int, required=True)
    parser.add_argument('--max-level', type=int, required=True)
    parser.add_argument('--highlight-level', type=int, required=False)
    parser.add_argument('--metatile-level', type=int, required=True)
    parser.add_argument('--format', choices=['files', 'mbtiles'], default='files', help='default is files')
    args = parser.parse_args()
    return args

def main():
    global config
    global border
    global vmaps_extents
    global tile_writer
    global image_optimizer
    config = parge_args()
    border = get_border()
    vmaps_extents = get_all_vmaps_extents()
    if config.no_size_optimize:
        image_encoder = save_png_rgba
    else:
        image_encoder = save_png_with_palette
    tile_writer_class = {'files': FilesWriter, 'mbtiles': MBTilesWriter}[config.format]
    tile_writer = tile_writer_class(config.out_path, image_encoder)

    tiles_at_metatile_level = make_tiles_from_metalevel_to_maxlevel()
    build_overviews(tiles_at_metatile_level, config.metatile_level)

t = time.time()
main()
print 'Done in %.1f seconds' % (time.time() - t)

