import os
import shutil
import glob
import sys
import re
from PIL import Image
import pyproj
import subprocess
import json
from shapely.geometry import MultiPolygon, box, Polygon
import argparse
import multiprocessing
import functools
import traceback

proj_wgs84 = pyproj.Proj('+init=epsg:4326')
proj_gmerc = pyproj.Proj('+init=epsg:3785')
max_coord = pyproj.transform(proj_wgs84, proj_gmerc, 180, 0)[0]

class ChildException(Exception):
    pass
    
def mpimap_wrapper(func, args):
    result = {'error': None}
    try:
        result['value'] = func(*args)
    except Exception:
        tb = sys.exc_info()[2]
        result['error'] = ''.join(traceback.format_tb(tb))
    return result
    
def mpimap(func, job, **kwargs):
    if kwargs:
        func = functools.partial(func, **kwargs)
    func = functools.partial(mpimap_wrapper, func)
    pool = multiprocessing.Pool()        
    for result in pool.imap_unordered(func, job):
        error = result['error']
        if error:
            raise ChildException(error)
        yield result['value']
        
def shell_execute(command, stdin=None, env_variables=None, check=False, on_segfault=RuntimeError, **kwargs):
  if env_variables:
    env = os.environ.copy()
    env.update(env_variables)
  else:
    env = env_variables
  if hasattr(command, '__iter__'):
    shell = False
  else:
    command = [command]
    shell = True
  p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, shell=shell, **kwargs)
  out, err = p.communicate(stdin)
  command = ' '.join(command)
  if p.returncode == -11 and on_segfault is not None:
    raise on_segfault('Got a segfault (exit code -11) while runnig\n "%s"' %  command)
  if check and p.returncode != 0:
    raise RuntimeError('Error %s running "%s":\n%s' % (p.returncode, command, err))
  return p.returncode, out, err

def read_vmap_extents(vmap_filename):
    for line in open(vmap_filename):
        if line.startswith('BRD'):
            corners = line.split()[1:]
            corners = [map(int, c.split(',')) for c in corners]
            corners = zip(*corners)
            minx = min(corners[0]) / 1000000.
            maxx = max(corners[0]) / 1000000.
            miny = min(corners[1]) / 1000000.
            maxy = max(corners[1]) / 1000000.
            return minx, miny, maxx, maxy

def get_all_vmaps_extents(vmaps_dir):
    vmap_extents = {}
    for vmap_name in os.listdir(vmaps_dir):
        vmap_name = os.path.join(vmaps_dir, vmap_name)
        vmap_extents[vmap_name]  = read_vmap_extents(vmap_name)
    return vmap_extents

        
def transform_extents(s_srs, t_srs, extents):
    p1 = extents[:2]
    p2 = extents[2:]
    result = pyproj.transform(s_srs, t_srs, *zip(p1, p2))
    result = zip(*result)
    return result[0] + result[1]

def get_tile_size_meters(level):
    return max_coord * 2 / 2 ** level

def tile_from_meters(point, level):
    x, y = point
    tile_size = get_tile_size_meters(level)
    tx = (x + max_coord) / tile_size 
    ty = (-y + max_coord) / tile_size 
    return int(tx), int(ty)

def get_tile_extents_meters(tx, ty, level):
    tile_size =  get_tile_size_meters(level)
    x = tx * tile_size - max_coord
    y = -ty * tile_size + max_coord
    return [x, y - tile_size, x + tile_size, y]

def extents_intersect(ext1, ext2):
    left1, bottom1, right1, top1 = ext1
    left2, bottom2, right2, top2 = ext2
    return left1 < right2 and right1 > left2 and bottom1 < top2 and top1 > bottom2
    
def get_vmaps_intersecting_extent(vmap_extents, tile_extents_gmerc):
    result = []    
    tile_extents_wgs84 = transform_extents(proj_gmerc, proj_wgs84, tile_extents_gmerc)
    for vmap, vmap_extents in vmap_extents.items():
        if extents_intersect(tile_extents_wgs84, vmap_extents):
            result.append(vmap)
    return result

def combine_extents(extents_list):
    extents = zip(*extents_list)
    total_extents = [
        min(extents[0]),
        min(extents[1]),
        max(extents[2]),
        max(extents[3])
        ]
    return total_extents

def load_border(border_filename):
    '''Load border from json format and transform it to wgs84'''
    areas = json.load(open(border_filename))
    for area in areas:
        for polygon in area:
            x, y = zip(*polygon)
            x, y = pyproj.transform(proj_gmerc, proj_wgs84, x, y)
            polygon[:] = zip(x, y)
    border = MultiPolygon([(area[0], area[1:]) for area in areas])
    return border

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
    p = Polygon(p)
    return p

def join_vmaps_for_tile(vmaps, out_name, tile_extents, map_border):
#    arg_vmaps = ' '.join(vmaps)
    tile_extents = transform_extents(proj_gmerc, proj_wgs84, tile_extents)
    tile_box = box(*tile_extents)
    tile_boundary = tile_box
    if map_border:
        tile_boundary = tile_boundary.intersection(map_border)
        if isinstance(tile_boundary, MultiPolygon):
            tile_boundary = reduce(link_polygons, tile_boundary)
        if not isinstance(tile_boundary, Polygon):
            # the tile is totaly outside border
            return False
    arg_geom = list(tile_boundary.exterior.coords)
    arg_geom= sum(arg_geom, ())
    arg_geom = ','.join(map(str, arg_geom))
    cmd = [
        'mapsoft_vmap', '-v',  
        '--range_action',  'crop_spl', 
        ] + vmaps + [
        '-o',  out_name,
        '--range_datum', 'sphere', 
        '--range_proj', 'google',
        '--set_brd', arg_geom
        ]
    shell_execute(cmd, check=True)
    return True

def render_tile(tile_extents_meters, tile_size_pixels, vmap_name, png_name, low_quality):
    tile_width_meters = tile_extents_meters[2] - tile_extents_meters[0]
    dx = tile_extents_meters[2] - tile_extents_meters[0]
    dy = tile_extents_meters[3] - tile_extents_meters[1]
    arg_geom = '%sx%s+%s+%s' % (dx, dy, tile_extents_meters[0], tile_extents_meters[1])
    # there were some problems with rounding
    tile_size_pixels = tile_size_pixels + 0.49    
    if low_quality:
        arg_qual = ['--contours', '0', '--label_style', '0']
        tile_size_pixels /= 4
    else:
        arg_qual = []
    rscale = 50000/0.57
    dpi = rscale * tile_size_pixels * 2.54 / (tile_width_meters * 100)
    cmd = [
        'vmap_render',
        '-d', str(dpi), 
        '--datum', 'sphere', 
        '--proj', 'google',
        '--geom', arg_geom, 
        '--transp_margins', '1', 
        '--rscale=%s' % rscale] + arg_qual + [
        vmap_name, png_name]
    shell_execute(cmd, check=True)

def dice_metatile(meta_level, meta_tile_x, meta_tile_y, max_level,
                  metatile_png_name, low_quality, out_dir):
    im = Image.open(metatile_png_name)
    if low_quality:
        # it was rendered half the size 
        w, h = im.size
        im = im.resize((w * 4, h * 4))
    meta_delta = max_level - meta_level
    expected_tile_size = 2 ** meta_delta * 256 + 512
    assert im.size == (expected_tile_size, expected_tile_size)
    slices_number = 2 ** meta_delta
    for rel_tile_x in xrange(slices_number):
        abs_tile_x = meta_tile_x * slices_number + rel_tile_x
        pixel_x0 = 256 + rel_tile_x * 256
        for rel_tile_y in xrange(slices_number):
            abs_tile_y = meta_tile_y * slices_number + rel_tile_y
            pixel_y0 = 256 + rel_tile_y * 256
            im2 = im.crop([pixel_x0, pixel_y0, pixel_x0+256, pixel_y0+256])
            bands = im2.split()
            # Do not save blank tiles
            # Remove alpha channel if tile is fully opaque
            if len(bands) == 4:
                extrema = bands[3].getextrema()
                if extrema[1] == 0:
                    continue
                if extrema[0] == 255:
                    im2 = im2.convert('RGB')
            out_name = '%s_%s_%s.png' % (max_level, abs_tile_y, abs_tile_x)
            out_name = os.path.join(out_dir, out_name)
            im2.save(out_name)


def iterate_metatiles(max_level, metatile_level, total_gmerc_extents):
    metatile_minx, metatile_maxy = tile_from_meters(total_gmerc_extents[:2], metatile_level)
    metatile_maxx, metatile_miny = tile_from_meters(total_gmerc_extents[2:], metatile_level)
    for tx in xrange(metatile_minx, metatile_maxx + 1):
        for ty in xrange(metatile_miny, metatile_maxy + 1):
            yield tx, ty, metatile_level
                
def process_metatile(metatile_index, max_level, vmaps_extents, map_border, out_dir,
                     low_quality):
    tx, ty, meta_level = metatile_index
    metatile_extents_meters = get_tile_extents_meters(tx, ty, meta_level)
    margin = get_tile_size_meters(max_level)
    metatile_extents_meters[0] -= margin
    metatile_extents_meters[1] -= margin
    metatile_extents_meters[2] += margin
    metatile_extents_meters[3] += margin
    meta_delta = max_level - meta_level    
    metatile_size_pixels = 256 * (2 ** meta_delta + 2) + 0.49
    vmaps = get_vmaps_intersecting_extent(vmaps_extents, metatile_extents_meters)
    if vmaps:
        tile_index_str = '%s_%s' % (tx, ty)
        tmp_vmap_name = os.path.join(out_dir, '_tmp_%s.vmap' % tile_index_str)
        tmp_png_name = os.path.join(out_dir, '_tmp_%s.png' % tile_index_str)
        try:
            if join_vmaps_for_tile(vmaps, tmp_vmap_name, metatile_extents_meters, map_border):
                render_tile(metatile_extents_meters, metatile_size_pixels, 
                            tmp_vmap_name, tmp_png_name, low_quality)
                dice_metatile(meta_level, tx, ty, max_level, tmp_png_name,
                              low_quality, out_dir)
        finally:
            if os.path.exists(tmp_vmap_name):
                os.remove(tmp_vmap_name)
            if os.path.exists(tmp_png_name):
                os.remove(tmp_png_name)

def build_max_level(vmaps_dir, max_level, metatile_level, map_border, out_dir, 
                    low_quality):
    vmaps_extents = get_all_vmaps_extents(vmaps_dir)
    total_extents_wgs84 = combine_extents(vmaps_extents.values())
    total_extents_gmerc = transform_extents(proj_wgs84, proj_gmerc, total_extents_wgs84)
    metatiles_indexes = iterate_metatiles(max_level, metatile_level, total_extents_gmerc)
    metatiles_indexes = list(metatiles_indexes)
    metatiles_n = len(metatiles_indexes)
#    for n, metatile_index in enumerate(metatiles_indexes):
#        process_metatile(metatile_index, max_level, vmaps_extents, map_border,
#                         out_dir, low_quality)
    for n, _ in enumerate(mpimap(process_metatile, metatiles_indexes, 
                                 max_level=max_level, 
                                 vmaps_extents=vmaps_extents,
                                 map_border=map_border,
                                 out_dir=out_dir, low_quality=low_quality)):
        print '\r%s%%' % ((n + 1) * 100 / metatiles_n)
        sys.stdout.flush()
    print

def iterate_overview_jobs(level, out_dir):
    src_level = level + 1
    existing_src_tiles = []
    for src_tile_name in os.listdir(out_dir):
        z, x, y = map(int, re.match(r'(\d+)_(\d+)_(\d+)\.png', src_tile_name).groups())
        if z == src_level:
            existing_src_tiles.append((x, y))
    src_tile_coords = zip(*existing_src_tiles)
    ovr_min_tx = min(src_tile_coords[0]) / 2
    ovr_min_ty = min(src_tile_coords[1]) / 2 
    ovr_max_tx = max(src_tile_coords[0]) / 2
    ovr_max_ty = max(src_tile_coords[1]) / 2
    for tx in xrange(ovr_min_tx, ovr_max_tx+1):
        for ty in xrange(ovr_min_ty, ovr_max_ty + 1):
            src_tiles = []
            for dx, dy in [(0,0), (0, 1), (1, 0), (1,1)]:
                tx2 = tx * 2 + dx
                ty2 = ty * 2 + dy
                if (tx2, ty2) in existing_src_tiles:
                    tile_name = '%s_%s_%s.png' % (src_level, tx2, ty2)
                    tile_name = os.path.join(out_dir, tile_name)
                    src_tiles.append(tile_name)
                else:
                    src_tiles.append(None)
                if any(src_tiles):
                    ovr_filename = '%s_%s_%s.png' % (level, tx, ty)
                    ovr_filename = os.path.join(out_dir, ovr_filename)
                    yield ovr_filename,  src_tiles
            
def build_overview_tile(src_tiles_names, target_tile_png, low_quality):
    im = Image.new('RGBA', (512, 512))
    if src_tiles_names[0]:
        im.paste(Image.open(src_tiles_names[0]), (0, 0))
    if src_tiles_names[1]:
        im.paste(Image.open(src_tiles_names[1]), (256, 0))
    if src_tiles_names[2]:
        im.paste(Image.open(src_tiles_names[2]), (0, 256))
    if src_tiles_names[3]:
        im.paste(Image.open(src_tiles_names[3]), (256, 256))
    resampler = Image.ANTIALIAS if not low_quality else Image.NEAREST
    im = im.resize((256, 256), resampler)
    alpha = im.split()[3]
    extrema = alpha.getextrema()
    if extrema[1] > 0: # the image is not fully transparent
        if extrema[0] == 255: # the image is fully opaque
            im = im.convert('RGB')
        im.save(target_tile_png)
        


def build_overviews(max_level, out_dir, low_qiality):
    for level in xrange(max_level-1, -1, -1):
        print 'Overview', level
        overview_jobs = list(iterate_overview_jobs(level, out_dir))
        overviews_n = len(overview_jobs)
#        for n, (ovr_filename, src_tiles) in enumerate(overview_jobs):
#            build_overview_tile(src_tiles, ovr_filename, low_qiality)
        for n, _ in enumerate(mpimap(build_overview_tile, overview_jobs, low_qiality=low_qiality)):
            print '\r%s%%' % ((n + 1) * 100 / overviews_n)
        sys.stdout.flush()
    pass

def optimize_png(png_name):
    shell_execute(['pngnq', '-e', '.png_', png_name], check=True)
    os.rename(png_name + '_', png_name)

def optimize_tiles(dir_path):
    files = os.listdir(dir_path)
    files = [(os.path.join(dir_path, filename),) for filename in files]
    files_n = len(files)
#    for n, filename in enumerate(files):
#        optimize_png(filename)
    for n, _ in enumerate(mpimap(optimize_png, files)):
        print '\r%s%%' % ((n + 1) * 100 / files_n)
        sys.stdout.flush()
    print

def parge_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low-quality', action='store_true')
    parser.add_argument('--no-size-optimize', action='store_true')
    parser.add_argument('--border', metavar='FILE', dest='border_filename')
    parser.add_argument('--vmap', metavar='DIR', dest='vmaps_dir', required=True)
    parser.add_argument('--out', metavar='DIR', dest='out_dir', required=True)
    args = parser.parse_args()
    return args
    
def main():
    args = parge_args()
    metatile_level = 9
    if args.low_quality:
        max_level= 12
    else:
        max_level= 14
    
    if args.border_filename:    
        border = load_border(args.border_filename)
    else:
        border = None
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    else:
        for fn in glob.glob(os.path.join(args.out_dir, '*')):
            shutil.rmtree(fn, onerror=lambda _, fn, __: os.remove(fn) if os.path.exists(fn) else 0)
        
    print 'Building level', max_level
    build_max_level(args.vmaps_dir, max_level, metatile_level, border, args.out_dir, 
                    args.low_quality)
    print 'Building overviews'
    build_overviews(max_level, args.out_dir, args.low_quality)
    
    if not args.no_size_optimize:
        print 'Optimizing size'
        optimize_tiles(args.out_dir)

main()