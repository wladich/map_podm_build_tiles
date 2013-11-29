# -*- coding: utf-8 -*-
import sys
import os
import  pysqlite2.dbapi2 as sqlite

mbtiles_scheme = '''
CREATE TABLE tiles(zoom_level integer, tile_column integer, tile_row integer, tile_data blob);
CREATE UNIQUE INDEX idx_tiles ON tiles(zoom_level, tile_column, tile_row);
CREATE INDEX idx_z ON tiles(zoom_level);
'''

if len(sys.argv) != 3:
    print 'Usage: tiles_to_mbtile.py tiles/dir out_tile.mb'
    exit(1)
    
tiles_dir, tiles_db = sys.argv[1:]
if os.path.exists(tiles_db):
    os.remove(tiles_db)
    
conn = sqlite.connect(tiles_db)
conn.executescript(mbtiles_scheme)
files = os.listdir(tiles_dir)
for n, filename in enumerate(files):
    z, y, x = filename.rsplit('.', 1)[0].split('_')
    y = 2 ** int(z) - int(y) - 1
    with open(os.path.join(tiles_dir, filename)) as f:
        s = buffer(f.read())
    conn.execute('INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?,?,?,?)', (z, x, y, s))
    print '\r %s%%' % int(100 * n /  len(files)),
conn.commit()
conn.close()


