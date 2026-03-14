import os
import random
from pathlib import Path

ARTISTS = [
    'aerosmith', 'beatles', 'creedence_clearwater_revival', 'cure',
    'dave_matthews_band', 'depeche_mode', 'fleetwood_mac', 'garth_brooks',
    'green_day', 'led_zeppelin', 'madonna', 'metallica', 'prince', 'queen',
    'radiohead', 'roxette', 'steely_dan', 'suzanne_vega', 'tori_amos', 'u2'
]
ARTIST2IDX = {a: i for i, a in enumerate(ARTISTS)}

def get_split(train_val_dir, val_album_idx=4, seed=42):
    """Returns (train_files, val_files) as list of (path, label_idx)."""
    random.seed(seed)
    train_files, val_files = [], []
    for artist in ARTISTS:
        artist_dir = Path(train_val_dir) / artist
        albums = sorted(os.listdir(artist_dir))
        # Use last album as validation
        for i, album in enumerate(albums):
            album_dir = artist_dir / album
            mp3s = sorted(album_dir.glob("*.mp3"))
            for mp3 in mp3s:
                if i == val_album_idx - 1:  # 5th album = val
                    val_files.append((str(mp3), ARTIST2IDX[artist]))
                else:
                    train_files.append((str(mp3), ARTIST2IDX[artist]))
    return train_files, val_files

if __name__ == "__main__":
    train_val_dir = os.path.expanduser("~/CommE5070/hw1/hw1/artist20/train_val")
    train, val = get_split(train_val_dir)
    print(f"Train: {len(train)}, Val: {len(val)}")
