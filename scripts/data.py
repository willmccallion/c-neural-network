import os
import requests
import zipfile
import gzip
import shutil
import numpy as np
import struct
from tqdm import tqdm

# Configurations
DATA_DIR = "data"
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

# QuickDraw Categories
DRAWINGS = [
    "apple", "book", "candle", "cloud", "cup", "door", "envelope", 
    "eye", "fish", "guitar", "hammer", "hat", "ice cream", "leaf", "lightning", 
    "moon", "mountain", "star", "tent", "tree", "umbrella", "wheel"
]

def download_file(url, filepath):
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return False
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as bar:
            for data in response.iter_content(1024):
                f.write(data)
                bar.update(len(data))
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

def setup_emnist():
    print("\nSetting up EMNIST")

    # Target files
    img_path = os.path.join(DATA_DIR, "emnist-balanced-train-images-idx3-ubyte")
    lbl_path = os.path.join(DATA_DIR, "emnist-balanced-train-labels-idx1-ubyte")

    if os.path.exists(img_path) and os.path.exists(lbl_path):
        print("EMNIST files already present.")
        return

    # Download zip if missing
    zip_path = os.path.join(DATA_DIR, "emnist.zip")
    if not os.path.exists(zip_path):
        print("Downloading EMNIST (approx 530MB)... this may take a moment.")
        download_file(EMNIST_URL, zip_path)

    # Extract specific files from zip
    print("Extracting EMNIST...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # The zip structure is usually gzip/emnist-balanced-train-images-idx3-ubyte.gz
        for member in z.namelist():
            if "balanced-train-images" in member and member.endswith(".gz"):
                with z.open(member) as src, open(img_path + ".gz", 'wb') as dst:
                    shutil.copyfileobj(src, dst)
            elif "balanced-train-labels" in member and member.endswith(".gz"):
                with z.open(member) as src, open(lbl_path + ".gz", 'wb') as dst:
                    shutil.copyfileobj(src, dst)

    # Gunzip
    print("Decompressing...")
    for p in [img_path, lbl_path]:
        with gzip.open(p + ".gz", 'rb') as f_in:
            with open(p, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(p + ".gz") # Cleanup .gz

    # Optional: Cleanup big zip
    # os.remove(zip_path) 
    print("EMNIST Ready.")

def load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols)

def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def save_idx(images, labels, img_out, lbl_out):
    with open(img_out, 'wb') as f:
        f.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        f.write(images.tobytes())
    with open(lbl_out, 'wb') as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(labels.tobytes())

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Get EMNIST
    setup_emnist()

    # Load EMNIST
    print("\nLoading data")
    emnist_img_path = os.path.join(DATA_DIR, "emnist-balanced-train-images-idx3-ubyte")
    emnist_lbl_path = os.path.join(DATA_DIR, "emnist-balanced-train-labels-idx1-ubyte")

    emnist_imgs = load_idx_images(emnist_img_path)
    emnist_lbls = load_idx_labels(emnist_lbl_path)

    # Rotate EMNIST to match QuickDraw orientation (EMNIST is flipped/rotated by default)
    print("Normalizing EMNIST orientation...")
    emnist_imgs = emnist_imgs.reshape(-1, 28, 28)
    emnist_imgs = np.transpose(emnist_imgs, (0, 2, 1)) 
    emnist_imgs = emnist_imgs.reshape(-1, 784)

    all_images = [emnist_imgs]
    all_labels = [emnist_lbls]

    # Get QuickDraw
    print("\nDownloading drawings")
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    current_label_id = 47 
    
    for item in DRAWINGS:
        local_path = os.path.join(DATA_DIR, f"{item}.npy")

        # Download if missing or empty
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000:
            print(f"Downloading {item}...")
            # Try URL encoded space first
            safe_name = item.replace(" ", "%20")
            success = download_file(base_url + safe_name + ".npy", local_path)

            # Retry with underscore if failed
            if not success:
                print(f"  > Retrying with underscore...")
                safe_name = item.replace(" ", "_")
                success = download_file(base_url + safe_name + ".npy", local_path)

            if not success:
                print(f"  > FAILED to download {item}. Skipping.")
                continue

        try:
            # allow_pickle=True needed for some numpy versions, though these should be bitmaps
            data = np.load(local_path, allow_pickle=True)
            data = data[:4000] # Balance dataset
            labels = np.full(len(data), current_label_id, dtype=np.uint8)

            all_images.append(data)
            all_labels.append(labels)
            print(f"Added {item} (ID: {current_label_id})")
            current_label_id += 1
        except Exception as e:
            print(f"Error loading {item}: {e}")

    # Merge
    print("\nMerging and saving")
    final_images = np.concatenate(all_images)
    final_labels = np.concatenate(all_labels)

    # Shuffle
    perm = np.random.permutation(len(final_images))
    final_images = final_images[perm]
    final_labels = final_labels[perm]

    out_img = os.path.join(DATA_DIR, "extended-train-images-idx3-ubyte")
    out_lbl = os.path.join(DATA_DIR, "extended-train-labels-idx1-ubyte")

    save_idx(final_images, final_labels, out_img, out_lbl)
    print(f"SUCCESS! Saved to: {out_img}")
    print(f"Total Samples: {len(final_images)}")

if __name__ == "__main__":
    main()
