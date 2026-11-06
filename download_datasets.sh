#!/usr/bin/env bash
# Downloads the few-shot classification benchmarks of CLIP-LoRA / CoOp
# (https://github.com/MaxZanella/CLIP-LoRA/blob/main/DATASETS.md) into the layout their loaders expect:
#
#   $DATA/caltech-101/  oxford_pets/  stanford_cars/  oxford_flowers/  food-101/  fgvc_aircraft/  sun397/  dtd/
#   $DATA/eurosat/  ucf101/  imagenet/  imagenetv2/  imagenet-sketch/  imagenet-adversarial/  imagenet-rendition/
#
# Usage:
#   bash download_datasets.sh --data /path/to/DATA                  # every dataset
#   bash download_datasets.sh --data /path/to/DATA dtd eurosat      # only these
#   bash download_datasets.sh --list
#
# Re-running is safe: finished datasets (marked with .complete) are skipped and interrupted downloads resume.
# Archives are deleted after extraction (KEEP_ARCHIVES=1 to keep them in $DATA/_downloads).
#
# ImageNet: point IMAGENET_DIR at an existing copy with train/ and val/ (1,000 class folders each) and it is linked into
# $DATA/imagenet/images. Without it only val/ is downloaded (the 50,000 validation images, about 6.4 GB), read out of
# the 155 GB zip of the Kaggle competition imagenet-object-localization-challenge: that needs a Kaggle API token
# (~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY) and the competition's rules accepted once on its website.
# Its classnames.txt (used by the four ImageNet variants too) is downloaded either way.
# The split_zhou_*.json files, classnames.txt and UCF101 are on Google Drive and need gdown (pip install gdown).
# Where the original links are dead, known mirrors are tried (Caltech data repository, Zenodo, HuggingFace, GitHub,
# Kaggle). SUN397 needs about 80 GB free while it is unpacked (39 GB at the end).

set -uo pipefail

DATA="${DATA:-}"
ALL=(caltech101 oxford_pets stanford_cars flowers102 food101 fgvc_aircraft sun397 dtd eurosat ucf101
     imagenet imagenetv2 imagenet_sketch imagenet_a imagenet_r)
SELECTED=()

while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --list) printf '%s\n' "${ALL[@]}"; exit 0 ;;
        -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
        *) SELECTED+=("$1"); shift ;;
    esac
done
[ -n "$DATA" ] || { echo "set the destination with --data DIR (or DATA=DIR)"; exit 1; }
[ ${#SELECTED[@]} -gt 0 ] || SELECTED=("${ALL[@]}")
for name in "${SELECTED[@]}"; do
    [[ " ${ALL[*]} " == *" $name "* ]] || { echo "unknown dataset $name (--list)"; exit 1; }
done

mkdir -p "$DATA/_downloads"
DATA="$(cd "$DATA" && pwd)"
DL="$DATA/_downloads"

# ------------------------------------------------------------------------------------------------------- helpers

log() { echo "[$(date +%H:%M:%S)] $*"; }

# fetch FILE URL [URL...]: the first URL that works, resuming a partial download; FILE.ok marks a finished one
fetch() {
    local out="$DL/$1"; shift
    [ -f "$out.ok" ] && { log "  have $(basename "$out")"; return 0; }
    local before
    for url in "$@"; do
        log "  downloading $url"
        while true; do
            before=$(stat -c %s "$out" 2>/dev/null || echo 0)
            if curl $([ -t 2 ] && echo -# || echo -sS) -L --fail --connect-timeout 30 --retry 3 --retry-delay 5 -C - -o "$out" "$url"; then
                touch "$out.ok"; return 0
            fi
            # a connection dropped mid-transfer (which --retry does not cover): resume while the download progresses
            [ "$(stat -c %s "$out" 2>/dev/null || echo 0)" -gt "$before" ] || break
            log "  interrupted at $(( $(stat -c %s "$out") / 1000000 )) MB, resuming"; sleep 5
        done
        rm -f "$out"
        log "  failed: $url"
    done
    return 1
}

# gfetch DEST_FILE DRIVE_ID: a Google Drive file
gfetch() {
    local out="$1" id="$2"
    [ -s "$out" ] && return 0
    command -v gdown >/dev/null || { log "  gdown not found (pip install gdown)"; return 1; }
    log "  downloading $(basename "$out") from Google Drive"
    gdown "https://drive.google.com/uc?id=$id" -O "$out" || { rm -f "$out"; return 1; }
}

extract() {  # extract ARCHIVE DIR
    log "  extracting $(basename "$1")"
    if ! case "$1" in
        *.zip) unzip -q -o "$1" -d "$2" ;;
        *) tar -xf "$1" -C "$2" ;;
    esac; then  # a broken download (e.g. an error page saved under the archive's name): fetched again on a re-run
        log "  $(basename "$1") is not a valid archive ($(head -c 100 "$1" | tr -cd '[:print:]' | cut -c1-60)): deleted"
        rm -f "$1" "$1.ok"; return 1
    fi
}

done_archive() { [ "${KEEP_ARCHIVES:-0}" = 1 ] || rm -f "$1" "$1.ok"; }

PY=python3; "$PY" -c "" 2>/dev/null || PY=python  # for the scripts below (python3 may be a stub on Windows)

classnames() {  # ImageNet's class names (CLIP), into $DATA/imagenet and the variants
    mkdir -p "$DATA/imagenet"
    gfetch "$DATA/imagenet/classnames.txt" 1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF
    [ -z "${1:-}" ] || cp "$DATA/imagenet/classnames.txt" "$1/"
}

# ------------------------------------------------------------------------------------------------------ datasets

caltech101() {
    local d="$DATA/caltech-101"; mkdir -p "$d"
    if fetch 101_ObjectCategories.tar.gz \
        http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz; then
        extract "$DL/101_ObjectCategories.tar.gz" "$d"; done_archive "$DL/101_ObjectCategories.tar.gz"
    else  # the dataset moved to the Caltech data repository: a zip with the same tar.gz inside
        fetch caltech-101.zip "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
        extract "$DL/caltech-101.zip" "$DL"
        extract "$DL/caltech-101/101_ObjectCategories.tar.gz" "$d"
        rm -rf "$DL/caltech-101" "$DL/__MACOSX"; done_archive "$DL/caltech-101.zip"
    fi
    gfetch "$d/split_zhou_Caltech101.json" 1hyarUivQE36mY6jSomru6Fjd-JzwcCzN
    [ -d "$d/101_ObjectCategories" ]
}

oxford_pets() {
    local d="$DATA/oxford_pets"; mkdir -p "$d"
    for f in images annotations; do
        fetch pets_$f.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/$f.tar.gz
        extract "$DL/pets_$f.tar.gz" "$d"; done_archive "$DL/pets_$f.tar.gz"
    done
    gfetch "$d/split_zhou_OxfordPets.json" 1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs
    [ -d "$d/images" ] && [ -d "$d/annotations" ]
}

stanford_cars() {
    local d="$DATA/stanford_cars"; mkdir -p "$d"
    if fetch cars_train.tgz http://ai.stanford.edu/~jkrause/car196/cars_train.tgz &&
       fetch cars_test.tgz http://ai.stanford.edu/~jkrause/car196/cars_test.tgz &&
       fetch car_devkit.tgz https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz &&
       fetch cars_test_annos_withlabels.mat http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat; then
        for f in cars_train cars_test car_devkit; do extract "$DL/$f.tgz" "$d"; done_archive "$DL/$f.tgz"; done
        cp "$DL/cars_test_annos_withlabels.mat" "$d/"; done_archive "$DL/cars_test_annos_withlabels.mat"
    else  # ai.stanford.edu has been down for a while: a GitHub repository has the same files in stanford_cars/
        fetch StanfordCars-Dataset-main.tar.gz \
            https://github.com/jhpohovey/StanfordCars-Dataset/archive/refs/heads/main.tar.gz || return 1
        extract "$DL/StanfordCars-Dataset-main.tar.gz" "$DL"
        for f in "$DL"/StanfordCars-Dataset-main/stanford_cars/*; do rm -rf "$d/$(basename "$f")"; mv "$f" "$d/"; done
        rm -rf "$DL/StanfordCars-Dataset-main"; done_archive "$DL/StanfordCars-Dataset-main.tar.gz"
    fi
    gfetch "$d/split_zhou_StanfordCars.json" 1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT
    [ -d "$d/cars_train" ] && [ -d "$d/cars_test" ] && [ -d "$d/devkit" ]
}

flowers102() {
    local d="$DATA/oxford_flowers"; mkdir -p "$d"
    fetch 102flowers.tgz https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    extract "$DL/102flowers.tgz" "$d"; done_archive "$DL/102flowers.tgz"
    fetch imagelabels.mat https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
    cp "$DL/imagelabels.mat" "$d/"; done_archive "$DL/imagelabels.mat"
    gfetch "$d/cat_to_name.json" 1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0
    gfetch "$d/split_zhou_OxfordFlowers.json" 1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT
    [ -d "$d/jpg" ]
}

food101() {
    if fetch food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz && extract "$DL/food-101.tar.gz" "$DATA"; then
        done_archive "$DL/food-101.tar.gz"
    else  # behind some proxies ETH answers with a web page: a Kaggle copy has the same images/ (5.7 GB, no account)
        fetch food41.zip https://www.kaggle.com/api/v1/datasets/download/kmader/food41
        log "  extracting food41.zip (images/ and meta/, not the .h5 files)"
        mkdir -p "$DATA/food-101"; unzip -q -o "$DL/food41.zip" -x '*.h5' -d "$DATA/food-101"; done_archive "$DL/food41.zip"
    fi
    gfetch "$DATA/food-101/split_zhou_Food101.json" 1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl
    [ -d "$DATA/food-101/images" ]
}

fgvc_aircraft() {
    fetch fgvc-aircraft-2013b.tar.gz https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
    extract "$DL/fgvc-aircraft-2013b.tar.gz" "$DL"
    rm -rf "$DATA/fgvc_aircraft"; mv "$DL/fgvc-aircraft-2013b/data" "$DATA/fgvc_aircraft"  # only data/ is kept
    rm -rf "$DL/fgvc-aircraft-2013b"; done_archive "$DL/fgvc-aircraft-2013b.tar.gz"
    [ -d "$DATA/fgvc_aircraft/images" ]
}

sun397() {
    local d="$DATA/sun397"; mkdir -p "$d"
    gfetch "$d/split_zhou_SUN397.json" 1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq
    if [ ! -d "$d/SUN397" ]; then
        # vision.princeton.edu no longer serves SUN397, and the HuggingFace copies drop the file names the split refers
        # to. A Kaggle copy of the 108,754 original images (a 39 GB zip, no account needed) is flat
        # (images/abbey_sun_x.jpg) with each image's class path in a CSV: the images are moved back into SUN397's folders
        fetch sun397.zip https://www.kaggle.com/api/v1/datasets/download/alifshahariar/sun397-training-ready-dataset-csv-labels
        rm -rf "$DL/sun397-processed"
        extract "$DL/sun397.zip" "$DL"
        log "  moving the images into SUN397's original folders"
        "$PY" -I - "$DL/sun397-processed" "$d/SUN397.partial" <<'EOF'
import csv, os, sys
src, dst = sys.argv[1], sys.argv[2]
moved = 0
with open(os.path.join(src, 'sun397-image-labels.csv'), newline='') as f:
    for row in csv.DictReader(f):  # image: abbey_sun_x.jpg or apartment_building_outdoor_sun_x.jpg, class_name: abbey or apartment_building/outdoor
        name, cls = row['image'], row['class_name']
        prefix = cls.replace('/', '_') + '_'
        assert name.startswith(prefix), (name, cls)
        out_dir = os.path.join(dst, cls[0], *cls.split('/'))
        os.makedirs(out_dir, exist_ok=True)
        os.replace(os.path.join(src, 'images', name), os.path.join(out_dir, name[len(prefix):]))
        moved += 1
print(f'moved {moved} images')
EOF
        mv "$d/SUN397.partial" "$d/SUN397"
        rm -rf "$DL/sun397-processed"; done_archive "$DL/sun397.zip"
    fi
    log "  checking the images against the CoOp split"
    "$PY" -I - "$d" <<'EOF'
import json, os, sys
root = sys.argv[1]
images = sum(len(files) for _, _, files in os.walk(os.path.join(root, 'SUN397')))
split = json.load(open(os.path.join(root, 'split_zhou_SUN397.json')))
missing = [p for s in ('train', 'val', 'test') for p, _, _ in split[s] if not os.path.isfile(os.path.join(root, 'SUN397', p))]
print(f'{images} images (expected 108754); split: {len(split["train"])} train, {len(split["val"])} val, '
      f'{len(split["test"])} test, {len(missing)} missing')
sys.exit(1 if missing or images != 108754 else 0)
EOF
}

dtd() {
    fetch dtd-r1.0.1.tar.gz https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    extract "$DL/dtd-r1.0.1.tar.gz" "$DATA"; done_archive "$DL/dtd-r1.0.1.tar.gz"
    gfetch "$DATA/dtd/split_zhou_DescribableTextures.json" 1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x
    [ -d "$DATA/dtd/images" ]
}

eurosat() {
    local d="$DATA/eurosat"; mkdir -p "$d"
    # the DFKI link is often down; the Zenodo copy has the same RGB images in EuroSAT_RGB/
    fetch EuroSAT.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1" \
        http://madm.dfki.de/files/sentinel/EuroSAT.zip
    extract "$DL/EuroSAT.zip" "$d"; done_archive "$DL/EuroSAT.zip"
    if [ ! -d "$d/2750" ] && [ -d "$d/EuroSAT_RGB" ]; then mv "$d/EuroSAT_RGB" "$d/2750"; fi
    gfetch "$d/split_zhou_EuroSAT.json" 1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o
    [ -d "$d/2750" ]
}

ucf101() {
    local d="$DATA/ucf101"; mkdir -p "$d"
    gfetch "$DL/UCF-101-midframes.zip" 10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O
    extract "$DL/UCF-101-midframes.zip" "$d"; rm -rf "$d/__MACOSX"; done_archive "$DL/UCF-101-midframes.zip"
    gfetch "$d/split_zhou_UCF101.json" 1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y
    [ -d "$d/UCF-101-midframes" ]
}

imagenet() {
    local d="$DATA/imagenet"; mkdir -p "$d/images"
    classnames
    if [ -n "${IMAGENET_DIR:-}" ]; then
        for split in train val; do
            [ -d "$IMAGENET_DIR/$split" ] || { log "  no $IMAGENET_DIR/$split"; return 1; }
            # e.g. the Kaggle copy's ILSVRC/Data/CLS-LOC/val is flat (its classes are in LOC_val_solution.csv)
            [ "$(find "$IMAGENET_DIR/$split/" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1000 ] ||
                { log "  $IMAGENET_DIR/$split does not have 1,000 class folders"; return 1; }
            ln -sfn "$(cd "$IMAGENET_DIR/$split" && pwd)" "$d/images/$split"
        done
        log "  linked $IMAGENET_DIR/{train,val} into $d/images"
    elif [ ! -d "$d/images/val" ]; then
        # only the validation split, read out of the Kaggle competition's 155 GB zip with range requests: the zip's
        # central directory gives where the 50,000 validation images are, and only that part of the zip is downloaded
        log "  downloading the ImageNet validation split from Kaggle (no train/: set IMAGENET_DIR for it)"
        "$PY" -I - "$d/images/val" "$DL/imagenet-val.part" <<'EOF'
import base64, csv, http.client, io, json, os, sys, time, urllib.error, urllib.request, zipfile
URL = 'https://www.kaggle.com/api/v1/competitions/data/download-all/imagenet-object-localization-challenge'
RULES = 'https://www.kaggle.com/c/imagenet-object-localization-challenge/rules'
out, part = sys.argv[1], sys.argv[2]

user, key = os.environ.get('KAGGLE_USERNAME'), os.environ.get('KAGGLE_KEY')
if not (user and key):
    token = os.path.join(os.environ.get('KAGGLE_CONFIG_DIR', os.path.expanduser('~/.kaggle')), 'kaggle.json')
    if not os.path.isfile(token):
        sys.exit(f'no Kaggle API token: create one at https://www.kaggle.com/settings (API, Create New Token) and save it '
                 f'as {token}, or set KAGGLE_USERNAME and KAGGLE_KEY')
    with open(token) as f:
        c = json.load(f)
    user, key = c['username'], c['key']
auth = {'Authorization': 'Basic ' + base64.b64encode(f'{user}:{key}'.encode()).decode()}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None

def resolve():  # Kaggle redirects to a signed storage URL that expires, so it is asked again after a failure
    global url, headers
    try:
        urllib.request.build_opener(NoRedirect).open(urllib.request.Request(URL, headers={**auth, 'Range': 'bytes=0-0'}))
        url, headers = URL, auth  # served directly
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            url, headers = e.headers['Location'], {}
        elif e.code in (401, 403):
            sys.exit(f'Kaggle refused the download (HTTP {e.code}): check the API token and accept the rules at {RULES}')
        else:
            raise

def get(start, end):  # bytes start..end (inclusive) of the zip
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={**headers, 'Range': f'bytes={start}-{end}'}),
                                          timeout=60)
        except (OSError, http.client.HTTPException) as e:
            print(f'  {e}, retrying', flush=True); time.sleep(10); resolve()
    sys.exit('Kaggle is not answering: re-run to resume')

def read(start, end):  # the same, read whole, also retrying a transfer cut short
    for attempt in range(5):
        try:
            with get(start, end) as r:
                return r.read()
        except (OSError, http.client.HTTPException) as e:
            print(f'  {e}, retrying', flush=True); time.sleep(10); resolve()
    sys.exit('Kaggle is not answering: re-run to resume')

class RemoteZip(io.RawIOBase):  # the zip as a seekable file: range requests, or the downloaded part once there is one
    def __init__(self):
        self.pos, self.local = 0, None
        self.size = int(get(0, 0).headers['Content-Range'].split('/')[1])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if self.local and self.start <= self.pos and self.pos + n <= self.start + self.length:
            self.local.seek(self.pos - self.start); data = self.local.read(n)
        else:
            data = read(self.pos, self.pos + n - 1)
        b[:len(data)] = data; self.pos += len(data)
        return len(data)

resolve()
remote = RemoteZip()
z = zipfile.ZipFile(io.BufferedReader(remote, 1 << 20))  # reads the central directory (about 1.4M entries)
val = [i for i in z.infolist() if i.filename.startswith('ILSVRC/Data/CLS-LOC/val/') and not i.is_dir()]
labels = {row['ImageId']: row['PredictionString'].split()[0]  # ImageId,PredictionString = "<wnid> x y x y ..."
          for row in csv.DictReader(io.TextIOWrapper(z.open('LOC_val_solution.csv'), newline=''))}
assert len(val) == len(labels) == 50000, (len(val), len(labels))

# the validation images are stored next to each other: download from the first one's header to the last one's end
# (plus room for the local headers' extra fields, which the central directory does not give)
remote.start = min(i.header_offset for i in val)
end = min(max(i.header_offset + 30 + len(i.orig_filename.encode()) + i.compress_size for i in val) + (1 << 16), remote.size)
remote.length = end - remote.start
images = sum(i.compress_size for i in val)
if remote.length > 2 * images:
    sys.exit(f'the validation images are spread over {remote.length / 1e9:.0f} GB of the zip, not {images / 1e9:.1f} GB: '
             f'the zip is not laid out as expected')
for attempt in range(10):
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have >= remote.length:
        break
    print(f'  {have / 1e9:.1f} of {remote.length / 1e9:.1f} GB', flush=True)
    try:
        with get(remote.start + have, end - 1) as r, open(part, 'ab') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                have += len(chunk)
                if have % (1 << 30) < len(chunk):
                    print(f'  {have / 1e9:.1f} of {remote.length / 1e9:.1f} GB', flush=True)
    except (OSError, http.client.HTTPException) as e:
        print(f'  {e}, resuming', flush=True); time.sleep(10); resolve()
else:
    sys.exit('download failed: re-run to resume')

print('  extracting into class folders', flush=True)
remote.local = open(part, 'rb')
for i in val:
    name = os.path.basename(i.filename)
    os.makedirs(os.path.join(out + '.partial', labels[name.split('.')[0]]), exist_ok=True)
    with open(os.path.join(out + '.partial', labels[name.split('.')[0]], name), 'wb') as f:
        f.write(z.read(i))
os.replace(out + '.partial', out)
classes = os.listdir(out)
print(f'  {sum(len(os.listdir(os.path.join(out, c))) for c in classes)} images in {len(classes)} classes')
sys.exit(0 if len(classes) == 1000 else 1)
EOF
        done_archive "$DL/imagenet-val.part"
    fi
    [ "$(find -L "$d/images/val/" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1000 ] ||
        { log "  $d/images/val does not have 1,000 class folders"; return 1; }
}

imagenetv2() {
    local d="$DATA/imagenetv2"; mkdir -p "$d"
    fetch imagenetv2-matched-frequency.tar.gz \
        https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz \
        https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
    extract "$DL/imagenetv2-matched-frequency.tar.gz" "$d"; done_archive "$DL/imagenetv2-matched-frequency.tar.gz"
    classnames "$d"
    [ -d "$d/imagenetv2-matched-frequency-format-val" ]
}

imagenet_sketch() {
    local d="$DATA/imagenet-sketch"; mkdir -p "$d"
    # the authors' copy on HuggingFace (the repository links to Google Drive / Kaggle); it extracts to sketch/
    fetch ImageNet-Sketch.zip https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip
    extract "$DL/ImageNet-Sketch.zip" "$d"; done_archive "$DL/ImageNet-Sketch.zip"
    if [ ! -d "$d/images" ] && [ -d "$d/sketch" ]; then mv "$d/sketch" "$d/images"; fi
    classnames "$d"
    [ -d "$d/images" ]
}

imagenet_a() {
    local d="$DATA/imagenet-adversarial"; mkdir -p "$d"
    fetch imagenet-a.tar https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
    extract "$DL/imagenet-a.tar" "$d"; done_archive "$DL/imagenet-a.tar"
    classnames "$d"
    [ -d "$d/imagenet-a" ]
}

imagenet_r() {
    local d="$DATA/imagenet-rendition"; mkdir -p "$d"
    fetch imagenet-r.tar https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
    extract "$DL/imagenet-r.tar" "$d"; done_archive "$DL/imagenet-r.tar"
    classnames "$d"
    [ -d "$d/imagenet-r" ]
}

# ---------------------------------------------------------------------------------------------------------- main

declare -A FOLDER=([caltech101]=caltech-101 [oxford_pets]=oxford_pets [stanford_cars]=stanford_cars
    [flowers102]=oxford_flowers [food101]=food-101 [fgvc_aircraft]=fgvc_aircraft [sun397]=sun397 [dtd]=dtd
    [eurosat]=eurosat [ucf101]=ucf101 [imagenet]=imagenet [imagenetv2]=imagenetv2
    [imagenet_sketch]=imagenet-sketch [imagenet_a]=imagenet-adversarial [imagenet_r]=imagenet-rendition)

ok=(); failed=()
for name in "${SELECTED[@]}"; do
    marker="$DATA/${FOLDER[$name]}/.complete"
    if [ -f "$marker" ]; then log "$name: already complete"; ok+=("$name"); continue; fi
    log "$name"
    ( set -e; "$name" )  # in a subshell with set -e: any failed step stops this dataset only
    if [ $? -eq 0 ]; then
        touch "$marker"; ok+=("$name"); log "$name: done"
    else
        failed+=("$name"); log "$name: FAILED (see above; re-run to resume)"
    fi
done

echo
log "complete: ${ok[*]:-none}"
[ ${#failed[@]} -eq 0 ] || { log "failed:   ${failed[*]}"; exit 1; }
