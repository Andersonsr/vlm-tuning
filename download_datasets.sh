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
# ImageNet needs an account at image-net.org, so it is not downloaded: point IMAGENET_DIR at an existing copy with
# train/ and val/ (1,000 class folders each) and it is linked into $DATA/imagenet/images; its classnames.txt (used by
# the four ImageNet variants too) is downloaded either way.
# The split_zhou_*.json files, classnames.txt and UCF101 are on Google Drive and need gdown (pip install gdown).
# Where the original links are dead, known mirrors are tried (Caltech data repository, Zenodo, HuggingFace).
# No longer downloadable (checked 2026-10): StanfordCars (ai.stanford.edu) and SUN397 (vision.princeton.edu); the
# script says which files to put where, and a re-run finishes them.

set -uo pipefail

DATA="${DATA:-}"
ALL=(caltech101 oxford_pets stanford_cars flowers102 food101 fgvc_aircraft sun397 dtd eurosat ucf101
     imagenet imagenetv2 imagenet_sketch imagenet_a imagenet_r)
SELECTED=()

while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --list) printf '%s\n' "${ALL[@]}"; exit 0 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
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
    for url in "$@"; do
        log "  downloading $url"
        if curl $([ -t 2 ] && echo -# || echo -sS) -L --fail --connect-timeout 30 --retry 3 --retry-delay 5 -C - -o "$out" "$url"; then
            touch "$out.ok"; return 0
        fi
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
    case "$1" in
        *.zip) unzip -q -o "$1" -d "$2" ;;
        *) tar -xf "$1" -C "$2" ;;
    esac
}

done_archive() { [ "${KEEP_ARCHIVES:-0}" = 1 ] || rm -f "$1" "$1.ok"; }

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
    # the ai.stanford.edu links have been down for a while; then get the files elsewhere (e.g. Kaggle) into $d
    if ! { fetch cars_train.tgz http://ai.stanford.edu/~jkrause/car196/cars_train.tgz &&
           fetch cars_test.tgz http://ai.stanford.edu/~jkrause/car196/cars_test.tgz &&
           fetch car_devkit.tgz https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz &&
           fetch cars_test_annos_withlabels.mat http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat; }; then
        log "  StanfordCars is no longer served by Stanford: put cars_train/, cars_test/, devkit/ and"
        log "  cars_test_annos_withlabels.mat in $d by hand (e.g. from Kaggle), then re-run for the split file"
        gfetch "$d/split_zhou_StanfordCars.json" 1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT
        return 1
    fi
    for f in cars_train cars_test car_devkit; do extract "$DL/$f.tgz" "$d"; done_archive "$DL/$f.tgz"; done
    cp "$DL/cars_test_annos_withlabels.mat" "$d/"; done_archive "$DL/cars_test_annos_withlabels.mat"
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
    fetch food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    extract "$DL/food-101.tar.gz" "$DATA"; done_archive "$DL/food-101.tar.gz"
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
    # vision.princeton.edu no longer serves SUN397, and the HuggingFace copies drop the file names the split refers to
    if ! { fetch SUN397.tar.gz http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz &&
           fetch Partitions.zip https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip; }; then
        log "  SUN397 is offline at Princeton: put SUN397.tar.gz and Partitions.zip obtained elsewhere in $DL,"
        log "  touch SUN397.tar.gz.ok and Partitions.zip.ok there, and re-run"
        return 1
    fi
    extract "$DL/SUN397.tar.gz" "$d"; done_archive "$DL/SUN397.tar.gz"
    extract "$DL/Partitions.zip" "$d"; done_archive "$DL/Partitions.zip"
    if [ -d "$d/Partitions" ]; then mv "$d"/Partitions/*.txt "$d/" && rmdir "$d/Partitions"; fi  # the .txt files go in sun397/
    [ -d "$d/SUN397" ]
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
            ln -sfn "$(cd "$IMAGENET_DIR/$split" && pwd)" "$d/images/$split"
        done
        log "  linked $IMAGENET_DIR/{train,val} into $d/images"
    else
        log "  ImageNet needs an image-net.org account: extract train/ and val/ (class folders) into $d/images,"
        log "  or re-run with IMAGENET_DIR=<folder with train/ and val/> to link an existing copy"
        return 1
    fi
    [ -d "$d/images/train" ] && [ -d "$d/images/val" ]
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
