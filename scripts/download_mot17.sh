#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# download_mot17.sh — download and extract MOT17 sequences for this project
#
# Default (no flags): extracts only MOT17-04-FRCNN and MOT17-09-FRCNN (~350 MB)
# With --all:         extracts all train + test sequences (~5 GB)
#
# Source: PaddleDetection mirror of MOT17 (official motchallenge.net is often
# unreachable). Layout inside zip: MOT17/images/{train,test}/<seq>/...
# We strip that prefix so files land under data/mot17/train/ and data/mot17/test/
#
# Usage:
#   bash scripts/download_mot17.sh          # partial (dev + held-out only)
#   bash scripts/download_mot17.sh --all    # full benchmark
# -----------------------------------------------------------------------------

set -euo pipefail

MIRROR_URL="https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip"
ZIP_SHA256="4253cf596550847a74f58859fee6a1263a03c5bd946ec9545c0119e8e5e5e800"
ZIP_FILE="/tmp/MOT17.zip"

# Sequences needed by the project (SDP variant in zip, renamed to FRCNN)
DEV_SEQ="MOT17-04"
HELD_SEQ="MOT17-09"

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$PROJECT_ROOT/data/mot17"

ALL=false
if [[ "${1:-}" == "--all" ]]; then
    ALL=true
fi


# helpers

check_cmd() {
    command -v "$1" &>/dev/null || { echo "ERROR: '$1' not found. Please install it."; exit 1; }
}

verify_sha256() {
    local file="$1" expected="$2"
    if command -v sha256sum &>/dev/null; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum &>/dev/null; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        echo "WARNING: cannot verify SHA-256 (neither sha256sum nor shasum found)"
        return 0
    fi
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: SHA-256 mismatch!"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        exit 1
    fi
    echo "SHA-256 verified OK."
}

patch_seqinfo() {
    # Replace name=MOT17-XX-SDP with name=MOT17-XX-FRCNN in seqinfo.ini
    local ini="$1"
    if [[ -f "$ini" ]]; then
        sed -i.bak 's/name=MOT17-\([0-9]*\)-SDP/name=MOT17-\1-FRCNN/' "$ini" && rm -f "${ini}.bak"
    fi
}


# checks

check_cmd curl
check_cmd unzip


# download

if [[ -f "$ZIP_FILE" ]]; then
    echo "Zip already present at $ZIP_FILE — skipping download."
    echo "Verifying existing file..."
    verify_sha256 "$ZIP_FILE" "$ZIP_SHA256"
else
    echo "Downloading MOT17 from PaddleDetection mirror (~2.4 GB)..."
    curl -L --fail --progress-bar -o "$ZIP_FILE" "$MIRROR_URL"
    echo "Download complete. Verifying SHA-256..."
    verify_sha256 "$ZIP_FILE" "$ZIP_SHA256"
fi

# extract

mkdir -p "$OUT_DIR/train" "$OUT_DIR/test"

extract_seq() {
    local split="$1"   # train or test
    local sdp_name="$2"  # e.g. MOT17-04-SDP
    local out_name="$3"  # e.g. MOT17-04-FRCNN
    local out_path="$OUT_DIR/$split/$out_name"

    if [[ -d "$out_path/img1" ]]; then
        echo "  $out_name already extracted — skipping."
        return
    fi

    echo "  Extracting $sdp_name -> $out_name ..."
    local prefix="MOT17/images/$split/$sdp_name/"

    # Extract into a temp dir, then move to final location
    local tmp="$OUT_DIR/.tmp_extract"
    rm -rf "$tmp"
    mkdir -p "$tmp"

    unzip -q "$ZIP_FILE" "${prefix}*" -d "$tmp" || true

    if [[ -d "$tmp/$prefix" ]]; then
        mv "$tmp/$prefix" "$out_path"
    elif [[ -d "$tmp/MOT17/images/$split/$sdp_name" ]]; then
        mv "$tmp/MOT17/images/$split/$sdp_name" "$out_path"
    else
        echo "ERROR: expected path $prefix not found in zip. Check zip layout."
        rm -rf "$tmp"
        exit 1
    fi

    rm -rf "$tmp"
    patch_seqinfo "$out_path/seqinfo.ini"
    echo "  Done -> $out_path"
}

if $ALL; then
    echo "Extracting all train sequences..."
    # List all train sequences in the zip and extract each
    all_seqs=$(unzip -l "$ZIP_FILE" | grep -oE 'MOT17/images/train/MOT17-[0-9]+-SDP/' | sort -u | sed 's|MOT17/images/train/||;s|/||')
    for sdp in $all_seqs; do
        frcnn="${sdp/-SDP/-FRCNN}"
        extract_seq "train" "$sdp" "$frcnn"
    done

    echo "Extracting test sequences..."
    test_seqs=$(unzip -l "$ZIP_FILE" | grep -oE 'MOT17/images/test/MOT17-[0-9]+-SDP/' | sort -u | sed 's|MOT17/images/test/||;s|/||')
    for sdp in $test_seqs; do
        frcnn="${sdp/-SDP/-FRCNN}"
        extract_seq "test" "$sdp" "$frcnn"
    done
else
    echo "Extracting dev and held-out sequences only..."
    extract_seq "train" "${DEV_SEQ}-SDP"   "${DEV_SEQ}-FRCNN"
    extract_seq "train" "${HELD_SEQ}-SDP"  "${HELD_SEQ}-FRCNN"
fi

# verify

echo ""
echo "Verifying extraction..."
ok=true
for seq in "${DEV_SEQ}-FRCNN" "${HELD_SEQ}-FRCNN"; do
    seq_path="$OUT_DIR/train/$seq"
    if [[ ! -d "$seq_path/img1" ]]; then
        echo "  FAIL: $seq_path/img1 missing"
        ok=false
    elif [[ ! -f "$seq_path/gt/gt.txt" ]]; then
        echo "  FAIL: $seq_path/gt/gt.txt missing"
        ok=false
    else
        n_imgs=$(ls "$seq_path/img1/"*.jpg 2>/dev/null | wc -l | tr -d ' ')
        echo "  OK: $seq — $n_imgs frames, gt.txt present"
    fi
done

if $ok; then
    echo ""
    echo "MOT17 setup complete. Sequences are at:"
    echo "  $OUT_DIR/train/${DEV_SEQ}-FRCNN   (dev)"
    echo "  $OUT_DIR/train/${HELD_SEQ}-FRCNN  (held-out)"
else
    echo "One or more sequences failed verification."
    exit 1
fi
