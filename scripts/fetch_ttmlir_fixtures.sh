#!/usr/bin/env bash
# Fetch the pinned tt-mlir release wheel and the tagged syntax fixtures into
# .deps/ttmlir/. The wheel is the parser gate for emitted TTIR; the fixture
# files are the syntax evidence quoted by docs/ttir.md. The script is safe to
# re-run: when the cache matches the lock file it exits without the network.
#
# Run it inside the project environment, which provides curl, sha256sum, and
# python3.11:
#
#   devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh
#
# Nothing here needs a Tenstorrent card. The wheel is extracted to
# .deps/ttmlir/site; the Python parse test puts that directory on PYTHONPATH
# and runs it under python3.11, which the wheel is built for.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

lock_file="$repo_root/deps/tt-mlir.lock"
deps_dir="$repo_root/.deps/ttmlir"
download_dir="$deps_dir/downloads"
fixtures_dir="$deps_dir/fixtures"
site_dir="$deps_dir/site"
site_stamp="$site_dir/.ttmlir-wheel"

fail() {
    echo "error: $*" >&2
    exit 1
}

require_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail "required tool '$tool' is not on PATH; run this script as 'devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh'"
    fi
}

sha256_of() {
    sha256sum "$1" | cut -d' ' -f1
}

# Verify a file without reaching the network. A digest mismatch removes the
# file so the download step fetches a clean copy. The digest comparison is the
# only reason the cache is trusted.
cache_matches() {
    local path="$1"
    local expected="$2"
    if [[ ! -f "$path" ]]; then
        return 1
    fi
    local actual
    actual="$(sha256_of "$path")"
    if [[ "$actual" == "$expected" ]]; then
        return 0
    fi
    echo "warning: digest mismatch for $path, removing it" >&2
    rm -f "$path"
    return 1
}

# Download one URL to one destination, verify the digest, and only then move it
# into place so an interrupted download never looks like a cached file.
download_to() {
    local url="$1"
    local expected_sha256="$2"
    local dest="$3"

    if cache_matches "$dest" "$expected_sha256"; then
        return 0
    fi
    echo "downloading $url"
    mkdir -p "$(dirname "$dest")"
    local partial="$dest.partial"
    rm -f "$partial"
    if ! curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 \
        --output "$partial" "$url"; then
        rm -f "$partial"
        fail "download failed: $url"
    fi
    local actual
    actual="$(sha256_of "$partial")"
    if [[ "$actual" != "$expected_sha256" ]]; then
        rm -f "$partial"
        fail "digest mismatch for $(basename "$dest"): expected $expected_sha256, got $actual"
    fi
    mv "$partial" "$dest"
}

# Extract the wheel to the site directory. The stamp records which wheel the
# directory came from, so a pin change re-extracts and a re-run does not.
extract_wheel() {
    local stamp=""
    if [[ -f "$site_stamp" ]]; then
        stamp="$(cat "$site_stamp")"
    fi
    if [[ "$stamp" == "$TT_MLIR_WHEEL" && -f "$site_dir/ttmlir/ir.py" ]]; then
        return 0
    fi
    echo "extracting $TT_MLIR_WHEEL into .deps/ttmlir/site"
    rm -rf "$site_dir"
    mkdir -p "$site_dir"
    if ! python3.11 -m zipfile -e "$wheel_path" "$site_dir"; then
        rm -rf "$site_dir"
        fail "could not extract $wheel_path"
    fi
    if [[ ! -f "$site_dir/ttmlir/ir.py" ]]; then
        rm -rf "$site_dir"
        fail "extracted wheel is missing ttmlir/ir.py"
    fi
    printf '%s' "$TT_MLIR_WHEEL" > "$site_stamp"
}

for tool in curl sha256sum python3.11 mktemp; do
    require_tool "$tool"
done

[[ -f "$lock_file" ]] || fail "missing lock file $lock_file"
# shellcheck source=/dev/null
source "$lock_file"

for variable in TT_MLIR_TAG TT_MLIR_COMMIT TT_MLIR_RELEASE_BASE \
    TT_MLIR_WHEEL TT_MLIR_WHEEL_SHA256 TT_MLIR_RAW_BASE TT_MLIR_FILES; do
    if [[ -z "${!variable:-}" ]]; then
        fail "lock file $lock_file does not define $variable"
    fi
done

mkdir -p "$download_dir" "$fixtures_dir"
wheel_path="$download_dir/$TT_MLIR_WHEEL"

# The wheel holds the real parser, so its digest is checked on every fetch.
download_to "$TT_MLIR_RELEASE_BASE/$TT_MLIR_WHEEL" "$TT_MLIR_WHEEL_SHA256" "$wheel_path"

# Fetch each syntax fixture at the tagged commit. The fixture list lives in the
# lock file so a pin change is one edit, not a script edit.
fetched=0
cached=0
while read -r expected_sha256 relative_path; do
    # The lock file list is indented and may contain blank lines.
    if [[ -z "${expected_sha256:-}" ]]; then
        continue
    fi
    if [[ -z "${relative_path:-}" ]]; then
        fail "malformed TT_MLIR_FILES entry for digest $expected_sha256"
    fi
    destination="$fixtures_dir/$relative_path"
    if cache_matches "$destination" "$expected_sha256"; then
        cached=$((cached + 1))
        continue
    fi
    download_to "$TT_MLIR_RAW_BASE/$relative_path" "$expected_sha256" "$destination"
    fetched=$((fetched + 1))
done <<< "$TT_MLIR_FILES"

extract_wheel

echo "tt-mlir $TT_MLIR_TAG is cached in .deps/ttmlir (fixtures fetched: $fetched, cached: $cached)"
echo "hint: python3 -m pytest python/tensixflow/tests/test_ttir_parse.py runs the parser gate"
