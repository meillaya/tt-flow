#!/usr/bin/env bash
# Fetch and extract the pinned TT-Metalium release artifacts and the ttsim
# simulator into .deps/. The script is safe to re-run: when the .deps cache is
# complete it exits without touching the network.
#
# Run it inside the project environment, which provides curl, ar, tar, zstd,
# and sha256sum:
#
#   devenv shell -- bash scripts/fetch_tt_metal.sh
#
# Everything lands in .deps/, which is gitignored. Nothing here needs a
# Tenstorrent card: this step only prepares headers, libraries, the JIT
# compiler, the pinned example sources, and the simulator shared library.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

lock_file="$repo_root/deps/tt-metal.lock"
deps_dir="$repo_root/.deps"
download_dir="$deps_dir/downloads"
ttsim_dir="$deps_dir/ttsim"

fail() {
    echo "error: $*" >&2
    exit 1
}

require_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail "required tool '$tool' is not on PATH; run this script as 'devenv shell -- bash scripts/fetch_tt_metal.sh'"
    fi
}

sha256_of() {
    sha256sum "$1" | cut -d' ' -f1
}

# Verify a file without reaching the network. Returns 0 when the file exists
# with the expected digest. A digest mismatch removes the file so that the
# download step fetches a clean copy.
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
# into place so an interrupted download never looks like a cached asset.
download_to() {
    local url="$1"
    local expected_sha256="$2"
    local dest="$3"

    if cache_matches "$dest" "$expected_sha256"; then
        echo "cached $(basename "$dest")"
        return 0
    fi
    echo "downloading $(basename "$dest")"
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

# A deb is an ar archive whose payload member is normally data.tar.zst on
# Ubuntu 24.04. Extract the payload into the shared prefix so all four debs
# land in one filesystem tree.
extract_deb() {
    local deb_path="$1"
    local prefix="$2"
    local staging
    staging="$(mktemp -d "$deps_dir/extract.XXXXXX")"
    trap "rm -rf '$staging'" RETURN

    ( cd "$staging" && ar x "$deb_path" )
    shopt -s nullglob
    local members=("$staging"/data.tar.* "$staging"/data.tar)
    shopt -u nullglob
    if [[ ${#members[@]} -eq 0 ]]; then
        fail "no data.tar member inside $(basename "$deb_path")"
    fi

    local data_member="${members[0]}"
    case "$data_member" in
        *.tar.zst) tar --zstd -xf "$data_member" -C "$prefix" ;;
        *.tar.xz)  tar -xJf "$data_member" -C "$prefix" ;;
        *.tar.gz)  tar -xzf "$data_member" -C "$prefix" ;;
        *.tar)     tar -xf "$data_member" -C "$prefix" ;;
        *)         fail "unsupported deb payload compression: $(basename "$data_member")" ;;
    esac
    rm -rf "$staging"
    trap - RETURN
}

for tool in curl ar tar zstd sha256sum mktemp; do
    require_tool "$tool"
done

[[ -f "$lock_file" ]] || fail "missing lock file $lock_file"
# shellcheck source=/dev/null
source "$lock_file"

metal_version="${TT_METAL_TAG#v}"
metal_prefix="$deps_dir/tt-metal/$metal_version"
# The deb payloads carry their own top-level usr/, so the extraction root is
# the version directory and the install prefix used by CMake is prefix/usr.
metal_dir="$metal_prefix/usr"
marker_file="$metal_prefix/.extract-complete"
marker_content="$TT_METAL_TAG $TT_METAL_DEV_SHA256 $TT_METAL_RUNTIME_SHA256 $TT_METAL_JIT_SHA256 $TT_METAL_EXAMPLES_SHA256"

asset_deb_names=("$TT_METAL_DEV_DEB" "$TT_METAL_RUNTIME_DEB" "$TT_METAL_JIT_DEB" "$TT_METAL_EXAMPLES_DEB")
asset_deb_shas=("$TT_METAL_DEV_SHA256" "$TT_METAL_RUNTIME_SHA256" "$TT_METAL_JIT_SHA256" "$TT_METAL_EXAMPLES_SHA256")
asset_deb_urls=(
    "$TT_METAL_RELEASE_BASE/$TT_METAL_DEV_DEB"
    "$TT_METAL_RELEASE_BASE/$TT_METAL_RUNTIME_DEB"
    "$TT_METAL_RELEASE_BASE/$TT_METAL_JIT_DEB"
    "$TT_METAL_RELEASE_BASE/$TT_METAL_EXAMPLES_DEB"
)

cache_is_complete() {
    local i
    for i in "${!asset_deb_names[@]}"; do
        if ! cache_matches "$download_dir/${asset_deb_names[$i]}" "${asset_deb_shas[$i]}"; then
            return 1
        fi
    done
    if ! cache_matches "$ttsim_dir/$TTSIM_WH_SO" "$TTSIM_WH_SHA256"; then
        return 1
    fi
    if ! cache_matches "$ttsim_dir/soc_descriptor.yaml" "$SOC_DESCRIPTOR_SHA256"; then
        return 1
    fi
    if [[ ! -f "$marker_file" ]]; then
        return 1
    fi
    if [[ "$(cat "$marker_file")" != "$marker_content" ]]; then
        return 1
    fi
    # The marker alone cannot prove that the extracted tree is usable; an
    # interrupted or earlier layout must force a re-extract.
    local config_candidates=(
        "$metal_dir/lib/cmake/tt-metalium/tt-metalium-config.cmake"
        "$metal_dir/lib64/cmake/tt-metalium/tt-metalium-config.cmake"
    )
    local found_config=0
    local candidate
    for candidate in "${config_candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            found_config=1
            break
        fi
    done
    if [[ "$found_config" -eq 0 ]]; then
        return 1
    fi
    if [[ ! -d "$metal_dir/include/tt-metalium" ]]; then
        return 1
    fi
    return 0
}

if cache_is_complete; then
    echo "cache complete: .deps already holds $TT_METAL_TAG and ttsim $TTSIM_TAG, no network needed"
else
    mkdir -p "$download_dir" "$ttsim_dir"
    for i in "${!asset_deb_names[@]}"; do
        download_to "${asset_deb_urls[$i]}" "${asset_deb_shas[$i]}" "$download_dir/${asset_deb_names[$i]}"
    done
    download_to "$TTSIM_BASE/$TTSIM_WH_SO" "$TTSIM_WH_SHA256" "$ttsim_dir/$TTSIM_WH_SO"
    download_to "$SOC_DESCRIPTOR_URL" "$SOC_DESCRIPTOR_SHA256" "$ttsim_dir/soc_descriptor.yaml"

    echo "extracting $TT_METAL_TAG into .deps/tt-metal/$metal_version/usr"
    rm -rf "$metal_prefix"
    mkdir -p "$metal_dir"
    for i in "${!asset_deb_names[@]}"; do
        extract_deb "$download_dir/${asset_deb_names[$i]}" "$metal_prefix"
    done
    printf '%s' "$marker_content" > "$marker_file"
fi

# Discover the library directories so the env hint file does not assume a
# single distribution layout (usr/lib on most builds, a multiarch directory on
# some). The pinned 24.04 debs are expected to use usr/lib.
metal_lib_dirs=()
for candidate in "$metal_dir/lib" "$metal_dir/lib64" "$metal_dir/lib/x86_64-linux-gnu"; do
    if [[ -d "$candidate" ]]; then
        metal_lib_dirs+=("$candidate")
    fi
done
if [[ ${#metal_lib_dirs[@]} -eq 0 ]]; then
    fail "no library directory under $metal_dir; extraction is incomplete"
fi
metal_libs_joined="$(IFS=:; echo "${metal_lib_dirs[*]}")"

env_file="$deps_dir/env.sh"
runtime_home="$metal_dir/libexec/tt-metalium"
cache_dir="$deps_dir/tt-metal-cache"
{
    echo "# Generated by scripts/fetch_tt_metal.sh. Source this file to build and"
    echo "# run against the pinned TT-Metalium release and the ttsim simulator."
    echo "# Every variable below was needed by a real run on a CPU-only machine."
    echo "export TT_METAL_VERSION=\"$TT_METAL_TAG\""
    echo "export TT_METAL_PREFIX=\"$metal_dir\""
    echo "export CMAKE_PREFIX_PATH=\"$metal_dir\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}\""
    echo "export LD_LIBRARY_PATH=\"$metal_libs_joined:$ttsim_dir\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
    # TT_METAL_RUNTIME_ROOT is the name this tag reads (the directory holding
    # tt_metal/ and runtime/). TT_METAL_HOME is set to the same path because
    # ttsim's own documentation and some tt-metal entry points still use it.
    echo "export TT_METAL_RUNTIME_ROOT=\"$runtime_home\""
    echo "export TT_METAL_HOME=\"$runtime_home\""
    # The JIT writes kernels and firmware caches here. The default under
    # ~/.cache is not writable in a sandboxed run, so pin it inside .deps.
    echo "mkdir -p \"$cache_dir\""
    echo "export TT_METAL_CACHE=\"$cache_dir\""
    echo "export TT_METAL_SIMULATOR=\"$ttsim_dir/$TTSIM_WH_SO\""
    echo "export TT_METAL_SLOW_DISPATCH_MODE=1"
} > "$env_file"

echo "fetched $TT_METAL_TAG and ttsim $TTSIM_TAG into .deps/"
echo "hint: source .deps/env.sh before building or running"
