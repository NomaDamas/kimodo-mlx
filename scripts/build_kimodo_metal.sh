#!/bin/sh
set -eu

upstream_dir=${1:-.cache/kimodo.cpp}
build_dir=${2:-.cache/kimodo-build-metal}

if [ ! -f "$upstream_dir/CMakeLists.txt" ]; then
  printf '%s\n' "missing upstream kimodo.cpp checkout: $upstream_dir" >&2
  exit 2
fi

git -C "$upstream_dir" submodule update --init --recursive
if ! git -C "$upstream_dir" apply --check "$(pwd)/patches/0001-kimodo-ggml-metal.patch"; then
  if ! git -C "$upstream_dir" diff --quiet -- CMakeLists.txt src/ggml_weights.cpp; then
    printf '%s\n' "upstream checkout already contains local changes; use a clean checkout" >&2
    exit 2
  fi
fi
git -C "$upstream_dir" apply "$(pwd)/patches/0001-kimodo-ggml-metal.patch"
cmake -S "$upstream_dir" -B "$build_dir" \
  -DKIMODO_ENABLE_VULKAN=OFF \
  -DKIMODO_ENABLE_METAL=ON \
  -DKIMODO_BUILD_TESTS=OFF
cmake --build "$build_dir" --parallel
