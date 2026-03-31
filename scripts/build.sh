#!/bin/bash

# 1. Ensure we execute from the root of the K1DIY workspace
cd "$(dirname "$0")/.."

# 2. Suppress annoying Python warnings from colcon
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning,ignore::FutureWarning"

colcon_args=()
skip_vision=false

# 3. Parse arguments
for arg in "$@"; do
    case "$arg" in
        --without-vision|--no-vision)
            skip_vision=true
            ;;
        *)
            colcon_args+=("$arg")
            ;;
    esac
done

if [[ "$skip_vision" == true ]]; then
    echo "--> Skipping the 'vision' package (Building brain & interfaces only)"
    colcon_args+=(--packages-skip vision)
fi

# 4. Clean up stale CMake caches 
# (CRITICAL: This prevents errors caused by deleting the old NaovaK1 packages)
echo "--> Checking for stale CMake caches..."
shopt -s nullglob
for cache in build/*/CMakeCache.txt; do
    package_dir="$(dirname "$cache")"
    package_name="$(basename "$package_dir")"
    source_dir="$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$cache" | cut -d= -f2-)"

    if [[ -n "$source_dir" && ! -d "$source_dir" ]]; then
        echo "    [Clean] Removing stale cache for $package_name"
        rm -rf "build/$package_name" "install/$package_name"
    fi
done

# 5. Build the workspace
echo "--> Compiling K1DIY workspace..."
colcon build --symlink-install --parallel-workers "$(nproc)" --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${colcon_args[@]}"
build_exit_code=$?

# 6. User Feedback
if [[ $build_exit_code -eq 0 ]]; then
    echo "Build Completed Successfully!"
    echo "   To source the workspace, run:"
    echo "   source install/setup.bash"
else
	echo "Build failed"
fi

exit $build_exit_code