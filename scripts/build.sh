#!/bin/bash

cd "$(dirname "$0")"
cd ..

export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning,ignore::FutureWarning"

skip_vision=false
colcon_args=()

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
	echo "Building without vision package"
	colcon_args+=(--packages-skip vision)
fi

shopt -s nullglob
for cache in build/*/CMakeCache.txt; do
	package_dir="$(dirname "$cache")"
	package_name="$(basename "$package_dir")"
	source_dir="$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$cache" | cut -d= -f2-)"

	if [[ -n "$source_dir" && ! -d "$source_dir" ]]; then
		echo "Removing stale cache for $package_name (missing: $source_dir)"
		rm -rf "build/$package_name" "install/$package_name"
	fi
done

colcon build --symlink-install --parallel-workers "$(nproc)" "${colcon_args[@]}"
build_exit_code=$?

if [[ $build_exit_code -eq 0 ]]; then
	espeak "build complete" >/dev/null 2>&1 || echo "Build complete"
else
	echo "Build failed"
fi

exit $build_exit_code
