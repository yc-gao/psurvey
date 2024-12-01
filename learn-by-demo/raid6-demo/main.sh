#!/usr/bin/env bash
set -ex

info() {
    echo "[info]" "$@"
}

err() {
    echo "[error]" "$@" 2>&1
}

die() {
    local code="$1"
    shift

    err "$@"
    exit "${code}"
}

__func_defered=()
do_defer() {
    for ((i=${#__func_defered[@]}-1;i>=0;i--)); do
        if ! eval "${__func_defered[i]}"; then
            die 1 "eval cmd failed, cmd: \"${__func_defered[i]}\""
        fi
    done
}
trap do_defer EXIT
defer() {
    __func_defered+=("$*")
}

# opt_volumes=(
#     "/dev/loop0"
#     "/dev/loop1"
#     "/dev/loop2"
#     "/dev/loop3"
#     "/dev/loop4"
# )
# opt_cvolume="/dev/loop5"
# opt_vg="demo"
# opt_lv="demo_lv"
opt_volumes=()
opt_cvolume=
opt_vg=
opt_lv=

do_raid6() {
    pvcreate "${opt_volumes[@]}"
    vgcreate "${opt_vg}" "${opt_volumes[@]}"
    lvcreate --type raid6 -i "$((${#opt_volumes[@]}-2))" -l 100%FREE -n "${opt_lv}" "${opt_vg}"

    if [ -n "${opt_cvolume}" ]; then
        pvcreate "${opt_cvolume}"
        vgextend "${opt_vg}" "${opt_cvolume}"
        lvcreate --type cache-pool -l 100%FREE -n cache_pool "${opt_vg}" "${opt_cvolume}"
        lvconvert --type cache --cachepool "${opt_vg}/cache_pool" "${opt_vg}/${opt_lv}"
    fi
}

main() {
    local action=
    while (($#)); do
        case "$1" in
            --volume)
                opt_volumes+=("$2")
                shift 2
                ;;
            --cvolume)
                opt_cvolume="$2"
                shift 2
                ;;
            --volume-group)
                opt_vg="$2"
                shift 2
                ;;
            --logic-volume)
                opt_lv="$2"
                shift 2
                ;;
            --raid6)
                action="do_raid6"
                shift
                ;;
            *)
                break
                ;;
        esac
    done

    if [ -z "${action}" ]; then
        die 1 "undefined action"
    fi
    "${action}"
}

main "$@"
