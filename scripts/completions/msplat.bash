# bash completion for msplat — source this file or install to
# $(brew --prefix)/etc/bash_completion.d/msplat (see the Homebrew formula).
# CLI11 2.4.2 does not generate completions, so this list is maintained by hand;
# the `--help` output is the source of truth (CI checks the man page against it).
_msplat() {
    local cur prev opts
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="-h --help --version -o --output -s --save-every --resume \
--val --val-image --val-render --eval --test-every \
-n --num-iters -d --downscale-factor --num-downscales --resolution-schedule \
--sh-degree --sh-degree-interval --ssim-weight \
--refine-every --warmup-length --reset-alpha-every \
--densify-grad-thresh --densify-size-thresh --stop-screen-size-at --split-screen-size \
--keep-crs --colmap-image-path --bg-color --debug-bg --preset \
--max-splats --memory-budget \
--progress-every --progress-format --quiet --verbose"

    case "$prev" in
        --preset)          COMPREPLY=( $(compgen -W "draft balanced production" -- "$cur") ); return ;;
        --progress-format) COMPREPLY=( $(compgen -W "plain jsonl" -- "$cur") ); return ;;
        -o|--output|--resume|--val-render|--colmap-image-path)
            COMPREPLY=( $(compgen -f -- "$cur") ); return ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
    else
        COMPREPLY=( $(compgen -d -- "$cur") )   # positional: the dataset directory
    fi
}
complete -F _msplat msplat
