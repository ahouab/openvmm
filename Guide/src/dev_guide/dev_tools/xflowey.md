# cargo xflowey

To implement various developer workflows (both locally, as well as in CI), the
OpenVMM project relies on `flowey`: a custom, in-house Rust library/framework
for writing maintainable, cross-platform automation.

`cargo xflowey` is a cargo alias that makes it easy for developers to run
`flowey`-based pipelines locally.

Some particularly notable pipelines:

- `cargo xflowey build-igvm` - primarily dev-tool used to build OpenHCL IGVM files locally
- `cargo xflowey ci checkin-gates` - runs the entire PR checkin suite locally
- `cargo xflowey restore-packages` - restores external packages needed to compile and run OpenVMM / OpenHCL

### `xflowey` vs `xtask`

In a nutshell:

- `cargo xtask`: implements novel, standalone tools/utilities
- `cargo xflowey`: orchestrates invoking a sequence of tools/utilities, without
  doing any non-trivial data processing itself
