# Examples

This page will suggest a handful of example commands you can use to launch
various Guest operating systems using OpenVMM.

> These examples all use `cargo run --`, with the assumption that you are a
> developer building your own copy of OpenVMM locally!
>
> To run these examples using a pre-compiled copy of OpenVMM, swap `cargo run
> --` with `/path/to/openvmm`.

If you run into any issues, please refer to [Troubleshooting](../troubleshooting.md).

## Preface: Quitting OpenVMM

By default, OpenVMM will connect the guests's COM1 serial port to the current
terminal session, forwarding all keystrokes directly to the VM.

As such, a simple `ctrl-c` does not suffice to quit OpenVMM!

Instead, you can type `crtl-q` to enter OpenVMM's
[interactive console](./interactive_console.md), and enter `q` to quit.

## Sample Linux Kernel, via direct-boot

This example will launch Linux via direct boot (i.e: without going through UEFI
or BIOS), and appends `single` to the kernel command line.

The Linux guest's console will be hooked up to COM1, and is relayed to the host
terminal by default.

To launch Linux with an interactive console into the shell within initrd, simply
run:

```shell
cargo run
```

This works by setting the default `[env]` vars in `.cargo/config.toml` to
configure OpenVMM to use a set of pre-compiled test kernel + initrd images,
which are downloaded as part of the `cargo xflowey restore-packages` command.
Note that this behavior only happens when run via `cargo run` (as `cargo` is the
tool which ensures the required env-vars are set).

The source for the sample kernel + initrd can be found on the
[microsoft/openvmm-deps](https://github.com/microsoft/openvmm-deps) repo.

The kernel and initrd can be controlled via options:

* `--kernel <PATH>`: The kernel image. Must be an uncompressed kernel (vmlinux, not bzImage).
* `--initrd <PATH>`: The initial ramdisk image.
* `-c <STRING>` or `--cmdline <STRING>`: Extra kernel command line options, such as `root=/dev/sda`.

## Windows, via UEFI

This example will launch a modern copy of Windows via UEFI, using the `mu_msvm`
firmware package.

A copy of the `mu_msvm` UEFI firmware is automatically downloaded via `cargo
xflowey restore-packages`.

```shell
cargo run -- --uefi --disk memdiff:path/to/windows.vhdx --gfx
```

For more info on `--gfx`, and how to actually interact with the VM using a
mouse/keyboard/video, see the [Graphical Console](./graphical_console.md)
docs.

The file `windows.vhdx` can be any format of VHD(X).

Note that OpenVMM does not currently support using dynamic VHD/VHDX files on
Linux hosts. Unless you have a fixed VHD1 image, you will need to convert the
image to raw format, using the following command:

```shell
qemu-img convert -f vhdx -O raw windows.vhdx windows.img
```

Also, note the use of `memdiff`, which creates a memory-backed "differencing
disk" shim between the VMM and the backing disk image, which ensures that any
writes the VM makes to the VHD are not persisted between runs. This is very
useful when iterating on OpenVMM code, since booting the VM becomes repeatable
and you don't have to worry about shutting down properly. Use `file` instead for
normal persistent storage.

## DOS, via PCAT BIOS

While DOS in particular is not a scenario that the OpenVMM has heavily invested
in, the fact DOS is able to boot in OpenVMM serves as a testament to OpenVMM's
solid support of legacy x86 devices and infrastructure.

The following command will boot a copy of DOS from a virtual floppy disk, using
the Hyper-V PCAT BIOS.

Booting via PCAT is not just for DOS though! Many older operating systems,
including older copies of Windows / Linux, require booting via BIOS.

```bash
cargo run -- --pcat --gfx --floppy memdiff:/path/to/msdos.vfd --pcat-boot-order=floppy,optical,hdd
```

> Due to licensing restrictions, the OpenVMM project is not able to redistribute
> a copy of the Hyper-V PCAT BIOS.
>
> Windows 11 ships copies of the PCAT BIOS firmware in-box under `System32` as
> either `vmfirmwarepcat.dll` or `vmfirmware.dll`. OpenVMM will automatically
> scan for these files, and use them if present.
>
> In the future, it would be interesting to support alternative BIOS
> implementations, such as SeaBIOS, but this is not currently supported.
