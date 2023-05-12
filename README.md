# WGPU compute demo

This is a demo of a shader to compute inverse square root on gpu using [rust-gpu](https://github.com/EmbarkStudios/rust-gpu) shader and [wgpu](https://github.com/gfx-rs/wgpu) library to run the shader.

## Try it out!

1. Install [Rust](https://rustup.rs/)
2. Build and run the app:
```bash 
$ cargo build
$ cargo run
```
3. You will see the input and output values displayed in your terminal:
```
input = [
    4.0,
    25.0,
    100.0,
]
output = [
    0.5,
    0.2,
    0.1,
]
```