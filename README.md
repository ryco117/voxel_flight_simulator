![fractal_sugar](res/voxel_flight_simulator.ico)
# Voxel Flight Simulator

A simple game developed for the purposes of both showcasing interesting features of sparse-voxel-octree raycasting and to utilize the Vulkan graphics API in a project.
Written in Rust, I decided to use the crate [Vulkano](https://github.com/vulkano-rs/vulkano) to access Vulkan API through a Rust-friendly wrapper. Original .NET version
written in F# is archived [here](https://github.com/ryco117/Voxel-Flight-Simulator-FSharp).

## Controls

The game can be played using either a keyboard, gamepad controller, or H.O.T.A.S. stick.

#### Keyboard
| Key | Action |
|:-:|----------|
| **App-Window** | - |
| F11 | Toggle window fullscreen |
| ESC | If fullscreen, then enter windowed mode. Else, close the application |
| ENTER | *Only Windows release builds:* Toggle the visibility of the output command prompt |
| **Overlay-Window** | - |
| F1 | Toggle showing the Help window |
| o | Toggle showing the Options window |
| **Game** | - |
| F5 | Generate a new random world and reset game |
| **Flight** | - |
| UP | Pitch down |
| DOWN | Pitch up |
| LEFT | Roll left |
| RIGHT | Roll right |
| a | Yaw left |
| d | Yaw right |
| SPACE | Boost |
