/*
    voxel_flight_simulator - A simple game where you fly around randomly generated, recursive, voxel worlds.
    Copyright (C) 2023 Ryan Andersen

    voxel_flight_simulator is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    voxel_flight_simulator is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with voxel_flight_simulator. If not, see <https://www.gnu.org/licenses/>.
*/

// Ensure Windows release builds are not console apps.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use voxel_flight_simulator::App;
use winit::{
    event::{Event, KeyboardInput, WindowEvent},
    event_loop::ControlFlow,
};

mod game;
mod helens;
mod voxel_flight_simulator;
mod voxels;

fn main() {
    // Initialize the app window, engine, and game state.
    let (mut app, event_loop, mut gui, mut window_manager) = App::new();

    // Load icon from file resources.
    let icon = {
        // The data below is read at compile time from the file.
        let icon_bytes = std::include_bytes!("../res/voxel_flight_simulator.ico");

        // Use the stored icon data to parse for an image that winit can understand.
        let ico_reader = std::io::Cursor::<&[u8]>::new(icon_bytes);
        let ico_list = ico::IconDir::read(ico_reader).unwrap();
        let ico = ico_list
            .entries()
            .get(0)
            .expect("Icon doesn't have any layers");
        let image = ico.decode().unwrap();

        // Convert the image into a winit icon.
        match winit::window::Icon::from_rgba(
            image.rgba_data().to_vec(),
            image.width(),
            image.height(),
        ) {
            Ok(icon) => Some(icon),
            Err(e) => {
                println!("Failed to parse icon: {e:?}");
                None
            }
        }
    };

    // Apply icon to the window.
    window_manager
        .get_primary_window()
        .unwrap()
        .set_window_icon(icon);

    // Run event loop until app exits.
    event_loop.run(move |event, _, control_flow| {
        let window_size = window_manager.get_primary_renderer().unwrap().window_size();
        if window_size.contains(&0.0f32) {
            return;
        }
        match event {
            Event::WindowEvent { event, .. } => {
                // Update the egui with our events so the UI can work!
                let pass_events_to_game = !gui.update(&event);
                match event {
                    WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                        // Make app aware of the new window size.
                        app.resize(&mut window_manager);
                    }
                    WindowEvent::CloseRequested => {
                        // The window has been instructed to close.
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    } => {
                        if pass_events_to_game {
                            app.handle_keyboard_inputs(
                                keycode,
                                state,
                                &mut window_manager,
                                control_flow,
                            );
                        }
                    }
                    WindowEvent::CursorMoved { .. } => {
                        app.overlay.last_cursor_movement = std::time::Instant::now();
                    }
                    _ => (),
                }
            }

            // Update the app state and render a frame.
            Event::MainEventsCleared => app.tock_frame(&mut gui, &mut window_manager, window_size),
            _ => (),
        }
    });
}
