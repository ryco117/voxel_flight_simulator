// Ensure Windows builds are not console apps
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
                        window_manager.get_primary_renderer_mut().unwrap().resize();
                    }
                    WindowEvent::CloseRequested => {
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
