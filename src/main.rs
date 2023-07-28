use std::{sync::Arc, time::Instant};

use cgmath::{Quaternion, Rad, Rotation, Rotation3, Vector3};
use egui_winit_vulkano::{Gui, GuiConfig};
use gilrs::Gilrs;
use helens::Allocators;
use voxels::VoxelCompact;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{GraphicsPipeline, Pipeline},
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Fullscreen,
};

mod helens;
mod voxels;

const TITLE: &str = "voxel_flight_simulator";

const SHOW_OVERLAY_AT_LAUNCH: bool = false;

const DEFAULT_CAMERA_POSITION: Vector3<f32> = Vector3::new(0.01, 0.2, -2.);
const DEFAULT_CAMERA_ORIENTATION: Quaternion<f32> = Quaternion::new(1., 0., 0., 0.);
const DEFAULT_CAMERA_SPEED: f32 = 0.175;
const CAMERA_BOOST_FACTOR: f32 = 3.5;

struct App {
    pub app_start_time: Instant,
    pub descriptor_set: Arc<PersistentDescriptorSet>,
    pub engine: helens::Engine,
    pub game_state: GameState,
    pub last_draw_time: Option<Instant>,
    pub overlay: Overlay,
    pub voxel_buffer: Subbuffer<[VoxelCompact]>,
    pub window_manager: VulkanoWindows,
}

struct Overlay {
    pub gui: Gui,
    pub is_visible: bool,
}

struct GameState {
    pub camera_position: Vector3<f32>,
    pub camera_quaternion: Quaternion<f32>,
    pub camera_speed: f32,
    pub gamepad_state: GamepadState,
    pub gilrs: Gilrs,
    pub key_state: KeyState,
    pub points: u32,
    pub waiting_for_input: bool,
}

#[derive(Clone, Copy, Default)]
struct KeyState {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub space: bool,
}

#[derive(Copy, Clone, Default)]
struct GamepadState {
    pub left_stick: [f32; 2],
    pub yaw: f32,
    pub south_button: bool,
}

fn main() {
    // Initialize the app window, engine, and game state.
    let (event_loop, mut app) = App::new();

    // Run event loop until app exits.
    event_loop.run(move |event, _, control_flow| {
        let window_size =  app.window_manager.get_primary_renderer().unwrap().window_size();
        if window_size.contains(&0.0f32) {
            return;
        }
        match event {
            Event::WindowEvent { event, .. } => {
                // Update Egui integration so the UI works!
                let _pass_events_to_game = !app.overlay.gui.update(&event);
                match event {
                    WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                        app.window_manager.get_primary_renderer_mut().unwrap().resize();
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    } => match keycode {
                        VirtualKeyCode::Escape => {
                            // If fullscreen then leave fullscreen, else exit the app.
                            let window = app.window_manager.get_primary_window().unwrap();
                            match window.fullscreen() {
                                None => *control_flow = ControlFlow::Exit,
                                Some(_) => {
                                    window.set_fullscreen(None);
                                }
                            }
                        }
                        VirtualKeyCode::F5 => {
                            app.new_random_world();
                            app.game_state = GameState::default();
                        }
                        VirtualKeyCode::F11 => {
                            // Toggle fullscreen.
                            let window = app.window_manager.get_primary_window().unwrap();
                            match window.fullscreen() {
                                None => {
                                    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                }
                                Some(_) => {
                                    window.set_fullscreen(None);
                                }
                            }
                        }
                        VirtualKeyCode::O => {
                            // Toggle overlay visibility.
                            app.overlay.is_visible = !app.overlay.is_visible;

                            // Only show the cursor when the overlay is visible.
                            app.window_manager.get_primary_window().unwrap().set_cursor_visible(app.overlay.is_visible);
                        }

                        // Camera controls
                        VirtualKeyCode::Up => {
                            app.game_state.key_state.up = true;
                            app.game_state.waiting_for_input = false;
                        }
                        VirtualKeyCode::Down => {
                            app.game_state.key_state.down = true;
                            app.game_state.waiting_for_input = false;
                        }
                        VirtualKeyCode::Left => {
                            app.game_state.key_state.left = true;
                            app.game_state.waiting_for_input = false;
                        }
                        VirtualKeyCode::Right => {
                            app.game_state.key_state.right = true;
                            app.game_state.waiting_for_input = false;
                        }
                        VirtualKeyCode::Space => {
                            app.game_state.key_state.space = true;
                            app.game_state.waiting_for_input = false;
                        }
                        _ => (),
                    },
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Released,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    } => match keycode {
                        // Camera controls
                        VirtualKeyCode::Up => {
                            app.game_state.key_state.up = false;
                        }
                        VirtualKeyCode::Down => {
                            app.game_state.key_state.down = false;
                        }
                        VirtualKeyCode::Left => {
                            app.game_state.key_state.left = false;
                        }
                        VirtualKeyCode::Right => {
                            app.game_state.key_state.right = false;
                        }
                        VirtualKeyCode::Space => {
                            app.game_state.key_state.space = false;
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
            Event::RedrawRequested(..) => {
                // Update game state.
                let delta_time = if let Some(instant) = app.last_draw_time {
                    instant.elapsed()
                } else {
                    app.app_start_time.elapsed()
                }
                .as_secs_f32();
                app.last_draw_time = Some(Instant::now());

                // Update gamepad state.
                app.game_state.waiting_for_input &= !update_controller_inputs(&mut app.game_state.gilrs, &mut app.game_state.gamepad_state, false);

                if app.game_state.waiting_for_input {
                    // Apply demo camera controls until we get input.
                    const DEMO_SPEED: f32 = 0.05;
                    let demo_rotation = Quaternion::from_angle_y(Rad(delta_time * DEMO_SPEED));
                    app.game_state.camera_position =
                        demo_rotation.rotate_vector(app.game_state.camera_position);
                    app.game_state.camera_quaternion =
                        demo_rotation * app.game_state.camera_quaternion;
                } else {
                    use voxels::Intersection;
                    let intersection = voxels::octree_scale_and_collision_of_point(
                        app.game_state.camera_position,
                        &app.voxel_buffer.read().unwrap(),
                    );
                    match intersection {
                        Intersection::Empty(scale) => {
                            const SMOOTHING_INCREASE_FACTOR: f32 = -0.125;
                            const SMOOTHING_DECREASE_FACTOR: f32 = -1.25;
                            const SCALING_FACTOR: f32 = 0.7;
                            const ROLL_SPEED: f32 = 2.;
                            const PITCH_SPEED: f32 = 1.25;
                            const YAW_SPEED: f32 = 0.5;

                            app.game_state.camera_position +=
                                if app.game_state.key_state.space || app.game_state.gamepad_state.south_button {
                                    CAMERA_BOOST_FACTOR
                                } else {
                                    1.
                                } * app.game_state.camera_quaternion.rotate_vector(Vector3::new(
                                    0.,
                                    0.,
                                    delta_time * app.game_state.camera_speed,
                                ));

                            // Use exponential smoothing to make the camera speed change with scale.
                            let target_speed = DEFAULT_CAMERA_SPEED / scale.powf(SCALING_FACTOR);
                            let smooth = |factor: f32| 1. - (factor * delta_time).exp();
                            app.game_state.camera_speed += if target_speed > app.game_state.camera_speed {
                                smooth(SMOOTHING_INCREASE_FACTOR)
                            } else {
                                smooth(SMOOTHING_DECREASE_FACTOR)
                            } * (target_speed - app.game_state.camera_speed);

                            let roll = (f32::from(app.game_state.key_state.left)
                                - f32::from(app.game_state.key_state.right)
                                - app.game_state.gamepad_state.left_stick[0]).clamp(-1., 1.);
                            let pitch = (f32::from(app.game_state.key_state.up)
                                - f32::from(app.game_state.key_state.down)
                                + app.game_state.gamepad_state.left_stick[1]).clamp(-1., 1.);
                            let yaw = app.game_state.gamepad_state.yaw;

                            app.game_state.camera_quaternion = app.game_state.camera_quaternion
                                * Quaternion::from_angle_z(Rad(delta_time * ROLL_SPEED * roll))
                                * Quaternion::from_angle_x(Rad(delta_time * PITCH_SPEED * pitch))
                                * Quaternion::from_angle_y(Rad(delta_time * YAW_SPEED * yaw));
                        }
                        Intersection::Collision => {
                            app.game_state = GameState {
                                gamepad_state: app.game_state.gamepad_state,
                                key_state: app.game_state.key_state,
                                ..GameState::default()
                            };
                        }
                        Intersection::Portal(depth) => {
                            let points_gained =
                                if depth == voxels::MAXIMUM_GOAL_DEPTH {
                                    6
                                } else {
                                    0
                                } + depth + 1 - voxels::MINIMUM_GOAL_DEPTH;
                            let new_points = app.game_state.points + points_gained;
                            // TODO: Write each run to a file.
                            println!(
                                "{:?}: Portal depth: {depth}, +{points_gained}, Score: {new_points}\n",
                                app.app_start_time.elapsed(),
                            );

                            app.new_random_world();

                            app.game_state = GameState {
                                key_state: app.game_state.key_state,
                                points: new_points,
                                waiting_for_input: false,
                                ..GameState::default()
                            };
                        }
                    };
                }
                let renderer = app.window_manager.get_primary_renderer_mut().unwrap();

                // Get secondary command buffer for rendering GUI.
                let gui_command_buffer = if app.overlay.is_visible {
                    // Update GUI state.
                    app.overlay.gui.immediate_ui(|gui| {
                        let ctx = gui.context();
                        egui::Window::new("App GUI").show(&ctx, |_| {});
                    });

                    Some(
                        app.overlay
                            .gui
                            .draw_on_subpass_image(renderer.swapchain_image_size()),
                    )
                } else {
                    None
                };

                // Render main app with overlay from GUI.
                let push_constants = {
                    fn light_dir(time: f32) -> Vector3<f32> {
                        let delta = time / -20.;
                        Vector3::new(0.9165 * delta.sin(), 0.4, 0.9165 * delta.cos())
                    }
                    let time = app.app_start_time.elapsed().as_secs_f32();
                    helens::ray_march_voxels_fs::Push {
                        aspect_ratio: window_size[0] / window_size[1],
                        time,
                        camera_position: app.game_state.camera_position.into(),
                        camera_quaternion: app.game_state.camera_quaternion.into(),
                        light_dir: light_dir(time).into(),
                    }
                };
                let after_future = app.engine.render_frame(
                    renderer,
                    gui_command_buffer,
                    push_constants,
                    app.descriptor_set.clone(),
                );

                // Present swapchain.
                renderer.present(after_future, true);
            }
            Event::MainEventsCleared => {
                app.window_manager.get_primary_renderer().unwrap().window().request_redraw();
            }
            _ => (),
        }
    });
}

impl App {
    pub fn new() -> (EventLoop<()>, Self) {
        // Winit event loop.
        let event_loop = EventLoop::new();

        // Get Vulkano context.
        let context = VulkanoContext::new(VulkanoConfig::default());

        // Vulkano windows (create one).
        let mut window_manager = VulkanoWindows::default();
        window_manager.create_window(
            &event_loop,
            &context,
            &WindowDescriptor {
                title: TITLE.to_string(),
                present_mode: vulkano::swapchain::PresentMode::Mailbox,
                cursor_visible: SHOW_OVERLAY_AT_LAUNCH,
                ..WindowDescriptor::default()
            },
            |_| {},
        );
        let renderer = window_manager.get_primary_renderer().unwrap();

        // Get the image format that will be used by the swapchain and is acceptable for the window surface.
        let image_format = renderer.swapchain_format();

        // Initialize standalone engine.
        let engine = helens::Engine::new(context.graphics_queue(), image_format);

        let overlay = {
            // Create GUI manager that will render as a subpass of our managed render pass.
            let gui = Gui::new_with_subpass(
                &event_loop,
                renderer.surface(),
                renderer.graphics_queue(),
                engine.gui_pass(),
                GuiConfig {
                    preferred_format: Some(renderer.swapchain_format()),
                    ..GuiConfig::default()
                },
            );
            Overlay {
                gui,
                is_visible: SHOW_OVERLAY_AT_LAUNCH,
            }
        };

        // Initialize storage buffer with random voxel-octree data.
        let (descriptor_set, voxel_buffer) =
            create_random_world(engine.allocators(), engine.pipeline());

        // Create an initial game state.
        let game_state = GameState::default();

        (
            event_loop,
            App {
                app_start_time: Instant::now(),
                descriptor_set,
                engine,
                game_state,
                last_draw_time: None,
                overlay,
                voxel_buffer,
                window_manager,
            },
        )
    }

    pub fn new_random_world(&mut self) {
        let (descriptor_set, voxel_buffer) =
            create_random_world(self.engine.allocators(), self.engine.pipeline());
        self.descriptor_set = descriptor_set;
        self.voxel_buffer = voxel_buffer;
    }
}

fn create_random_world(
    allocators: &Allocators,
    pipeline: &Arc<GraphicsPipeline>,
) -> (Arc<PersistentDescriptorSet>, Subbuffer<[VoxelCompact]>) {
    // Generate a random voxel-octree.
    let (voxel_octree, stats) = voxels::generate_recursive_voxel_octree(256, 10);
    println!(
        "Voxel Count: {:?}, Portal Count: {:?}",
        stats.voxel_count, stats.goal_count
    );

    // Upload the voxel-octree to the GPU.
    let storage_usage = BufferCreateInfo {
        usage: BufferUsage::STORAGE_BUFFER,
        ..Default::default()
    };
    let upload_usage = AllocationCreateInfo {
        usage: MemoryUsage::Upload,
        ..Default::default()
    };
    let buffer = Buffer::from_iter(
        &allocators.memory,
        storage_usage,
        upload_usage,
        voxel_octree.into_iter(),
    )
    .expect("Failed to create voxel buffer");

    (
        PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, buffer.clone())],
        )
        .expect("Failed to create voxel buffer descriptor set"),
        buffer,
    )
}

fn update_controller_inputs(gilrs: &mut Gilrs, gamepad_state: &mut GamepadState, joystick_mode: bool) -> bool {
    // Default to handling no events.
    let mut processed = false;

    // Process all queued events.
    while let Some(event) = gilrs.next_event() {
        gilrs.gamepad(event.id).name();

        use gilrs::ev::EventType;
        match event.event {
            EventType::AxisChanged(axis, val, _) => match axis {
                gilrs::Axis::LeftStickX => {
                    gamepad_state.left_stick[0] = val;
                    processed = true;
                }
                gilrs::Axis::LeftStickY => {
                    gamepad_state.left_stick[1] = val;
                    processed = true;
                }
                _ => (),
            }
            EventType::ButtonPressed(gilrs::Button::South, _) => {
                gamepad_state.south_button = true;
                processed = true;
            }
            EventType::ButtonReleased(gilrs::Button::South, _) => {
                gamepad_state.south_button = false;
                processed = true;
            }
            EventType::ButtonChanged(gilrs::Button::RightTrigger2, val, _) if joystick_mode => {
                gamepad_state.yaw = val + val - 1.;
                processed = true;
            }
            EventType::ButtonChanged(gilrs::Button::RightTrigger, val, _) if !joystick_mode => {
                gamepad_state.yaw = val;
                processed = true;
            }
            EventType::ButtonChanged(gilrs::Button::LeftTrigger, val, _) if !joystick_mode => {
                gamepad_state.yaw = -val;
                processed = true;
            }
            _ => (),
        }
    }
    processed
}

impl GameState {
    pub fn default() -> Self {
        GameState {
            camera_position: DEFAULT_CAMERA_POSITION,
            camera_quaternion: DEFAULT_CAMERA_ORIENTATION,
            camera_speed: DEFAULT_CAMERA_SPEED,
            gamepad_state: GamepadState::default(),
            gilrs: Gilrs::new().unwrap(),
            key_state: KeyState::default(),
            points: 0,
            waiting_for_input: true,
        }
    }
}
