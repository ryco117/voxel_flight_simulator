use std::{sync::Arc, time::Instant};

use cgmath::{Quaternion, Rad, Rotation, Rotation3, Vector3};
use egui_winit_vulkano::{Gui, GuiConfig};
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

const DEFAULT_CAMERA_SPEED: f32 = 0.25;

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
}

fn main() {
    // Initialize the app window, engine, and game state.
    let (event_loop, mut app) = App::new();

    // Run event loop until app exits.
    event_loop.run(move |event, _, control_flow| {
        let renderer = app.window_manager.get_primary_renderer_mut().unwrap();
        let window_size = renderer.window_size();
        if window_size.contains(&0.0f32) {
            return;
        }
        match event {
            Event::WindowEvent { event, .. } => {
                // Update Egui integration so the UI works!
                let _pass_events_to_game = !app.overlay.gui.update(&event);
                match event {
                    WindowEvent::Resized(_) => {
                        renderer.resize();
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        renderer.resize();
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
                            app.game_state.camera_position +=
                                app.game_state.camera_quaternion.rotate_vector(Vector3::new(
                                    0.,
                                    0.,
                                    delta_time * app.game_state.camera_speed,
                                ));

                            // Use exponential smoothing to make the camera speed change with scale.
                            const SMOOTHING_FACTOR: f32 = -0.67;
                            const SCALING_FACTOR: f32 = 0.78;
                            let smooth = 1. - (SMOOTHING_FACTOR * delta_time).exp();
                            app.game_state.camera_speed += smooth
                                * (DEFAULT_CAMERA_SPEED / scale.powf(SCALING_FACTOR)
                                    - app.game_state.camera_speed);

                            let pitch = f32::from(app.game_state.key_state.up)
                                - f32::from(app.game_state.key_state.down);
                            const PITCH_SPEED: f32 = 1.2;
                            let roll = f32::from(app.game_state.key_state.left)
                                - f32::from(app.game_state.key_state.right);
                            const ROLL_SPEED: f32 = 2.;
                            app.game_state.camera_quaternion = app.game_state.camera_quaternion
                                * Quaternion::from_angle_z(Rad(delta_time * ROLL_SPEED * roll))
                                * Quaternion::from_angle_x(Rad(delta_time * PITCH_SPEED * pitch));
                        }
                        Intersection::Collision => {
                            app.game_state = GameState::default();
                        }
                        Intersection::Portal(depth) => {
                            let points_gained = depth + 1 - voxels::MINIMUM_GOAL_DEPTH;
                            let new_points = app.game_state.points + points_gained;
                            println!("Portal depth: {depth}, +{points_gained}, Score: {new_points}\n");

                            let (descriptor_set, voxel_buffer) =
                                create_random_world(app.engine.allocators(), app.engine.pipeline());
                            app.descriptor_set = descriptor_set;
                            app.voxel_buffer = voxel_buffer;

                            app.game_state = GameState {
                                key_state: app.game_state.key_state,
                                points: new_points,
                                waiting_for_input: false,
                                ..GameState::default()
                            };
                        }
                    };
                }

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
                        let delta = time / -15.;
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
                renderer.window().request_redraw();
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
                ..WindowDescriptor::default()
            },
            |ci| {
                println!("Swapchain creation info {:?}", ci);
            },
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
                is_visible: false,
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
    let voxel_octree = voxels::generate_recursive_voxel_octree(256);
    println!("Octree Unique-Voxel Count: {:?}", voxel_octree.len());

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

impl GameState {
    pub fn default() -> Self {
        GameState {
            camera_position: Vector3::new(0.01, 0.2, -2.),
            camera_quaternion: Quaternion::new(1., 0., 0., 0.),
            camera_speed: DEFAULT_CAMERA_SPEED,
            key_state: KeyState::default(),
            points: 0,
            waiting_for_input: true,
        }
    }
}
