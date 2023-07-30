use std::{sync::Arc, time::Instant};

use cgmath::{Quaternion, Rad, Rotation, Rotation3, Vector3};
use egui_winit_vulkano::{Gui, GuiConfig};
use gilrs::Gilrs;
use helens::Allocators;
use voxels::VoxelCompact;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::SecondaryAutoCommandBuffer,
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{GraphicsPipeline, Pipeline},
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::VulkanoWindowRenderer,
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

const SHOW_OVERLAY_AT_LAUNCH: bool = true;

const DEFAULT_CAMERA_POSITION: Vector3<f32> = Vector3::new(0.01, 0.2, -2.);
const DEFAULT_CAMERA_ORIENTATION: Quaternion<f32> = Quaternion::new(1., 0., 0., 0.);
const DEFAULT_CAMERA_SPEED: f32 = 0.175;
const CAMERA_BOOST_FACTOR: f32 = 3.5;

struct App {
    pub app_start_time: Instant,
    pub descriptor_set: Arc<PersistentDescriptorSet>,
    pub engine: helens::Engine,
    pub game: GameState,
    pub last_draw_time: Option<Instant>,
    pub overlay: Overlay,
    pub voxel_buffer: Subbuffer<[VoxelCompact]>,
    pub window_manager: VulkanoWindows,
}

struct Overlay {
    pub gui: Gui,
    pub is_visible: bool,
    pub last_cursor_movement: Instant,
}

enum RunState {
    NewGame,
    Running(Instant),
}

// TODO: Make level/prtal-depth tracked as well as points.
struct GameState {
    pub camera_position: Vector3<f32>,
    pub camera_quaternion: Quaternion<f32>,
    pub camera_speed: f32,
    pub gamepad: GamepadState,
    pub gilrs: Gilrs,
    pub keyboard: KeyboardState,
    pub points: u32,
    pub run: RunState,
}

#[derive(Clone, Copy, Default)]
struct KeyboardState {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub space: bool,
    pub a: bool,
    pub d: bool,
}

#[derive(Clone, Copy)]
enum SharedAxis {
    Single(f32),
    Split(f32, f32),
}

#[derive(Copy, Clone, Default)]
struct GamepadState {
    pub left_stick: [f32; 2],
    pub yaw: SharedAxis,
    pub south_button: bool,
    pub joystick_mode: bool,
}

fn main() {
    // Initialize the app window, engine, and game state.
    let (event_loop, mut app) = App::new();

    // Run event loop until app exits.
    event_loop.run(move |event, _, control_flow| {
        let window_size = app
            .window_manager
            .get_primary_renderer()
            .unwrap()
            .window_size();
        if window_size.contains(&0.0f32) {
            return;
        }
        match event {
            Event::WindowEvent { event, .. } => {
                // Update Egui integration so the UI works!
                let pass_events_to_game = !app.overlay.gui.update(&event);
                match event {
                    WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                        app.window_manager
                            .get_primary_renderer_mut()
                            .unwrap()
                            .resize();
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
                            app.handle_keyboard_inputs(keycode, state, control_flow);
                        }
                    }
                    WindowEvent::CursorMoved { .. } => {
                        app.overlay.last_cursor_movement = Instant::now();
                    }
                    _ => (),
                }
            }
            Event::RedrawRequested(..) => {
                // Update frame-render timing.
                let delta_time = if let Some(instant) = app.last_draw_time {
                    instant.elapsed()
                } else {
                    app.app_start_time.elapsed()
                }
                .as_secs_f32();
                app.last_draw_time = Some(Instant::now());

                if let Some(window) = app.window_manager.get_primary_window() {
                    const CURSOR_WAIT_TO_HIDE_DURATION: f32 = 2.;
                    window.set_cursor_visible(
                        app.overlay.last_cursor_movement.elapsed().as_secs_f32()
                            < CURSOR_WAIT_TO_HIDE_DURATION,
                    );
                }

                // Update gamepad state.
                app.handle_controller_inputs();

                // Update camera state.
                app.update_camera_state(delta_time);

                // Get secondary command buffer for rendering GUI.
                let renderer = app.window_manager.get_primary_renderer_mut().unwrap();
                let gui_command_buffer =
                    create_updated_overlay(&mut app.overlay, &mut app.game, renderer);

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
                        camera_position: app.game.camera_position.into(),
                        camera_quaternion: app.game.camera_quaternion.into(),
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
                app.window_manager
                    .get_primary_renderer()
                    .unwrap()
                    .window()
                    .request_redraw();
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
                last_cursor_movement: Instant::now(),
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
                game: game_state,
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

    fn handle_keyboard_inputs(
        &mut self,
        keycode: VirtualKeyCode,
        state: ElementState,
        control_flow: &mut ControlFlow,
    ) {
        let mut start_game = false;
        match state {
            ElementState::Pressed => match keycode {
                VirtualKeyCode::Escape => {
                    // If fullscreen then leave fullscreen, else exit the app.
                    let window = self.window_manager.get_primary_window().unwrap();
                    match window.fullscreen() {
                        None => *control_flow = ControlFlow::Exit,
                        Some(_) => {
                            window.set_fullscreen(None);
                        }
                    }
                }
                VirtualKeyCode::F5 => {
                    self.new_random_world();
                    self.game.points = 0;
                    reset_camera(&mut self.game);
                }
                VirtualKeyCode::F11 => {
                    // Toggle fullscreen.
                    let window = self.window_manager.get_primary_window().unwrap();
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
                    self.overlay.is_visible = !self.overlay.is_visible;
                }

                // Camera controls.
                VirtualKeyCode::Up => {
                    self.game.keyboard.up = true;
                    start_game = true;
                }
                VirtualKeyCode::Down => {
                    self.game.keyboard.down = true;
                    start_game = true;
                }
                VirtualKeyCode::Left => {
                    self.game.keyboard.left = true;
                    start_game = true;
                }
                VirtualKeyCode::Right => {
                    self.game.keyboard.right = true;
                    start_game = true;
                }
                VirtualKeyCode::Space => {
                    self.game.keyboard.space = true;
                    start_game = true;
                }
                VirtualKeyCode::A => {
                    self.game.keyboard.a = true;
                    start_game = true;
                }
                VirtualKeyCode::D => {
                    self.game.keyboard.d = true;
                    start_game = true;
                }
                _ => (),
            },
            ElementState::Released => match keycode {
                // Camera controls.
                VirtualKeyCode::Up => {
                    self.game.keyboard.up = false;
                }
                VirtualKeyCode::Down => {
                    self.game.keyboard.down = false;
                }
                VirtualKeyCode::Left => {
                    self.game.keyboard.left = false;
                }
                VirtualKeyCode::Right => {
                    self.game.keyboard.right = false;
                }
                VirtualKeyCode::Space => {
                    self.game.keyboard.space = false;
                }
                VirtualKeyCode::A => {
                    self.game.keyboard.a = false;
                }
                VirtualKeyCode::D => {
                    self.game.keyboard.d = false;
                }
                _ => (),
            },
        }

        // If we processed any events, we're no longer waiting for input.
        self.game.run.ensure_running_if(start_game);
    }

    fn handle_controller_inputs(&mut self) {
        // Default to handling no events.
        let mut processed = false;

        // Process all queued events.
        while let Some(event) = self.game.gilrs.next_event() {
            use gilrs::ev::EventType;
            use SharedAxis::{Single, Split};
            match event.event {
                EventType::AxisChanged(axis, val, _) => match axis {
                    gilrs::Axis::LeftStickX => {
                        self.game.gamepad.left_stick[0] = val;
                        processed = true;
                    }
                    gilrs::Axis::LeftStickY => {
                        self.game.gamepad.left_stick[1] = val;
                        processed = true;
                    }
                    _ => (),
                },
                EventType::ButtonPressed(gilrs::Button::South, _) => {
                    self.game.gamepad.south_button = true;
                    processed = true;
                }
                EventType::ButtonReleased(gilrs::Button::South, _) => {
                    self.game.gamepad.south_button = false;
                    processed = true;
                }
                EventType::ButtonChanged(gilrs::Button::RightTrigger2, val, _)
                    if self.game.gamepad.joystick_mode =>
                {
                    self.game.gamepad.yaw = Single(val + val - 1.);
                    processed = true;
                }
                EventType::ButtonPressed(gilrs::Button::LeftTrigger, _)
                    if !self.game.gamepad.joystick_mode =>
                {
                    self.game.gamepad.yaw = if let Split(_, right) = self.game.gamepad.yaw {
                        Split(1., right)
                    } else {
                        Split(1., 0.)
                    };
                    processed = true;
                }
                EventType::ButtonReleased(gilrs::Button::LeftTrigger, _)
                    if !self.game.gamepad.joystick_mode =>
                {
                    self.game.gamepad.yaw = if let Split(_, right) = self.game.gamepad.yaw {
                        Split(0., right)
                    } else {
                        Split(0., 0.)
                    };
                    processed = true;
                }
                EventType::ButtonPressed(gilrs::Button::RightTrigger, _)
                    if !self.game.gamepad.joystick_mode =>
                {
                    self.game.gamepad.yaw = if let Split(left, _) = self.game.gamepad.yaw {
                        Split(left, 1.)
                    } else {
                        Split(0., 1.)
                    };
                    processed = true;
                }
                EventType::ButtonReleased(gilrs::Button::RightTrigger, _)
                    if !self.game.gamepad.joystick_mode =>
                {
                    self.game.gamepad.yaw = if let Split(left, _) = self.game.gamepad.yaw {
                        Split(left, 0.)
                    } else {
                        Split(0., 0.)
                    };
                    processed = true;
                }
                _ => (),
            }
        }

        // If we processed any events, we're no longer waiting for input.
        self.game.run.ensure_running_if(processed);
    }

    fn update_camera_state(&mut self, delta_time: f32) {
        match self.game.run {
            RunState::NewGame => {
                // Apply demo camera controls until we get input.
                const DEMO_SPEED: f32 = 0.05;
                let demo_rotation = Quaternion::from_angle_y(Rad(delta_time * DEMO_SPEED));
                self.game.camera_position = demo_rotation.rotate_vector(self.game.camera_position);
                self.game.camera_quaternion = demo_rotation * self.game.camera_quaternion;
            }
            RunState::Running(_) => {
                use voxels::Intersection;
                let intersection = voxels::octree_scale_and_collision_of_point(
                    self.game.camera_position,
                    &self.voxel_buffer.read().unwrap(),
                );
                match intersection {
                    Intersection::Empty(scale) => {
                        const SMOOTHING_INCREASE_FACTOR: f32 = -0.12;
                        const SMOOTHING_DECREASE_FACTOR: f32 = -1.4;
                        const SCALING_FACTOR: f32 = 0.7;
                        const ROLL_SPEED: f32 = 2.;
                        const PITCH_SPEED: f32 = 1.25;
                        const YAW_SPEED: f32 = 0.5;

                        self.game.camera_position += if self.game.keyboard.space
                            || self.game.gamepad.south_button
                        {
                            CAMERA_BOOST_FACTOR
                        } else {
                            1.
                        } * self.game.camera_quaternion.rotate_vector(
                            Vector3::new(0., 0., delta_time * self.game.camera_speed),
                        );

                        // Use exponential smoothing to make the camera speed change with scale.
                        let target_speed = DEFAULT_CAMERA_SPEED / scale.powf(SCALING_FACTOR);
                        let smooth = |factor: f32| 1. - (factor * delta_time).exp();
                        self.game.camera_speed += if target_speed > self.game.camera_speed {
                            smooth(SMOOTHING_INCREASE_FACTOR)
                        } else {
                            smooth(SMOOTHING_DECREASE_FACTOR)
                        } * (target_speed - self.game.camera_speed);

                        let roll = (f32::from(self.game.keyboard.left)
                            - f32::from(self.game.keyboard.right)
                            - self.game.gamepad.left_stick[0])
                            .clamp(-1., 1.);
                        let pitch = (f32::from(self.game.keyboard.up)
                            - f32::from(self.game.keyboard.down)
                            + self.game.gamepad.left_stick[1])
                            .clamp(-1., 1.);
                        let yaw = (f32::from(self.game.keyboard.d)
                            - f32::from(self.game.keyboard.a)
                            + match self.game.gamepad.yaw {
                                SharedAxis::Single(value) => value,
                                SharedAxis::Split(left, right) => right - left,
                            })
                        .clamp(-1., 1.);

                        self.game.camera_quaternion = self.game.camera_quaternion
                            * Quaternion::from_angle_z(Rad(delta_time * ROLL_SPEED * roll))
                            * Quaternion::from_angle_x(Rad(delta_time * PITCH_SPEED * pitch))
                            * Quaternion::from_angle_y(Rad(delta_time * YAW_SPEED * yaw));
                    }
                    Intersection::Collision => {
                        reset_camera(&mut self.game);
                        self.game.run = RunState::NewGame;
                        self.game.points = 0;
                    }
                    Intersection::Portal(depth) => {
                        let points_gained = if depth == voxels::MAXIMUM_GOAL_DEPTH {
                            6
                        } else {
                            0
                        } + depth
                            + 1
                            - voxels::MINIMUM_GOAL_DEPTH;
                        self.game.points += points_gained;
                        // TODO: Write each run to a file.
                        println!(
                            "{:.3}s: Portal depth: {depth}, +{points_gained}, Score: {}\n",
                            self.app_start_time.elapsed().as_secs_f32(),
                            self.game.points,
                        );

                        self.new_random_world();

                        reset_camera(&mut self.game);
                    }
                }
            }
        }
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

fn create_updated_overlay(
    overlay: &mut Overlay,
    game: &mut GameState,
    renderer: &mut VulkanoWindowRenderer,
) -> Option<SecondaryAutoCommandBuffer> {
    if overlay.is_visible {
        // Update GUI state.
        overlay.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            egui::Window::new("App GUI").show(&ctx, |ui| {
                ui.heading("Options");
                ui.checkbox(&mut game.gamepad.joystick_mode, "Treat gamepad as joystick");

                // Add bottom UI based on the game state.
                match game.run {
                    RunState::Running(start_time) => {
                        ui.separator();
                        ui.heading(format!("Score: {}", game.points));
                        ui.label(format!("Time: {:.3}s", start_time.elapsed().as_secs_f32()));
                    }
                    RunState::NewGame => (),
                }
            });
        });

        Some(
            overlay
                .gui
                .draw_on_subpass_image(renderer.swapchain_image_size()),
        )
    } else {
        None
    }
}

fn reset_camera(game: &mut GameState) {
    game.camera_position = DEFAULT_CAMERA_POSITION;
    game.camera_quaternion = DEFAULT_CAMERA_ORIENTATION;
    game.camera_speed = DEFAULT_CAMERA_SPEED;
}

impl RunState {
    pub fn ensure_running_if(&mut self, start_game: bool) {
        // If the game has received actions to start and isn't running, transition to the running state.
        match self {
            RunState::NewGame if start_game => *self = RunState::Running(Instant::now()),
            _ => (),
        }
    }
}

impl GameState {
    pub fn default() -> Self {
        GameState {
            camera_position: DEFAULT_CAMERA_POSITION,
            camera_quaternion: DEFAULT_CAMERA_ORIENTATION,
            camera_speed: DEFAULT_CAMERA_SPEED,
            gamepad: GamepadState::default(),
            gilrs: Gilrs::new().unwrap(),
            keyboard: KeyboardState::default(),
            points: 0,
            run: RunState::NewGame,
        }
    }
}

impl Default for SharedAxis {
    fn default() -> Self {
        SharedAxis::Single(0.)
    }
}
