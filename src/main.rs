// Ensure Windows builds are not console apps
#![windows_subsystem = "windows"]

use std::{fs::File, path, sync::Arc, time::Instant};

mod game;
mod helens;
mod voxels;

use cgmath::{Quaternion, Rad, Rotation, Rotation3, Vector3};
use egui::Context;
use egui_winit_vulkano::{Gui, GuiConfig};
use game::{HoldOrToggle, Run, SharedAxis};
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

#[cfg(target_os = "windows")]
use companion_console::ConsoleState;

const TITLE: &str = "voxel_flight_simulator";

const SHOW_OVERLAY_AT_LAUNCH: bool = true;

const CAMERA_BOOST_FACTOR: f32 = 3.5;

struct LogFile(File);

struct App {
    pub app_start_time: Instant,
    pub descriptor_set: Arc<PersistentDescriptorSet>,
    pub engine: helens::Engine,
    pub game: game::State,
    pub last_draw_time: Option<Instant>,
    pub log_file: LogFile,
    pub overlay: Overlay,
    pub random: voxels::RandomOctreeHelper,
    pub voxel_buffer: Subbuffer<[VoxelCompact]>,

    #[cfg(target_os = "windows")]
    pub console: ConsoleState,
}

struct Overlay {
    pub is_visible: bool,
    pub last_cursor_movement: Instant,
    pub seed_string: String,
}

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
                // Update Egui integration so the UI works!
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

                if let Some(window) = window_manager.get_primary_window() {
                    const CURSOR_WAIT_TO_HIDE_DURATION: f32 = 2.;
                    window.set_cursor_visible(
                        app.overlay.last_cursor_movement.elapsed().as_secs_f32()
                            < CURSOR_WAIT_TO_HIDE_DURATION,
                    );
                }

                // Update gamepad state.
                app.handle_controller_inputs();

                // Update camera state.
                app.update_player_state(delta_time);

                // Get secondary command buffer for rendering GUI.
                let renderer = window_manager.get_primary_renderer_mut().unwrap();
                let gui_command_buffer = app.create_updated_overlay(&mut gui, renderer);

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
                window_manager
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
    pub fn new() -> (Self, EventLoop<()>, Gui, VulkanoWindows) {
        // Create a console window for debugging.
        #[cfg(target_os = "windows")]
        let console = ConsoleState::new(false).expect("Could not allocate a console window.");

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

        // Create the RNG to be used for voxel-world generation.
        let mut random = voxels::RandomOctreeHelper::default();

        // Create GUI manager that will render as a subpass of our render pass.
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

        // Create manager for the GUI overlay and state.
        let overlay = {
            Overlay {
                is_visible: SHOW_OVERLAY_AT_LAUNCH,
                last_cursor_movement: Instant::now(),
                seed_string: random.get_seed().to_string(),
            }
        };

        // Create a log file for app convenience.
        let mut log_file = LogFile::default();

        // Initialize storage buffer with random voxel-octree data.
        let (descriptor_set, voxel_buffer) = create_random_world(
            engine.allocators(),
            &mut random,
            engine.pipeline(),
            &mut log_file,
        );

        // Create an initial game state.
        let game_state = game::State::default();

        (
            App {
                app_start_time: Instant::now(),
                descriptor_set,
                engine,
                game: game_state,
                last_draw_time: None,
                log_file,
                overlay,
                random,
                voxel_buffer,

                #[cfg(target_os = "windows")]
                console,
            },
            event_loop,
            gui,
            window_manager,
        )
    }

    pub fn new_random_world(&mut self, world_seed: u64) {
        // Ensure that creating a new world always requires updating to a new seed.
        self.random.set_seed(world_seed);

        // Update the overlay with the new seed.
        self.overlay.seed_string = world_seed.to_string();

        // Create GPU buffer and descriptor set for new world.
        let (descriptor_set, voxel_buffer) = create_random_world(
            self.engine.allocators(),
            &mut self.random,
            self.engine.pipeline(),
            &mut self.log_file,
        );
        self.descriptor_set = descriptor_set;
        self.voxel_buffer = voxel_buffer;

        // Reset the camera since we never enter a new world at a non-start orientation.
        self.game.reset_camera();
    }

    fn handle_keyboard_inputs(
        &mut self,
        keycode: VirtualKeyCode,
        state: ElementState,
        window_manager: &mut VulkanoWindows,
        control_flow: &mut ControlFlow,
    ) {
        let mut game_starting_event = false;
        match state {
            ElementState::Pressed => match keycode {
                VirtualKeyCode::Escape => {
                    // If fullscreen then leave fullscreen, else exit the app.
                    let window = window_manager.get_primary_window().unwrap();
                    match window.fullscreen() {
                        None => *control_flow = ControlFlow::Exit,
                        Some(_) => {
                            window.set_fullscreen(None);
                        }
                    }
                }
                VirtualKeyCode::F5 => {
                    use rand::Rng;
                    self.game.run = Run::default();
                    self.new_random_world(rand::thread_rng().gen());
                }
                VirtualKeyCode::F11 => {
                    // Toggle fullscreen.
                    let window = window_manager.get_primary_window().unwrap();
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
                    game_starting_event = true;
                }
                VirtualKeyCode::Down => {
                    self.game.keyboard.down = true;
                    game_starting_event = true;
                }
                VirtualKeyCode::Left => {
                    self.game.keyboard.left = true;
                    game_starting_event = true;
                }
                VirtualKeyCode::Right => {
                    self.game.keyboard.right = true;
                    game_starting_event = true;
                }
                VirtualKeyCode::Space => {
                    self.game.keyboard.space = true;
                    match &mut self.game.options.camera_boost {
                        HoldOrToggle::Toggle(t) => *t = !*t,
                        HoldOrToggle::Hold => (),
                    }
                    game_starting_event = true;
                }
                VirtualKeyCode::A => {
                    self.game.keyboard.a = true;
                    game_starting_event = true;
                }
                VirtualKeyCode::D => {
                    self.game.keyboard.d = true;
                    game_starting_event = true;
                }

                // Toggle Windows console visibility.
                #[cfg(target_os = "windows")]
                VirtualKeyCode::Return => {
                    if self.console.is_visible() {
                        self.console.hide();
                    } else {
                        self.console.show();
                    }
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

        // If we processed any game-starting events, we're no longer waiting for input.
        self.game.run.ensure_running_if(game_starting_event);
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
                    match &mut self.game.options.camera_boost {
                        HoldOrToggle::Toggle(t) => *t = !*t,
                        HoldOrToggle::Hold => (),
                    }
                    processed = true;
                }
                EventType::ButtonReleased(gilrs::Button::South, _) => {
                    self.game.gamepad.south_button = false;
                    processed = true;
                }
                EventType::ButtonChanged(gilrs::Button::RightTrigger2, val, _)
                    if self.game.options.hotas_mode =>
                {
                    self.game.gamepad.yaw = Single(val + val - 1.);
                    processed = true;
                }
                EventType::ButtonPressed(gilrs::Button::LeftTrigger, _)
                    if !self.game.options.hotas_mode =>
                {
                    self.game.gamepad.yaw = if let Split(_, right) = self.game.gamepad.yaw {
                        Split(1., right)
                    } else {
                        Split(1., 0.)
                    };
                    processed = true;
                }
                EventType::ButtonReleased(gilrs::Button::LeftTrigger, _)
                    if !self.game.options.hotas_mode =>
                {
                    self.game.gamepad.yaw = if let Split(_, right) = self.game.gamepad.yaw {
                        Split(0., right)
                    } else {
                        Split(0., 0.)
                    };
                    processed = true;
                }
                EventType::ButtonPressed(gilrs::Button::RightTrigger, _)
                    if !self.game.options.hotas_mode =>
                {
                    self.game.gamepad.yaw = if let Split(left, _) = self.game.gamepad.yaw {
                        Split(left, 1.)
                    } else {
                        Split(0., 1.)
                    };
                    processed = true;
                }
                EventType::ButtonReleased(gilrs::Button::RightTrigger, _)
                    if !self.game.options.hotas_mode =>
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

    // Update state for the player/camera and their run.
    fn update_player_state(&mut self, delta_time: f32) {
        match self.game.run.start {
            None => {
                // Apply demo camera controls until we get input.
                const DEMO_SPEED: f32 = 0.05;
                let demo_rotation = Quaternion::from_angle_y(Rad(delta_time * DEMO_SPEED));
                self.game.camera_position = demo_rotation.rotate_vector(self.game.camera_position);
                self.game.camera_quaternion = demo_rotation * self.game.camera_quaternion;
            }
            Some(_) => {
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
                            || self.game.options.camera_boost.into()
                        {
                            CAMERA_BOOST_FACTOR
                        } else {
                            1.
                        } * self.game.camera_quaternion.rotate_vector(
                            Vector3::new(0., 0., delta_time * self.game.camera_speed),
                        );

                        // Use exponential smoothing to make the camera speed change with scale.
                        let target_speed = game::DEFAULT_CAMERA_SPEED / scale.powf(SCALING_FACTOR);
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
                        self.game.reset_camera();
                        self.game.run = Run::default();
                    }
                    Intersection::Portal { depth, index } => {
                        let points_gained =
                            u32::from(depth == voxels::MAXIMUM_GOAL_DEPTH) + depth + 1
                                - voxels::MINIMUM_GOAL_DEPTH;
                        self.game.run.points += points_gained;
                        self.game.run.level += 1;

                        // Log the state of the run after taking the portal and gaining points.
                        self.log_file.log(format!("{:.3}s: Portal depth: {depth}, +{points_gained}, Score: {}, Levels: {}\n",
                            self.app_start_time.elapsed().as_secs_f32(),
                            self.game.run.points,
                            self.game.run.level,
                        ).as_str());

                        // Use the portal taken to seed the RNG for the next world.
                        self.new_random_world(self.random.get_seed() + u64::from(index));
                    }
                }
            }
        }
    }

    // Update the internal GUI state and return an optional command buffer to draw the overlay.
    fn create_updated_overlay(
        &mut self,
        gui: &mut Gui,
        renderer: &mut VulkanoWindowRenderer,
    ) -> Option<SecondaryAutoCommandBuffer> {
        // Options window helper.
        fn options_window(app: &mut App, ctx: &Context) {
            // Create an Egui window that starts closed.
            egui::Window::new("Options")
                .default_open(false)
                .show(ctx, |ui| {
                    // Create a toggle for reading inputs as a gamepad or H.O.T.A.S.
                    ui.checkbox(
                        &mut app.game.options.hotas_mode,
                        "Treat gamepad as H.O.T.A.S. stick",
                    );

                    // Create an option to either hold or toggle for boost.
                    let mut b = app.game.options.camera_boost != HoldOrToggle::Hold;
                    if ui.checkbox(&mut b, "Toggle boost").changed() {
                        app.game.options.camera_boost = match app.game.options.camera_boost {
                            HoldOrToggle::Hold => HoldOrToggle::Toggle(false),
                            HoldOrToggle::Toggle(_) => HoldOrToggle::Hold,
                        };
                    }

                    // Create an option to toggle the overlay's visibility.
                    ui.checkbox(&mut app.overlay.is_visible, "Show overlay");

                    // Allow user to view, edit, and set the world seed.
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut app.overlay.seed_string);
                        if ui.button("Set seed").clicked() {
                            if let Ok(seed) = app.overlay.seed_string.parse::<u64>() {
                                app.game.run = Run::default();
                                app.new_random_world(seed);
                            }
                        }
                    });
                });
        }

        // Help window helper.
        fn help_window(ctx: &Context) {
            // Helper enum for creating a grid of controls.
            // Each entry is a row in the grid.
            enum HelpWindowEntry {
                Title(&'static str),
                Item(&'static str, &'static str),
                Empty(),
            }
            use HelpWindowEntry::{Empty, Item, Title};

            // Create an Egui window that starts closed.
            egui::Window::new("Help")
                .default_open(false)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        // Describe the controls-help layout.
                        // TODO: Complete controls list.
                        let controls_list = [
                            Title("App-Window"),
                            Item("F11", "Toggle window fullscreen"),
                            Item("ESC", "If fullscreen, then enter windowed mode. Else, close the application"),
                            Item("o", "Toggle visibility of the app overlay"),
                            Empty(),
                            Title("Game"),
                            Item("F5", "Generate a new random world and reset game"),
                            Empty(),
                            Title("Flight"),
                            Item("UP", "Pitch up"),
                            Item("DOWN", "Pitch down"),
                            Item("LEFT", "Roll left"),
                            Item("RIGHT", "Roll right"),
                            Item("a", "Yaw left"),
                            Item("d", "Yaw right"),
                            Item("SPACE", "Boost"),
                        ];

                        // Grid of controls, showing the buttons and their corresponding actions.
                        egui::Grid::new("scheme_index_grid").show(ui, |ui| {
                            for entry in controls_list {
                                match entry {
                                    Empty() => {}
                                    Item(key, desc) => {
                                        ui.vertical_centered(|ui| ui.label(egui::RichText::new(key).monospace()));
                                        ui.label(desc);
                                    }
                                    Title(title) => {
                                        // Include a separator in the first row and the title in the second.
                                        ui.separator();
                                        ui.heading(title);
                                    }
                                }
                                ui.end_row();
                            }
                        });
                    });
                });
        }

        // If the overlay is not enabled, return no command buffer.
        if !self.overlay.is_visible {
            return None;
        }

        // Update the GUI state.
        gui.immediate_ui(|gui| {
            let ctx = gui.context();

            // Create a window for setting options.
            options_window(self, &ctx);

            // Create a window for describing the controls.
            help_window(&ctx);

            // Optionally create a window for showing run information.
            if let Some(start_time) = self.game.run.start {
                egui::Window::new("Run").show(&ctx, |ui| {
                    ui.heading(format!("Score: {}", self.game.run.points));
                    ui.label(format!("Level: {}", self.game.run.level));
                    ui.label(format!("Time: {:.3}s", start_time.elapsed().as_secs_f32()));
                });
            }
        });

        // Return a command buffer to draw the GUI.
        Some(gui.draw_on_subpass_image(renderer.swapchain_image_size()))
    }
}

fn create_random_world(
    allocators: &Allocators,
    random: &mut voxels::RandomOctreeHelper,
    pipeline: &Arc<GraphicsPipeline>,
    log_file: &mut LogFile,
) -> (Arc<PersistentDescriptorSet>, Subbuffer<[VoxelCompact]>) {
    // Generate a random voxel-octree.
    let (voxel_octree, stats) = voxels::generate_recursive_voxel_octree(random, 256, 10);
    log_file.log(
        format!(
            "Seed: {:?}, Voxel Count: {:?}, Portal Count: {:?}\n",
            random.get_seed(),
            stats.voxel_count,
            stats.goal_count
        )
        .as_str(),
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
    .expect("Failed to create voxel buffer.");

    (
        // Create a descriptor set for the voxel buffer data.
        PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, buffer.clone())],
        )
        .expect("Failed to create voxel buffer descriptor set."),
        buffer,
    )
}

// Simple helper to write logs to an app file and stdout.
impl LogFile {
    // Open a log file in the app directory.
    pub fn default() -> Self {
        // Get a reasonable path for the log file.
        let file_path = if let Some(p) = dirs::config_local_dir() {
            let dir = p.join(path::Path::new("voxel_flight_simulator"));
            if !dir.exists() {
                std::fs::create_dir_all(&dir).expect("Failed to create app directory.");
            }
            dir.join("log.txt")
        } else {
            path::PathBuf::from("voxel_flight_simulator_log.txt")
        };

        // Open the log file.
        Self(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)
                .expect("Failed to open log file."),
        )
    }

    // Write a message to the log file and stdout.
    pub fn log(&mut self, msg: &str) {
        use std::io::Write;
        print!("{msg}");
        if let Err(e) = write!(self.0, "{msg}") {
            eprintln!("Couldn't write to file: {e}");
        }
    }
}
