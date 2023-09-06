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

use std::time::Instant;

use cgmath::{Quaternion, Vector3};
use gilrs::Gilrs;

// Game constants.
pub const DEFAULT_CAMERA_POSITION: Vector3<f32> = Vector3::new(0.01, 0.2, -2.);
pub const DEFAULT_CAMERA_ORIENTATION: Quaternion<f32> = Quaternion::new(1., 0., 0., 0.);
pub const DEFAULT_CAMERA_SPEED: f32 = 0.175;

// Game state.
pub struct State {
    pub camera_position: Vector3<f32>,
    pub camera_quaternion: Quaternion<f32>,
    pub camera_speed: f32,
    pub gamepad: GamepadState,
    pub gilrs: Gilrs,
    pub keyboard: Keyboard,
    pub options: Options,
    pub run: Run,
}

// Game options.
pub struct Options {
    pub camera_boost: HoldOrToggle,
    pub hotas_mode: bool,
    pub invert_y: bool,
}

// Run state.
#[derive(Default)]
pub struct Run {
    pub level: u32,
    pub points: u32,
    pub start: Option<Instant>,
}

// State of the keyboard inputs relevant to the game.
#[allow(clippy::struct_excessive_bools)]
#[derive(Default)]
pub struct Keyboard {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub space: bool,
    pub a: bool,
    pub d: bool,
}

// State of the gamepad inputs relevant to the game.
#[derive(Default)]
pub struct GamepadState {
    pub left_stick: [f32; 2],
    pub yaw: SharedAxis,
    pub south_button: bool,
}

// Helper type for tracking an axis value when one or two buttons control the result.
pub enum SharedAxis {
    Single(f32),
    Split(f32, f32),
}

// Helper type for tracking activation that comes from either holding or toggling an input.
#[derive(Clone, Copy, PartialEq)]
pub enum HoldOrToggle {
    Hold,
    Toggle(bool),
}

// Manipulate the game state.
impl State {
    // Helper to reset the camera to the default position and orientation.
    pub fn reset_camera(&mut self) {
        self.camera_position = DEFAULT_CAMERA_POSITION;
        self.camera_quaternion = DEFAULT_CAMERA_ORIENTATION;
        self.camera_speed = DEFAULT_CAMERA_SPEED;
    }
}

// Initialize the game state with default values.
impl Default for State {
    fn default() -> Self {
        Self {
            camera_position: DEFAULT_CAMERA_POSITION,
            camera_quaternion: DEFAULT_CAMERA_ORIENTATION,
            camera_speed: DEFAULT_CAMERA_SPEED,
            gamepad: GamepadState::default(),
            gilrs: Gilrs::new().unwrap(),
            keyboard: Keyboard::default(),
            options: Options::default(),
            run: Run::default(),
        }
    }
}

// Default game options.
impl Default for Options {
    fn default() -> Self {
        Self {
            camera_boost: HoldOrToggle::Hold,
            hotas_mode: false,
            invert_y: true,
        }
    }
}

// Manipulate the run state.
impl Run {
    // Helper function to ensure a run has started if conditions are met.
    pub fn ensure_running_if(&mut self, start_game: bool) {
        match self.start {
            // If the game has received actions to start and isn't running, transition to the running state.
            None if start_game => self.start = Some(Instant::now()),
            _ => (),
        }
    }
}

// Make managaing the gamepad state easier with default axis value and type.
impl Default for SharedAxis {
    fn default() -> Self {
        Self::Single(0.)
    }
}

// Return the stored toggle state; false for hold and state dependent for toggle.
impl From<HoldOrToggle> for bool {
    fn from(hot: HoldOrToggle) -> Self {
        match hot {
            HoldOrToggle::Toggle(t) => t,
            HoldOrToggle::Hold => false,
        }
    }
}
