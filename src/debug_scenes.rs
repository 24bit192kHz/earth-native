use crate::camera::Vec3;

use std::time::Duration;

pub const SCENE_DURATION: Duration = Duration::from_secs(4);
pub const SCENE_COUNT: usize = 9;

#[derive(Clone, Copy, Debug)]
pub struct DebugScene {
    pub name: &'static str,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_distance: f32,
    pub sun_direction: Vec3,
    pub moon_direction: Vec3,
}

pub fn scene(index: usize) -> DebugScene {
    match index {
        0 => DebugScene {
            name: "day-earth",
            camera_yaw: 180.0,
            camera_pitch: 0.0,
            camera_distance: 5.5,
            sun_direction: Vec3::new(1.0, 0.0, 0.0),
            moon_direction: Vec3::new(-0.98, 0.2, 0.0),
        },
        1 => DebugScene {
            name: "night-earth",
            camera_yaw: 180.0,
            camera_pitch: 0.0,
            camera_distance: 5.5,
            sun_direction: Vec3::new(-1.0, 0.0, 0.0),
            moon_direction: Vec3::new(-0.98, 0.2, 0.0),
        },
        2 => DebugScene {
            name: "sun-limb",
            camera_yaw: 180.0,
            camera_pitch: 0.0,
            camera_distance: 5.5,
            sun_direction: Vec3::new(-0.98, -0.2, 0.04),
            moon_direction: Vec3::new(-0.98, 0.2, 0.0),
        },
        3 => DebugScene {
            name: "moon-limb",
            camera_yaw: 180.0,
            camera_pitch: 0.0,
            camera_distance: 5.5,
            sun_direction: Vec3::new(1.0, 0.0, 0.0),
            moon_direction: Vec3::new(-0.98, 0.2, 0.0),
        },
        4 => DebugScene {
            name: "twilight-close",
            camera_yaw: 180.0,
            camera_pitch: -8.0,
            camera_distance: 1.8,
            sun_direction: Vec3::new(0.35, 0.94, 0.0),
            moon_direction: Vec3::new(-0.98, 0.2, 0.0),
        },
        // Tuning views (ECEF sun vectors): look straight at the night north
        // pole to see the aurora oval, a half-lit globe for the terminator
        // gradient, and a close night ocean for lightning.
        5 => DebugScene {
            name: "aurora-pole",
            camera_yaw: 0.0,
            camera_pitch: -89.0,
            camera_distance: 4.0,
            sun_direction: Vec3::new(0.0, 0.0, -1.0),
            moon_direction: Vec3::new(0.0, 0.0, 1.0),
        },
        6 => DebugScene {
            name: "terminator-globe",
            camera_yaw: 90.0,
            camera_pitch: 0.0,
            camera_distance: 5.5,
            sun_direction: Vec3::new(1.0, 0.0, 0.0),
            moon_direction: Vec3::new(-0.98, 0.2, 0.0),
        },
        7 => DebugScene {
            name: "storm-night",
            camera_yaw: 180.0,
            camera_pitch: -10.0,
            camera_distance: 2.2,
            sun_direction: Vec3::new(-1.0, 0.0, 0.0),
            moon_direction: Vec3::new(-0.98, 0.2, 0.0),
        },
        8 => DebugScene {
            name: "aurora-oblique",
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            camera_distance: 4.5,
            sun_direction: Vec3::new(0.0, 0.0, -1.0),
            moon_direction: Vec3::new(0.0, 0.0, 1.0),
        },
        _ => unreachable!("debug scene index out of range"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_scene_set_covers_the_expected_validation_cases() {
        let names = (0..SCENE_COUNT)
            .map(|index| scene(index).name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "day-earth",
                "night-earth",
                "sun-limb",
                "moon-limb",
                "twilight-close",
                "aurora-pole",
                "terminator-globe",
                "storm-night",
                "aurora-oblique"
            ]
        );
        assert!(scene(0).sun_direction.dot(scene(1).sun_direction) < -0.9);
        assert!(scene(4).camera_distance < scene(0).camera_distance);
    }
}
