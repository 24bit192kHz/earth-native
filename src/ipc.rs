use std::{
    env,
    ffi::OsString,
    fs,
    io::{self, BufRead, BufReader, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

pub const SOCKET_NAME: &str = "earth-native.sock";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Status,
    CaptureFrame,
    WeatherReload,
    Camera { yaw: String, pitch: String, distance: String },
    CameraLive,
    /// The far globe view centred under the ISS.
    CameraGlobe,
    /// Switch between the ISS window view and the globe view.
    CameraNext,
    /// Zoom in (positive) or out by lens/orbit steps.
    CameraZoom { steps: i32 },
    /// Look straight ahead again with the default lens.
    CameraReset,
    /// Face the strongest aurora in the dark right now, from ISS altitude.
    CameraAurora,
    /// Onboard view: heading from the flight direction, pitch below the
    /// horizontal ("auto" frames the horizon), horizontal field of view.
    CameraIss { heading: String, pitch: String, fov: String },
    /// Fixed onboard-style view over a point: latitude, longitude, altitude
    /// km, heading from north, pitch ("auto" allowed), field of view.
    CameraPov { latitude: String, longitude: String, altitude: String, heading: String, pitch: String, fov: String },
    Stop,
    Control { monitor: String },
    Body { body: String },
    CelestialShow,
    CelestialFreeze,
    CelestialSun { yaw: String, pitch: String },
    CelestialMoon { yaw: String, pitch: String },
    CelestialLive,
    TimeShow,
    TimeUnix { seconds: i64 },
    TimeLive,
    /// Presentation look: realistic or cinematic (see look.rs).
    Look { look: String },
}

impl Request {
    pub fn encode(&self) -> String {
        match self {
            Self::Status => "status\n".to_owned(),
            Self::CaptureFrame => "capture_frame\n".to_owned(),
            Self::WeatherReload => "weather reload\n".to_owned(),
            Self::Camera { yaw, pitch, distance } => format!("camera {yaw} {pitch} {distance}\n"),
            Self::CameraLive => "camera live\n".to_owned(),
            Self::CameraGlobe => "camera globe\n".to_owned(),
            Self::CameraNext => "camera next\n".to_owned(),
            Self::CameraZoom { steps } => format!("camera zoom {}\n", if *steps > 0 { "in" } else { "out" }),
            Self::CameraReset => "camera reset\n".to_owned(),
            Self::CameraAurora => "camera aurora\n".to_owned(),
            Self::CameraIss { heading, pitch, fov } => format!("camera iss {heading} {pitch} {fov}\n"),
            Self::CameraPov { latitude, longitude, altitude, heading, pitch, fov } =>
                format!("camera pov {latitude} {longitude} {altitude} {heading} {pitch} {fov}\n"),
            Self::Stop => "stop\n".to_owned(),
            Self::Control { monitor } => format!("control {monitor}\n"),
            Self::Body { body } => format!("body {body}\n"),
            Self::CelestialShow => "celestial show\n".to_owned(),
            Self::CelestialFreeze => "celestial freeze\n".to_owned(),
            Self::CelestialSun { yaw, pitch } => format!("celestial sun {yaw} {pitch}\n"),
            Self::CelestialMoon { yaw, pitch } => format!("celestial moon {yaw} {pitch}\n"),
            Self::CelestialLive => "celestial live\n".to_owned(),
            Self::TimeShow => "time show\n".to_owned(),
            Self::TimeUnix { seconds } => format!("time unix {seconds}\n"),
            Self::TimeLive => "time live\n".to_owned(),
            Self::Look { look } => format!("look {look}\n"),
        }
    }

    pub fn parse(line: &str) -> Result<Self, &'static str> {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        match fields.as_slice() {
            ["status"] => Ok(Self::Status),
            ["capture_frame"] => Ok(Self::CaptureFrame),
            ["weather", "reload"] => Ok(Self::WeatherReload),
            ["camera", "live"] => Ok(Self::CameraLive),
            ["camera", "globe"] => Ok(Self::CameraGlobe),
            ["camera", "next"] => Ok(Self::CameraNext),
            ["camera", "zoom", "in"] => Ok(Self::CameraZoom { steps: 1 }),
            ["camera", "zoom", "out"] => Ok(Self::CameraZoom { steps: -1 }),
            ["camera", "reset"] => Ok(Self::CameraReset),
            ["camera", "aurora"] => Ok(Self::CameraAurora),
            ["camera", "iss", heading, pitch, fov] => Ok(Self::CameraIss {
                heading: (*heading).to_owned(), pitch: (*pitch).to_owned(), fov: (*fov).to_owned(),
            }),
            ["camera", "pov", latitude, longitude, altitude, heading, pitch, fov] => Ok(Self::CameraPov {
                latitude: (*latitude).to_owned(), longitude: (*longitude).to_owned(), altitude: (*altitude).to_owned(),
                heading: (*heading).to_owned(), pitch: (*pitch).to_owned(), fov: (*fov).to_owned(),
            }),
            ["camera", yaw, pitch, distance] => Ok(Self::Camera {
                yaw: (*yaw).to_owned(), pitch: (*pitch).to_owned(), distance: (*distance).to_owned(),
            }),
            ["stop"] => Ok(Self::Stop),
            ["control", monitor] if !monitor.is_empty() => Ok(Self::Control {
                monitor: (*monitor).to_owned(),
            }),
            ["body", name] => Ok(Self::Body {
                body: (*name).to_owned(),
            }),
            ["celestial", "show"] => Ok(Self::CelestialShow),
            ["celestial", "freeze"] => Ok(Self::CelestialFreeze),
            ["celestial", "live"] => Ok(Self::CelestialLive),
            ["celestial", "sun", yaw, pitch] => Ok(Self::CelestialSun {
                yaw: (*yaw).to_owned(),
                pitch: (*pitch).to_owned(),
            }),
            ["celestial", "moon", yaw, pitch] => Ok(Self::CelestialMoon {
                yaw: (*yaw).to_owned(),
                pitch: (*pitch).to_owned(),
            }),
            ["time", "show"] => Ok(Self::TimeShow),
            ["time", "unix", seconds] => Ok(Self::TimeUnix {
                seconds: seconds.parse().map_err(|_| "Unix seconds must be an i64")?,
            }),
            ["time", "live"] => Ok(Self::TimeLive),
            ["look", look] if crate::look::Look::parse(look).is_some() => Ok(Self::Look {
                look: look.to_ascii_lowercase(),
            }),
            _ => Err("expected: status | stop | capture_frame | camera {live|globe|next|aurora|zoom in|zoom out|reset|iss HEADING PITCH FOV|pov LAT LON ALT_KM HEADING PITCH FOV|YAW PITCH DISTANCE} | control <monitor> | body {earth|moon|mercury|venus|mars|jupiter|saturn|uranus|neptune} | celestial {show|freeze|sun|moon|live} | time {show|unix SECONDS|live} | look {realistic|cinematic}"),
        }
    }
}

pub fn socket_path() -> PathBuf {
    let runtime = env::var_os("XDG_RUNTIME_DIR")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OsString::from("/tmp"));
    PathBuf::from(runtime).join(SOCKET_NAME)
}

pub fn bind_listener() -> io::Result<UnixListener> {
    let path = socket_path();
    match UnixListener::bind(&path) {
        Ok(listener) => {
            listener.set_nonblocking(true)?;
            Ok(listener)
        }
        Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
            if request_raw("status").is_ok() {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    "earth-native is already running",
                ));
            }
            fs::remove_file(&path)?;
            let listener = UnixListener::bind(path)?;
            listener.set_nonblocking(true)?;
            Ok(listener)
        }
        Err(error) => Err(error),
    }
}

pub fn remove_socket() {
    let _ = fs::remove_file(socket_path());
}

pub fn request(request: Request) -> io::Result<String> {
    request_raw(&request.encode())
}

/// Queue a native Vulkan readback, then wait for all atomic completion sidecars.
pub fn capture_frame() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let response = request(Request::CaptureFrame)?;
    let mut ticket: serde_json::Value = serde_json::from_str(&response)
        .map_err(|_| format!("capture rejected: {response}"))?;
    let paths = ticket["images"].as_array().ok_or("capture has no images")?;
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut captures = Vec::new();
    for path in paths {
        let metadata = PathBuf::from(path.as_str().ok_or("invalid capture path")?)
            .with_extension("json");
        loop {
            match fs::read(&metadata) {
                Ok(bytes) => {
                    let metadata = serde_json::from_slice::<serde_json::Value>(&bytes)?;
                    if let Some(error) = metadata.get("error") {
                        return Err(format!("capture failed: {error}").into());
                    }
                    captures.push(metadata);
                    break;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound && Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(error) => return Err(format!("capture incomplete at {}: {error}", metadata.display()).into()),
            }
        }
    }
    ticket["captures"] = captures.into();
    ticket["status"] = "complete".into();
    Ok(ticket)
}

fn request_raw(message: &str) -> io::Result<String> {
    let mut stream = UnixStream::connect(socket_path())?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(message.as_bytes())?;
    stream.shutdown(std::net::Shutdown::Write)?;
    let mut response = String::new();
    // Requests are one short line (<64 B); the default 8 KiB BufReader buffer
    // is pure per-connection malloc churn.
    BufReader::with_capacity(256, stream).read_line(&mut response)?;
    Ok(response.trim_end().to_owned())
}

pub fn read_request(stream: &UnixStream) -> io::Result<Request> {
    stream.set_read_timeout(Some(Duration::from_millis(100)))?;
    let mut line = String::new();
    // Requests are one short line (<64 B); the default 8 KiB BufReader buffer
    // is pure per-connection malloc churn.
    BufReader::with_capacity(256, stream).read_line(&mut line)?;
    Request::parse(&line).map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))
}

pub fn reply(mut stream: UnixStream, response: &str) {
    let _ = writeln!(stream, "{response}");
}

pub fn selected_monitor() -> io::Result<String> {
    let output = Command::new("hyprctl")
        .args(["activeworkspace", "-j"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            "hyprctl could not resolve the active monitor",
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid hyprctl JSON: {error}"),
        )
    })?;
    value
        .get("monitor")
        .and_then(serde_json::Value::as_str)
        .filter(|monitor| !monitor.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "active workspace has no monitor"))
}

pub fn start_daemon(debug: bool) -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(status) = request(Request::Status) {
        return Ok(status);
    }

    let executable = env::current_exe()?;
    let mut command = Command::new(executable);
    command.arg("serve");
    if debug {
        command.arg("--debug");
    }
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        match request(Request::Status) {
            Ok(status) => return Ok(status),
            Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            Err(error) => return Err(format!("earth-native did not start: {error}").into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Request;

    #[test]
    fn request_protocol_is_closed_and_unambiguous() {
        assert_eq!(Request::parse("status\n"), Ok(Request::Status));
        assert_eq!(Request::parse(&Request::CaptureFrame.encode()), Ok(Request::CaptureFrame));
        assert!(Request::parse("capture_frame extra").is_err());
        assert_eq!(Request::parse("stop"), Ok(Request::Stop));
        assert_eq!(
            Request::parse("control DP-1"),
            Ok(Request::Control {
                monitor: "DP-1".to_owned()
            })
        );
        assert_eq!(Request::parse("celestial show"), Ok(Request::CelestialShow));
        assert_eq!(
            Request::parse("celestial freeze"),
            Ok(Request::CelestialFreeze)
        );
        assert_eq!(
            Request::parse("celestial sun 90 -12.5"),
            Ok(Request::CelestialSun {
                yaw: "90".to_owned(),
                pitch: "-12.5".to_owned()
            })
        );
        assert_eq!(Request::parse("celestial live"), Ok(Request::CelestialLive));
        assert_eq!(Request::parse("time show"), Ok(Request::TimeShow));
        assert_eq!(
            Request::parse("time unix -123"),
            Ok(Request::TimeUnix { seconds: -123 })
        );
        assert_eq!(Request::parse("time live"), Ok(Request::TimeLive));
        assert!(Request::parse("control").is_err());
        assert!(Request::parse("stop now").is_err());
        assert!(Request::parse("celestial sun 1").is_err());
        assert!(Request::parse("celestial moon 1 2 extra").is_err());
        assert!(Request::parse("time unix nope").is_err());
        assert!(Request::parse("time unix 1 extra").is_err());
    }

    #[test]
    fn camera_view_requests_round_trip() {
        for request in [Request::CameraNext, Request::CameraReset, Request::CameraAurora,
            Request::CameraZoom { steps: 1 }, Request::CameraZoom { steps: -1 }] {
            assert_eq!(Request::parse(&request.encode()), Ok(request));
        }
        assert_eq!(Request::parse("camera zoom in"), Ok(Request::CameraZoom { steps: 1 }));
        assert!(Request::parse("camera zoom").is_err());
        assert!(Request::parse("camera zoom sideways").is_err());
        assert!(Request::parse("camera next extra").is_err());
    }

    #[test]
    fn time_requests_encode_canonically() {
        assert_eq!(Request::TimeShow.encode(), "time show\n");
        assert_eq!(
            Request::TimeUnix { seconds: i64::MIN }.encode(),
            format!("time unix {}\n", i64::MIN)
        );
        assert_eq!(Request::TimeLive.encode(), "time live\n");
    }

    #[test]
    fn look_requests_round_trip() {
        let request = Request::Look { look: "cinematic".to_owned() };
        assert_eq!(Request::parse(&request.encode()), Ok(request));
        assert_eq!(Request::parse("look Realistic"), Ok(Request::Look { look: "realistic".to_owned() }));
        assert!(Request::parse("look pretty").is_err());
        assert!(Request::parse("look").is_err());
    }
}
