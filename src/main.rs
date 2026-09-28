mod asset_bundle;
mod astronomy;
mod atmosphere;
mod body;
mod camera;
mod data_dir;
mod day_color;
mod debug_capture;
mod debug_scenes;
mod earthvt;
mod ipc;
mod orbit;
mod sgp4;
mod sky;
mod star_catalog;
mod star_panorama;
mod vt_feedback;
#[allow(dead_code)]
mod vt_streamer;
mod vulkan;
mod wayland;
mod weather;

use std::{env, io, time::Duration};

use calloop::{
    generic::Generic,
    timer::{TimeoutAction, Timer},
    EventLoop, Interest, Mode, PostAction,
};
use calloop_wayland_source::WaylandSource;
use wayland_client::Connection;

use crate::{ipc::Request, vulkan::Renderer, wayland::NativeApp};

fn main() {
    if let Err(error) = run() {
        eprintln!("earth-native: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "start".to_owned());
    match command.as_str() {
        "start" => {
            let debug = parse_debug_flag(&mut args)?;
            println!("{}", ipc::start_daemon(debug)?);
            Ok(())
        }
        "stop" => {
            if args.next().is_some() {
                return usage();
            }
            println!("{}", ipc::request(Request::Stop)?);
            Ok(())
        }
        "restart" => {
            let debug = parse_debug_flag(&mut args)?;
            let _ = ipc::request(Request::Stop);
            for _ in 0..20 {
                if ipc::request(Request::Status).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            println!("{}", ipc::start_daemon(debug)?);
            Ok(())
        }
        "status" => {
            if args.next().is_some() {
                return usage();
            }
            println!("{}", ipc::request(Request::Status)?);
            Ok(())
        }
        "capture_frame" | "capture-frame" => {
            if args.next().is_some() {
                return usage();
            }
            println!("{}", ipc::capture_frame()?);
            Ok(())
        }
        "weather" => {
            if args.next().as_deref() != Some("reload") || args.next().is_some() { return usage(); }
            let response = ipc::request(Request::WeatherReload)?;
            if response.starts_with("error") { return Err(response.into()); }
            println!("{response}");
            Ok(())
        }
        "camera" => {
            let rest = args.collect::<Vec<_>>();
            if rest.is_empty() {
                return Err("usage: earth-native camera {live|globe|next|aurora|zoom in|zoom out|reset|iss HEADING PITCH|auto FOV|pov LAT LON ALT_KM HEADING PITCH|auto [FOV]|YAW PITCH DISTANCE}".into());
            }
            let mut fields = rest.clone();
            // The field of view is optional on the command line.
            if fields[0] == "pov" && fields.len() == 6 {
                fields.push("78".to_owned());
            }
            if fields[0] == "iss" && fields.len() == 3 {
                fields.push("78".to_owned());
            }
            let request = Request::parse(&format!("camera {}", fields.join(" ")))?;
            let response = ipc::request(request)?;
            if response.starts_with("error") { return Err(response.into()); }
            println!("{response}");
            Ok(())
        }
        "validate-bundle" => {
            let path = args
                .next()
                .ok_or("usage: earth-native validate-bundle MANIFEST.json")?;
            if args.next().is_some() {
                return usage();
            }
            let bundle = asset_bundle::validate(path)
                .map_err(|error| format!("invalid asset bundle: {error}"))?;
            println!(
                "ok bundle={} artifacts={} channels={}",
                bundle.manifest_path.display(),
                bundle.artifact_count,
                bundle.channels.len()
            );
            Ok(())
        }
        "control" => {
            let monitor = match args.next() {
                Some(monitor) if args.next().is_none() => monitor,
                Some(_) => return usage(),
                None => ipc::selected_monitor()?,
            };
            println!("{}", ipc::request(Request::Control { monitor })?);
            Ok(())
        }
        "body" => {
            let name = args
                .next()
                .ok_or("usage: earth-native body {earth|moon|mercury|venus|mars|jupiter|saturn|uranus|neptune}")?;
            if args.next().is_some() {
                return usage();
            }
            if body::Body::parse(&name).is_none() {
                return Err("body must be earth, moon, mercury, venus, mars, jupiter, saturn, uranus, or neptune".into());
            }
            println!(
                "{}",
                ipc::request(Request::Body {
                    body: name.to_ascii_lowercase()
                })?
            );
            Ok(())
        }
        "celestial" => {
            let subcommand = args
                .next()
                .ok_or("usage: earth-native celestial {show|freeze|sun|moon|live}")?;
            let request = match subcommand.as_str() {
                "show" if args.next().is_none() => Request::CelestialShow,
                "freeze" if args.next().is_none() => Request::CelestialFreeze,
                "live" if args.next().is_none() => Request::CelestialLive,
                "sun" => Request::CelestialSun {
                    yaw: args
                        .next()
                        .ok_or("usage: earth-native celestial sun YAW PITCH")?,
                    pitch: args
                        .next()
                        .ok_or("usage: earth-native celestial sun YAW PITCH")?,
                },
                "moon" => Request::CelestialMoon {
                    yaw: args
                        .next()
                        .ok_or("usage: earth-native celestial moon YAW PITCH")?,
                    pitch: args
                        .next()
                        .ok_or("usage: earth-native celestial moon YAW PITCH")?,
                },
                _ => return usage(),
            };
            if args.next().is_some() {
                return usage();
            }
            println!("{}", ipc::request(request)?);
            Ok(())
        }
        "time" => {
            let subcommand = args
                .next()
                .ok_or("usage: earth-native time {show|unix SECONDS|live}")?;
            let request = match subcommand.as_str() {
                "show" if args.next().is_none() => Request::TimeShow,
                "unix" => Request::TimeUnix {
                    seconds: args
                        .next()
                        .ok_or("usage: earth-native time unix SECONDS")?
                        .parse::<i64>()
                        .map_err(|_| "Unix seconds must be an i64")?,
                },
                "live" if args.next().is_none() => Request::TimeLive,
                _ => return usage(),
            };
            if args.next().is_some() {
                return usage();
            }
            println!("{}", ipc::request(request)?);
            Ok(())
        }
        "serve" => {
            let debug = parse_debug_flag(&mut args)?;
            serve(debug)
        }
        _ => usage(),
    }
}

fn usage() -> Result<(), Box<dyn std::error::Error>> {
    Err("usage: earth-native {start [--debug]|stop|restart [--debug]|control [MONITOR]|body {earth|moon|mercury|venus|mars|jupiter|saturn|uranus|neptune}|status|capture_frame|camera {live|globe|next|aurora|reset|zoom {in|out}|iss HEADING PITCH FOV|pov LAT LON ALT_KM HEADING PITCH|YAW PITCH DISTANCE}|validate-bundle MANIFEST.json|celestial {show|freeze|sun YAW PITCH|moon YAW PITCH|live}|time {show|unix SECONDS|live}|serve [--debug]}".into())
}

fn parse_debug_flag(
    args: &mut impl Iterator<Item = String>,
) -> Result<bool, Box<dyn std::error::Error>> {
    match args.next() {
        None => Ok(false),
        Some(flag) if flag == "--debug" && args.next().is_none() => Ok(true),
        Some(_) => usage().map(|_| false),
    }
}

/// Which display server hosts the background.
enum Backend {
    Wayland,
    X11,
}

impl Backend {
    /// `EARTH_NATIVE_BACKEND=wayland|x11` forces one; otherwise Wayland when
    /// a Wayland display is available, then Xorg.
    fn from_environment() -> Result<Self, Box<dyn std::error::Error>> {
        match env::var("EARTH_NATIVE_BACKEND").ok().as_deref() {
            Some("wayland") => return Ok(Self::Wayland),
            Some("x11") | Some("xorg") => return Ok(Self::X11),
            Some(other) if !other.is_empty() => {
                return Err(format!("EARTH_NATIVE_BACKEND must be wayland or x11, not {other}").into())
            }
            _ => {}
        }
        if env::var_os("WAYLAND_DISPLAY").is_some() || env::var_os("WAYLAND_SOCKET").is_some() {
            Ok(Self::Wayland)
        } else if env::var_os("DISPLAY").is_some() {
            Ok(Self::X11)
        } else {
            Err("no display: set WAYLAND_DISPLAY (Wayland) or DISPLAY (Xorg)".into())
        }
    }
}

fn serve(debug_screenshots: bool) -> Result<(), Box<dyn std::error::Error>> {
    match data_dir::configure_environment() {
        Some(textures) => eprintln!("earth-native: textures={}", textures.display()),
        None if env::var_os("EARTH_NATIVE_NASA_DATA").is_none() => eprintln!(
            "earth-native: no texture pack found (see README: install the release data or run pipeline/data_pipeline.py build); using procedural fallback"
        ),
        None => {}
    }
    let (quality, bundle) = asset_bundle::configure_environment()
        .map_err(|error| format!("asset configuration: {error}"))?;
    eprintln!(
        "earth-native: quality={} bundle={}",
        quality.name(),
        bundle
            .as_ref()
            .map(|b| b.manifest_path.display().to_string())
            .unwrap_or_else(|| "none".into())
    );
    let renderer = Renderer::new(debug_screenshots, quality, bundle)?;
    let mut event_loop: EventLoop<NativeApp> = EventLoop::try_new()?;
    let loop_handle = event_loop.handle();
    let mut app = match Backend::from_environment()? {
        Backend::Wayland => {
            let connection = Connection::connect_to_env()?;
            let mut queue = connection.new_event_queue();
            let queue_handle = queue.handle();
            let display = connection.display();
            let mut app = NativeApp::new(display.clone(), queue_handle.clone(), renderer, debug_screenshots)?;
            display.get_registry(&queue_handle, ());
            queue.roundtrip(&mut app)?;
            WaylandSource::new(connection, queue).insert(loop_handle.clone())?;
            eprintln!("earth-native: backend=wayland (wlr-layer-shell background)");
            app
        }
        Backend::X11 => {
            let mut app = NativeApp::new_x11(renderer, debug_screenshots)?;
            wayland::x11::attach(&mut app, &loop_handle)?;
            eprintln!("earth-native: backend=x11 (EWMH desktop windows)");
            app
        }
    };

    let listener = ipc::bind_listener()?;
    loop_handle.insert_source(
        Generic::new(listener, Interest::READ, Mode::Level),
        |_, listener, app| {
            loop {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let response = match ipc::read_request(&stream) {
                            Ok(request) => app.handle_socket_request(request),
                            Err(error) => format!("error {error}"),
                        };
                        ipc::reply(stream, &response);
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                    Err(error) => return Err(error),
                }
            }
            Ok(PostAction::Continue)
        },
    )?;
    loop_handle.insert_source(Timer::immediate(), |_, _, app| {
        if let Err(error) = app.render_frame() {
            eprintln!("earth-native: render failure: {error}");
        }
        TimeoutAction::ToDuration(app.next_frame_interval())
    })?;

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        while !app.stopping() {
            event_loop.dispatch(Some(app.next_frame_interval()), &mut app)?;
        }
        Ok(())
    })();
    app.shutdown_renderer();
    ipc::remove_socket();
    result
}
