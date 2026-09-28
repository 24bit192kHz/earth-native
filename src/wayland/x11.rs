//! Xorg frontend: one desktop-type window per RandR monitor.
//!
//! Each window is marked `_NET_WM_WINDOW_TYPE_DESKTOP`, kept below, sticky
//! and out of taskbars/pagers, so EWMH window managers treat it as the
//! desktop background. An empty SHAPE input region makes it click-through
//! until `earth-native control` hands it the pointer and keyboard, mirroring
//! the Wayland layer-shell input region. Rendering, camera, astronomy and IPC
//! are the shared `NativeApp` paths; only windows and input live here.

use std::{collections::BTreeMap, os::fd::AsRawFd};

use ash::vk;
use calloop::{generic::FdWrapper, generic::Generic, Interest, LoopHandle, Mode, PostAction};
use x11rb::{
    connection::{Connection, RequestConnection},
    protocol::{
        randr::{self, ConnectionExt as _},
        shape::{self, ConnectionExt as _},
        xproto::{
            AtomEnum, ButtonIndex, ConfigureWindowAux, ConnectionExt as _, CreateWindowAux,
            EventMask, InputFocus, KeyButMask, PropMode, StackMode, Window, WindowClass,
        },
        Event,
    },
    wrapper::ConnectionExt as _,
    xcb_ffi::XCBConnection,
    COPY_DEPTH_FROM_PARENT, CURRENT_TIME, NONE,
};

use super::{NativeApp, OutputState};
use crate::{
    camera::{OutputTransform, PixelSize, Vec2},
    vulkan::{LogicalRect, RendererResult, SurfaceSource},
};

/// Linux X servers deliver evdev keycodes offset by 8.
const EVDEV_OFFSET: u8 = 8;

struct Monitor {
    name: String,
    x: i16,
    y: i16,
    width: u16,
    height: u16,
}

struct X11Window {
    window: Window,
    width: u16,
    height: u16,
}

pub struct X11Parts {
    connection: XCBConnection,
    root: Window,
    root_visual: u32,
    black_pixel: u32,
    windows: BTreeMap<u32, X11Window>,
    atoms: Atoms,
}

struct Atoms {
    wm_window_type: u32,
    wm_window_type_desktop: u32,
    wm_state: u32,
    wm_state_below: u32,
    wm_state_sticky: u32,
    wm_state_skip_taskbar: u32,
    wm_state_skip_pager: u32,
    wm_desktop: u32,
    wm_name: u32,
    utf8_string: u32,
    motif_wm_hints: u32,
}

impl Atoms {
    fn intern(connection: &XCBConnection) -> RendererResult<Self> {
        let atom = |name: &str| -> RendererResult<u32> {
            Ok(connection.intern_atom(false, name.as_bytes())?.reply()?.atom)
        };
        Ok(Self {
            wm_window_type: atom("_NET_WM_WINDOW_TYPE")?,
            wm_window_type_desktop: atom("_NET_WM_WINDOW_TYPE_DESKTOP")?,
            wm_state: atom("_NET_WM_STATE")?,
            wm_state_below: atom("_NET_WM_STATE_BELOW")?,
            wm_state_sticky: atom("_NET_WM_STATE_STICKY")?,
            wm_state_skip_taskbar: atom("_NET_WM_STATE_SKIP_TASKBAR")?,
            wm_state_skip_pager: atom("_NET_WM_STATE_SKIP_PAGER")?,
            wm_desktop: atom("_NET_WM_DESKTOP")?,
            wm_name: atom("_NET_WM_NAME")?,
            utf8_string: atom("UTF8_STRING")?,
            motif_wm_hints: atom("_MOTIF_WM_HINTS")?,
        })
    }
}

impl X11Parts {
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Click-through by default; the controlled monitor takes pointer and
    /// keyboard (the default input shape is the whole window).
    pub fn set_input(&self, output_id: u32, enabled: bool) {
        let Some(entry) = self.windows.get(&output_id) else {
            return;
        };
        let result = if enabled {
            self.connection
                .shape_mask(shape::SO::SET, shape::SK::INPUT, entry.window, 0, 0, NONE)
                .map(|_| ())
                .and_then(|()| {
                    self.connection
                        .set_input_focus(InputFocus::PARENT, entry.window, CURRENT_TIME)
                        .map(|_| ())
                })
        } else {
            self.connection
                .shape_rectangles(
                    shape::SO::SET,
                    shape::SK::INPUT,
                    x11rb::protocol::xproto::ClipOrdering::UNSORTED,
                    entry.window,
                    0,
                    0,
                    &[],
                )
                .map(|_| ())
        };
        if let Err(error) = result.map_err(|e| e.to_string()).and_then(|()| {
            self.connection.flush().map_err(|e| e.to_string())
        }) {
            eprintln!("earth-native: X11 input region update failed: {error}");
        }
    }

    fn monitors(&self) -> RendererResult<Vec<Monitor>> {
        // RandR 1.5 monitors (one per logical monitor, including tiled
        // displays); fall back to the whole root when RandR is missing.
        if let Ok(cookie) = self.connection.randr_get_monitors(self.root, true) {
            if let Ok(reply) = cookie.reply() {
                let mut monitors = Vec::new();
                for info in reply.monitors {
                    let name = self
                        .connection
                        .get_atom_name(info.name)
                        .ok()
                        .and_then(|cookie| cookie.reply().ok())
                        .map(|reply| String::from_utf8_lossy(&reply.name).into_owned())
                        .unwrap_or_else(|| format!("monitor-{}", monitors.len()));
                    if info.width > 0 && info.height > 0 {
                        monitors.push(Monitor {
                            name,
                            x: info.x,
                            y: info.y,
                            width: info.width,
                            height: info.height,
                        });
                    }
                }
                if !monitors.is_empty() {
                    return Ok(monitors);
                }
            }
        }
        let geometry = self.connection.get_geometry(self.root)?.reply()?;
        Ok(vec![Monitor {
            name: "screen".to_owned(),
            x: 0,
            y: 0,
            width: geometry.width,
            height: geometry.height,
        }])
    }

    fn create_window(&self, monitor: &Monitor) -> RendererResult<Window> {
        let connection = &self.connection;
        let window = connection.generate_id()?;
        connection.create_window(
            COPY_DEPTH_FROM_PARENT,
            window,
            self.root,
            monitor.x,
            monitor.y,
            monitor.width,
            monitor.height,
            0,
            WindowClass::INPUT_OUTPUT,
            self.root_visual,
            &CreateWindowAux::new()
                .background_pixel(self.black_pixel)
                .event_mask(
                    EventMask::STRUCTURE_NOTIFY
                        | EventMask::BUTTON_PRESS
                        | EventMask::BUTTON_RELEASE
                        | EventMask::POINTER_MOTION
                        | EventMask::KEY_PRESS
                        | EventMask::KEY_RELEASE,
                ),
        )?;
        let atoms = &self.atoms;
        connection.change_property32(
            PropMode::REPLACE,
            window,
            atoms.wm_window_type,
            AtomEnum::ATOM,
            &[atoms.wm_window_type_desktop],
        )?;
        connection.change_property32(
            PropMode::REPLACE,
            window,
            atoms.wm_state,
            AtomEnum::ATOM,
            &[
                atoms.wm_state_below,
                atoms.wm_state_sticky,
                atoms.wm_state_skip_taskbar,
                atoms.wm_state_skip_pager,
            ],
        )?;
        // On every virtual desktop.
        connection.change_property32(PropMode::REPLACE, window, atoms.wm_desktop, AtomEnum::CARDINAL, &[u32::MAX])?;
        // No decorations (Motif hints: flags=decorations, decorations=0).
        connection.change_property32(PropMode::REPLACE, window, atoms.motif_wm_hints, atoms.motif_wm_hints, &[2, 0, 0, 0, 0])?;
        connection.change_property8(PropMode::REPLACE, window, atoms.wm_name, atoms.utf8_string, b"earth-native")?;
        connection.change_property8(
            PropMode::REPLACE,
            window,
            AtomEnum::WM_CLASS,
            AtomEnum::STRING,
            b"earth-native\0earth-native\0",
        )?;
        connection.map_window(window)?;
        // Window managers may place a mapped window; pin it to its monitor
        // and to the bottom of the stack.
        connection.configure_window(
            window,
            &ConfigureWindowAux::new()
                .x(i32::from(monitor.x))
                .y(i32::from(monitor.y))
                .width(u32::from(monitor.width))
                .height(u32::from(monitor.height))
                .stack_mode(StackMode::BELOW),
        )?;
        Ok(window)
    }
}

impl NativeApp {
    /// Build one desktop window per monitor and hand each to the renderer.
    fn x11_rebuild_outputs(&mut self) -> RendererResult<()> {
        for output_id in self.x11.as_ref().map(|x11| x11.windows.keys().copied().collect::<Vec<_>>()).unwrap_or_default() {
            self.renderer.destroy_output(output_id);
            if let Some(x11) = self.x11.as_mut() {
                if let Some(entry) = x11.windows.remove(&output_id) {
                    let _ = x11.connection.destroy_window(entry.window);
                }
            }
            self.outputs.remove(&output_id);
        }
        self.release_control();
        let Some(x11) = self.x11.as_mut() else {
            return Ok(());
        };
        let monitors = x11.monitors()?;
        let mut created = Vec::with_capacity(monitors.len());
        for (index, monitor) in monitors.iter().enumerate() {
            let output_id = index as u32 + 1;
            let window = x11.create_window(monitor)?;
            x11.windows.insert(output_id, X11Window { window, width: monitor.width, height: monitor.height });
            x11.set_input(output_id, false);
            created.push((output_id, window));
        }
        x11.connection.flush()?;
        for ((output_id, window), monitor) in created.into_iter().zip(&monitors) {
            self.outputs.insert(
                output_id,
                OutputState {
                    name: Some(monitor.name.clone()),
                    mode: PixelSize::new(u32::from(monitor.width), u32::from(monitor.height)),
                    scale: 1,
                    transform: OutputTransform::Normal,
                    logical_position: Some(Vec2::new(f32::from(monitor.x), f32::from(monitor.y))),
                    logical_size: Some(Vec2::new(f32::from(monitor.width), f32::from(monitor.height))),
                    ..OutputState::default()
                },
            );
            self.x11_configure_surface(output_id, window, monitor.width, monitor.height, &monitor.name);
        }
        self.layout_dirty = true;
        self.dirty = true;
        Ok(())
    }

    fn x11_configure_surface(&mut self, output_id: u32, window: Window, width: u16, height: u16, name: &str) {
        let Some(x11) = self.x11.as_ref() else {
            return;
        };
        let viewport = self
            .outputs
            .get(&output_id)
            .and_then(OutputState::logical_rect)
            .map(|rect| LogicalRect { x: rect.origin.x, y: rect.origin.y, width: rect.size.x, height: rect.size.y })
            .unwrap_or(LogicalRect { x: 0.0, y: 0.0, width: f32::from(width), height: f32::from(height) });
        let source = SurfaceSource::Xcb {
            connection: x11.connection.get_raw_xcb_connection().cast::<vk::xcb_connection_t>(),
            window,
        };
        let extent = vk::Extent2D { width: u32::from(width), height: u32::from(height) };
        if let Err(error) = unsafe { self.renderer.configure_output(output_id, source, extent, viewport, name) } {
            eprintln!("earth-native: Vulkan output {output_id} configuration failed: {error}");
        }
    }

    fn x11_output_for_window(&self, window: Window) -> Option<u32> {
        self.x11
            .as_ref()?
            .windows
            .iter()
            .find_map(|(id, entry)| (entry.window == window).then_some(*id))
    }

    fn x11_handle_event(&mut self, event: Event) -> RendererResult<()> {
        match event {
            Event::RandrScreenChangeNotify(_) | Event::RandrNotify(_) => self.x11_rebuild_outputs()?,
            Event::ConfigureNotify(configure) => {
                let Some(output_id) = self.x11_output_for_window(configure.window) else {
                    return Ok(());
                };
                let changed = self.x11.as_mut().and_then(|x11| x11.windows.get_mut(&output_id)).is_some_and(|entry| {
                    let changed = entry.width != configure.width || entry.height != configure.height;
                    entry.width = configure.width;
                    entry.height = configure.height;
                    changed
                });
                if changed {
                    // A window manager may size the window differently from its
                    // monitor; the projection must follow the real geometry or
                    // the globe is drawn with the monitor's aspect ratio.
                    let origin = self.x11.as_ref().and_then(|x11| {
                        x11.connection.translate_coordinates(configure.window, x11.root, 0, 0).ok()?.reply().ok()
                    });
                    if let Some(output) = self.outputs.get_mut(&output_id) {
                        output.mode = PixelSize::new(u32::from(configure.width), u32::from(configure.height));
                        output.logical_size = Some(Vec2::new(f32::from(configure.width), f32::from(configure.height)));
                        if let Some(origin) = origin {
                            output.logical_position = Some(Vec2::new(f32::from(origin.dst_x), f32::from(origin.dst_y)));
                        }
                    }
                    let name = self.outputs.get(&output_id).and_then(|output| output.name.clone()).unwrap_or_default();
                    self.x11_configure_surface(output_id, configure.window, configure.width, configure.height, &name);
                    self.layout_dirty = true;
                }
            }
            Event::ButtonPress(press) => {
                let output_id = self.x11_output_for_window(press.event);
                if output_id.is_none() || output_id != self.controlled_output {
                    return Ok(());
                }
                self.pointer.output_id = output_id;
                match press.detail {
                    button if button == u8::from(ButtonIndex::M1) => {
                        self.pointer.dragging = true;
                        self.pointer.last_position = Some((f64::from(press.event_x), f64::from(press.event_y)));
                        self.mark_interactive();
                    }
                    button if button == u8::from(ButtonIndex::M4) => {
                        if self.control_zoom(-1.0) {
                            self.mark_interactive();
                        }
                    }
                    button if button == u8::from(ButtonIndex::M5) => {
                        if self.control_zoom(1.0) {
                            self.mark_interactive();
                        }
                    }
                    _ => {}
                }
            }
            Event::ButtonRelease(release) if release.detail == u8::from(ButtonIndex::M1) => {
                self.pointer.dragging = false;
            }
            Event::MotionNotify(motion) => {
                let position = (f64::from(motion.event_x), f64::from(motion.event_y));
                let previous = self.pointer.last_position.replace(position);
                if self.pointer.dragging && self.pointer.output_id == self.controlled_output {
                    if let Some((previous_x, previous_y)) = previous {
                        let (orbit_x, orbit_y) =
                            super::orbit_delta_from_surface_motion(position.0 - previous_x, position.1 - previous_y);
                        if self.control_drag(orbit_x, orbit_y) {
                            self.mark_interactive();
                        }
                    }
                }
            }
            Event::KeyPress(key) | Event::KeyRelease(key) => {
                if self.controlled_output.is_none() {
                    return Ok(());
                }
                let pressed = matches!(event, Event::KeyPress(_));
                let code = key.detail.saturating_sub(EVDEV_OFFSET);
                let ctrl = u16::from(key.state) & u16::from(KeyButMask::CONTROL) != 0;
                self.handle_evdev_key(u32::from(code), pressed, ctrl);
            }
            _ => {}
        }
        Ok(())
    }
}

/// Connect to the X server, create the monitor windows, and feed X events
/// into the shared app from the calloop event loop.
pub fn attach(app: &mut NativeApp, handle: &LoopHandle<'static, NativeApp>) -> RendererResult<()> {
    let (connection, screen_index) = XCBConnection::connect(None)?;
    let screen = &connection.setup().roots[screen_index];
    let (root, root_visual, black_pixel) = (screen.root, screen.root_visual, screen.black_pixel);
    // Monitor hotplug and mode changes.
    if connection.extension_information(randr::X11_EXTENSION_NAME)?.is_some() {
        connection.randr_select_input(
            root,
            randr::NotifyMask::SCREEN_CHANGE | randr::NotifyMask::CRTC_CHANGE | randr::NotifyMask::OUTPUT_CHANGE,
        )?;
    }
    if connection.extension_information(shape::X11_EXTENSION_NAME)?.is_none() {
        eprintln!("earth-native: X SHAPE extension missing; the background window will not be click-through");
    }
    let atoms = Atoms::intern(&connection)?;
    let fd = connection.as_raw_fd();
    app.x11 = Some(X11Parts {
        connection,
        root,
        root_visual,
        black_pixel,
        windows: BTreeMap::new(),
        atoms,
    });
    app.x11_rebuild_outputs()?;
    // SAFETY: the fd belongs to the XCB connection owned by `app.x11`, which
    // lives until the event loop (and this source) is dropped at shutdown.
    let source = Generic::new(unsafe { FdWrapper::new(fd) }, Interest::READ, Mode::Level);
    handle.insert_source(source, |_, _, app: &mut NativeApp| {
        loop {
            let event = match app.x11.as_ref().map(|x11| x11.connection.poll_for_event()) {
                Some(Ok(Some(event))) => event,
                Some(Ok(None)) | None => break,
                Some(Err(error)) => {
                    eprintln!("earth-native: X connection error: {error}");
                    app.request_stop();
                    break;
                }
            };
            if let Err(error) = app.x11_handle_event(event) {
                eprintln!("earth-native: X event handling failed: {error}");
            }
        }
        if let Some(x11) = app.x11.as_ref() {
            let _ = x11.connection.flush();
        }
        Ok(PostAction::Continue)
    })?;
    Ok(())
}
