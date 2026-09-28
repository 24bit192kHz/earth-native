//! Non-blocking virtual-texture request, I/O, and residency scheduling.
//!
//! The render thread never opens a file, decodes an image, or waits for an
//! upload. It submits one-frame-late feedback to this scheduler and polls its
//! completed direct-upload payloads at a bounded rate. Vulkan atlas ownership
//! remains in the renderer so this module can be tested without a GPU.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::File,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError},
    thread,
    time::Instant,
};

use memmap2::{Mmap, MmapOptions};

use crate::earthvt::{
    AdmissionPlan, EarthVt, EarthVtError, LayerDescriptor, PixelFormat, ResidencyTracker,
    TextureChannel, TileKey, TileRequest, TileRequestQueue, LAYER_FLAG_CLAMP_Y, LAYER_FLAG_SRGB,
    LAYER_FLAG_WRAP_X,
};

// Keep back-pressure close to the renderer. A queued tile is a full GPU-ready
// payload, so a large completion queue can consume tens of MiB during a fast
// zoom without improving visual quality.
const REQUEST_CHANNEL_DEPTH: usize = 1_024;
const COMPLETION_CHANNEL_DEPTH: usize = 32;
const MAX_FEEDBACK_FRAMES: usize = 3;
const CROSSFADE_FRAMES: u64 = 6;
/// Requests/completions older than this that the newest feedback no longer
/// affirms are pan/zoom leftovers: drop before I/O (pending) or before GPU
/// upload (completed). 30 frames ~= 2 s idle / 1 s interactive; settled views
/// re-affirm every frame so are never affected. Crossfade is 6 frames, well inside.
const STALE_REQUEST_FRAMES: u64 = 30;

#[derive(Debug)]
pub struct UploadJob {
    pub key: TileKey,
    pub parent: Option<TileKey>,
    pub layer: LayerDescriptor,
    pub format: PixelFormat,
    pub content_hash: u32,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub enum StreamError {
    Io(std::io::Error),
    Container(EarthVtError),
    Worker(String),
    InvalidContainer(&'static str),
    Closed,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "virtual-texture I/O error: {error}"),
            Self::Container(error) => write!(formatter, "invalid .earthvt container: {error}"),
            Self::Worker(error) => write!(formatter, "virtual-texture worker error: {error}"),
            Self::InvalidContainer(reason) => {
                write!(formatter, "invalid virtual-texture container: {reason}")
            }
            Self::Closed => formatter.write_str("virtual-texture worker channel is closed"),
        }
    }
}

impl std::error::Error for StreamError {}

impl From<std::io::Error> for StreamError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<EarthVtError> for StreamError {
    fn from(error: EarthVtError) -> Self {
        Self::Container(error)
    }
}

/// A mapping handle intentionally reparses only the small validated metadata.
/// The payload remains borrowed from the mapping and never undergoes image
/// decoding on the render thread.
pub struct MappedEarthVt {
    mmap: Mmap,
}

impl MappedEarthVt {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StreamError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        EarthVt::parse(&mmap)?;
        Ok(Self { mmap })
    }

    pub fn parsed(&self) -> Result<EarthVt<'_>, StreamError> {
        Ok(EarthVt::parse(&self.mmap)?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Crossfade {
    pub parent: TileKey,
    pub child: TileKey,
    pub started_frame: u64,
}

impl Crossfade {
    pub fn child_weight(self, frame: u64) -> f32 {
        let elapsed = frame.saturating_sub(self.started_frame);
        (elapsed as f32 / CROSSFADE_FRAMES as f32).clamp(0.0, 1.0)
    }

    pub fn is_complete(self, frame: u64) -> bool {
        frame.saturating_sub(self.started_frame) >= CROSSFADE_FRAMES
    }
}

/// Render-thread facade around a single memory-mapped I/O worker.
pub struct VirtualTextureStreamer {
    requests: SyncSender<TileRequest>,
    completed: Receiver<(TileKey, Result<UploadJob, StreamError>)>,
    feedback_frames: VecDeque<(u64, Vec<TileRequest>)>,
    pending: TileRequestQueue,
    in_flight: BTreeMap<TileKey, TileRequest>,
    layers: Vec<LayerDescriptor>,
    residency: ResidencyTracker,
    crossfades: Vec<Crossfade>,
    crossfade_pins: BTreeMap<TileKey, usize>,
    dispatched_frame: u64,
    latest_visible: BTreeSet<TileKey>,
}

impl VirtualTextureStreamer {
    pub fn spawn(path: impl Into<PathBuf>, gpu_budget_bytes: u64) -> Result<Self, StreamError> {
        let path = path.into();
        let mapped = MappedEarthVt::open(&path)?;
        let container = mapped.parsed()?;
        validate_startup(&container)?;
        let layers = container.layers().to_vec();
        drop(container);
        drop(mapped);
        let (request_sender, request_receiver) = mpsc::sync_channel(REQUEST_CHANNEL_DEPTH);
        let (completion_sender, completion_receiver) = mpsc::sync_channel(COMPLETION_CHANNEL_DEPTH);
        thread::Builder::new()
            .name("earth-native-vt-io".to_owned())
            .spawn(move || worker_loop(path, request_receiver, completion_sender))
            .map_err(|error| StreamError::Worker(error.to_string()))?;
        Ok(Self::from_channels(
            request_sender,
            completion_receiver,
            layers,
            gpu_budget_bytes,
        ))
    }

    fn from_channels(
        requests: SyncSender<TileRequest>,
        completed: Receiver<(TileKey, Result<UploadJob, StreamError>)>,
        layers: Vec<LayerDescriptor>,
        gpu_budget_bytes: u64,
    ) -> Self {
        Self {
            requests,
            completed,
            feedback_frames: VecDeque::new(),
            pending: TileRequestQueue::default(),
            in_flight: BTreeMap::new(),
            layers,
            residency: ResidencyTracker::new(gpu_budget_bytes),
            crossfades: Vec::new(),
            crossfade_pins: BTreeMap::new(),
            dispatched_frame: 0,
            latest_visible: BTreeSet::new(),
        }
    }

    #[cfg(test)]
    fn new_for_test(channel_depth: usize) -> (Self, Receiver<TileRequest>) {
        let (request_sender, request_receiver) = mpsc::sync_channel(channel_depth);
        let (_completion_sender, completion_receiver) = mpsc::sync_channel(1);
        (
            Self::from_channels(request_sender, completion_receiver, Vec::new(), u64::MAX),
            request_receiver,
        )
    }

    pub fn layers(&self) -> &[LayerDescriptor] {
        &self.layers
    }

    pub fn layer_for_channel(&self, channel: TextureChannel) -> Option<LayerDescriptor> {
        self.layers
            .iter()
            .copied()
            .find(|layer| layer.channel == channel)
    }

    /// True while requested pages are queued, being read, or awaiting
    /// upload: the renderer keeps drawing frames until the view is resident
    /// (a static view is otherwise motion-gated and would stall streaming).
    pub fn busy(&self) -> bool {
        self.pending.len() > 0 || !self.in_flight.is_empty()
    }

    pub fn residency(&self) -> &ResidencyTracker {
        &self.residency
    }

    pub fn residency_mut(&mut self) -> &mut ResidencyTracker {
        &mut self.residency
    }

    /// Queue feedback generated on this frame. It is deliberately dispatched
    /// one frame later, matching asynchronous GPU readback semantics.
    pub fn submit_feedback(&mut self, frame: u64, requests: Vec<TileRequest>, zooming_in: bool) {
        let mut requests = requests;
        if zooming_in {
            requests.extend(prefetch_finer_mip(&requests));
        }
        self.feedback_frames.push_back((frame, requests));
        // Feedback is one-frame-late by design. Drop the oldest stale sample
        // if a slow upload or compositor stall lets producers outrun it.
        while self.feedback_frames.len() > MAX_FEEDBACK_FRAMES {
            self.feedback_frames.pop_front();
        }
    }

    pub fn dispatch_feedback(&mut self, current_frame: u64) -> Result<usize, StreamError> {
        let mut newest_visible = std::mem::take(&mut self.latest_visible);
        while self
            .feedback_frames
            .front()
            .is_some_and(|(frame, _)| current_frame > *frame)
        {
            let (_frame, requests) = self.feedback_frames.pop_front().expect("front checked");
            newest_visible.clear();
            newest_visible.extend(requests.iter().map(|request| request.key));
            for request in requests {
                if self.residency.is_resident(request.key) {
                    // Refresh recency so the 2s hysteresis below measures
                    // time-since-last-use (matching VtSlotAllocator::request),
                    // not time-since-upload. Settled-zoom tiles stay pinned
                    // by use; tiles that left the view age out on schedule.
                    self.residency.touch(request.key, Instant::now());
                    continue;
                }
                if let Some(existing) = self.in_flight.get_mut(&request.key) {
                    existing.merge(request);
                } else {
                    self.pending.enqueue(request);
                }
            }
        }
        self.dispatched_frame = current_frame;
        self.latest_visible = newest_visible;
        // Cheap path first: drop stale unaffirmed requests before they reach the worker FIFO.
        self.pending.retain_fresh(current_frame, STALE_REQUEST_FRAMES, &self.latest_visible);

        let mut sent = 0;
        let requests = self.pending.drain_prioritized();
        // `requests` is owned: clone once for the channel, move the original
        // into the in-flight map (was: two clones per request). Remainder
        // moves back by value on backpressure instead of re-cloning.
        let mut requests = requests.into_iter();
        while let Some(request) = requests.next() {
            let key = request.key;
            match self.requests.try_send(request.clone()) {
                Ok(()) => {
                    self.in_flight.insert(key, request);
                    sent += 1;
                }
                Err(mpsc::TrySendError::Full(_)) => {
                    self.pending.enqueue(request);
                    for unsent in requests {
                        self.pending.enqueue(unsent);
                    }
                    break;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => return Err(StreamError::Closed),
            }
        }
        Ok(sent)
    }

    /// Poll only completed worker results. The caller copies payloads into a
    /// persistently mapped Vulkan staging region and then calls `admit_upload`.
    pub fn poll_completed(&mut self, limit: usize) -> Result<Vec<UploadJob>, StreamError> {
        let mut jobs = Vec::new();
        while jobs.len() < limit {
            match self.completed.try_recv() {
                Ok((key, Ok(job))) => {
                    self.in_flight.remove(&key);
                    jobs.push(job);
                }
                Ok((key, Err(error))) => {
                    self.in_flight.remove(&key);
                    return Err(error);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(StreamError::Closed),
            }
        }
        Ok(jobs)
    }

    /// Single-job shorthand for `poll_completed(1)`: returns the next ready
    /// upload without allocating a one-element `Vec`. Stale unaffirmed
    /// completions are skipped (I/O already spent; saves the GPU upload).
    pub fn poll_one(&mut self) -> Result<Option<UploadJob>, StreamError> {
        loop {
            match self.completed.try_recv() {
                Ok((key, Ok(job))) => {
                    let stale = self.in_flight.get(&key).is_some_and(|request| {
                        !self.latest_visible.contains(&key)
                            && self.dispatched_frame.saturating_sub(request.requested_frame)
                                > STALE_REQUEST_FRAMES
                    });
                    self.in_flight.remove(&key);
                    if stale { continue; }
                    return Ok(Some(job));
                }
                Ok((key, Err(error))) => {
                    self.in_flight.remove(&key);
                    return Err(error);
                }
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => return Err(StreamError::Closed),
            }
        }
    }

    /// Applies fixed-budget LRU policy before a Vulkan upload. The caller must
    /// evict returned atlas slots before calling `mark_uploaded`.
    pub fn plan_upload(&self, job: &UploadJob, now: Instant) -> AdmissionPlan {
        self.residency.plan_admission(job.payload.len() as u64, now)
    }

    pub fn mark_uploaded(
        &mut self,
        job: &UploadJob,
        frame: u64,
        now: Instant,
    ) -> Result<(), StreamError> {
        let parent = job
            .parent
            .filter(|parent| self.residency.is_resident(*parent));
        if let Some(parent) = parent {
            *self.crossfade_pins.entry(parent).or_default() += 1;
            self.residency.set_pinned(parent, true);
        }
        if let Err(error) =
            self.residency
                .mark_resident(job.key, job.payload.len() as u64, now, false)
        {
            if let Some(parent) = parent {
                self.release_crossfade_pin(parent);
            }
            return Err(StreamError::Worker(error.to_string()));
        }
        if let Some(parent) = parent {
            self.crossfades.push(Crossfade {
                parent,
                child: job.key,
                started_frame: frame,
            });
        }
        Ok(())
    }

    pub fn crossfades(&mut self, frame: u64) -> impl Iterator<Item = Crossfade> + '_ {
        let mut expired = Vec::new();
        self.crossfades.retain(|fade| {
            if fade.is_complete(frame) {
                expired.push(fade.parent);
                false
            } else {
                true
            }
        });
        for parent in expired {
            self.release_crossfade_pin(parent);
        }
        self.crossfades.iter().copied()
    }

    fn release_crossfade_pin(&mut self, parent: TileKey) {
        match self.crossfade_pins.get_mut(&parent) {
            Some(count) if *count > 1 => *count -= 1,
            Some(_) => {
                self.crossfade_pins.remove(&parent);
                self.residency.set_pinned(parent, false);
            }
            None => {}
        }
    }
}

fn worker_loop(
    path: PathBuf,
    requests: Receiver<TileRequest>,
    completed: SyncSender<(TileKey, Result<UploadJob, StreamError>)>,
) {
    let mapped = match MappedEarthVt::open(&path) {
        Ok(mapped) => mapped,
        Err(error) => return worker_startup_error(completed, error),
    };
    let container = match mapped.parsed() {
        Ok(container) => container,
        Err(error) => return worker_startup_error(completed, error),
    };
    while let Ok(request) = requests.recv() {
        let key = request.key;
        if completed
            .send((key, build_upload_job(&container, key)))
            .is_err()
        {
            break;
        }
    }
}

fn worker_startup_error(
    completed: SyncSender<(TileKey, Result<UploadJob, StreamError>)>,
    error: StreamError,
) {
    let _ = completed.send((TileKey::new(0, 0, 0, 0), Err(error)));
}

fn validate_startup(container: &EarthVt<'_>) -> Result<(), StreamError> {
    let layer = container
        .layer_for_channel(TextureChannel::DayColor)
        .ok_or(StreamError::InvalidContainer("missing DayColor layer"))?;
    let required_flags = LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y;
    if layer.format != PixelFormat::Bc7 {
        return Err(StreamError::InvalidContainer("DayColor layer must use BC7"));
    }
    if layer.flags & required_flags != required_flags {
        return Err(StreamError::InvalidContainer(
            "DayColor layer must be SRGB, WRAP_X, and CLAMP_Y",
        ));
    }
    Ok(())
}

fn build_upload_job(container: &EarthVt<'_>, key: TileKey) -> Result<UploadJob, StreamError> {
    let layer = container.layer(key.layer).ok_or_else(|| {
        StreamError::Worker(format!("unknown virtual-texture layer {}", key.layer))
    })?;
    let tile = container
        .tile(key)
        .ok_or_else(|| StreamError::Worker(format!("missing tile {key:?}")))?;
    Ok(UploadJob {
        key,
        parent: container.parent_key(key),
        layer,
        format: layer.format,
        content_hash: tile.entry.content_hash,
        payload: tile.payload.to_vec(),
    })
}

fn prefetch_finer_mip(requests: &[TileRequest]) -> Vec<TileRequest> {
    let mut prefetch = Vec::new();
    for request in requests {
        if request.key.mip == 0 {
            continue;
        }
        let child_mip = request.key.mip - 1;
        for child_y in 0..2 {
            for child_x in 0..2 {
                prefetch.push(TileRequest::prefetch(
                    TileKey::new(
                        request.key.layer,
                        child_mip,
                        request.key.x.saturating_mul(2).saturating_add(child_x),
                        request.key.y.saturating_mul(2).saturating_add(child_y),
                    ),
                    request.priority / 4,
                    request.requested_frame,
                ));
            }
        }
    }
    prefetch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earthvt::{
        RequestKind, TileRequest, GUTTER_SIZE, HEADER_BYTES, INDEX_ENTRY_BYTES, LAYER_ENTRY_BYTES,
        MAGIC, TILE_SIZE, VERSION,
    };
    use std::{fs, path::PathBuf};

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn synthetic_container(channel: TextureChannel, format: PixelFormat, flags: u32) -> Vec<u8> {
        let mip_count = 9_usize;
        let layer_offset = HEADER_BYTES;
        let index_offset = layer_offset + LAYER_ENTRY_BYTES;
        let payload_offset = (index_offset + mip_count * INDEX_ENTRY_BYTES + 15) & !15;
        let tile_bytes = format.encoded_tile_bytes() as usize;
        let file_bytes = payload_offset + mip_count * tile_bytes;
        let mut bytes = vec![0_u8; file_bytes];
        bytes[..8].copy_from_slice(&MAGIC);
        put_u16(&mut bytes, 8, VERSION);
        put_u16(&mut bytes, 10, HEADER_BYTES as u16);
        put_u16(&mut bytes, 16, TILE_SIZE);
        put_u16(&mut bytes, 18, GUTTER_SIZE);
        put_u16(&mut bytes, 20, 1);
        put_u16(&mut bytes, 22, LAYER_ENTRY_BYTES as u16);
        put_u16(&mut bytes, 24, INDEX_ENTRY_BYTES as u16);
        put_u32(&mut bytes, 28, mip_count as u32);
        put_u64(&mut bytes, 32, layer_offset as u64);
        put_u64(&mut bytes, 40, index_offset as u64);
        put_u64(&mut bytes, 48, payload_offset as u64);
        put_u64(&mut bytes, 56, file_bytes as u64);
        put_u16(&mut bytes, layer_offset, 1);
        put_u16(&mut bytes, layer_offset + 2, channel.raw());
        put_u16(&mut bytes, layer_offset + 4, format.raw());
        put_u16(&mut bytes, layer_offset + 6, mip_count as u16);
        put_u32(&mut bytes, layer_offset + 8, 256);
        put_u32(&mut bytes, layer_offset + 12, 256);
        put_u32(&mut bytes, layer_offset + 16, 0);
        put_u32(&mut bytes, layer_offset + 20, mip_count as u32);
        put_u32(&mut bytes, layer_offset + 24, flags);
        for mip in 0..mip_count {
            let offset = index_offset + mip * INDEX_ENTRY_BYTES;
            let payload = payload_offset + mip * tile_bytes;
            put_u16(&mut bytes, offset, 1);
            put_u16(&mut bytes, offset + 2, mip as u16);
            put_u64(&mut bytes, offset + 16, payload as u64);
            put_u64(&mut bytes, offset + 24, tile_bytes as u64);
            put_u32(&mut bytes, offset + 32, mip as u32 + 1);
            bytes[payload] = mip as u8;
        }
        bytes
    }

    fn temp_container(bytes: &[u8], label: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("earth-native-vt-{label}-{}", std::process::id()));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn valid_layer() -> LayerDescriptor {
        LayerDescriptor {
            id: 1,
            channel: TextureChannel::DayColor,
            format: PixelFormat::Bc7,
            mip_count: 9,
            base_width: 256,
            base_height: 256,
            first_index: 0,
            index_count: 9,
            flags: LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
        }
    }

    #[test]
    fn finer_prefetch_expands_a_parent_tile() {
        let request = TileRequest::visible(TileKey::new(3, 4, 8, 5), 100, 7);
        let children = prefetch_finer_mip(&[request]);
        assert_eq!(children.len(), 4);
        assert!(children
            .iter()
            .all(|child| child.kind == RequestKind::Prefetch));
        assert!(children.iter().all(|child| child.key.mip == 3));
        assert!(children
            .iter()
            .any(|child| child.key.x == 16 && child.key.y == 10));
        assert!(children
            .iter()
            .any(|child| child.key.x == 17 && child.key.y == 11));
    }

    #[test]
    fn crossfade_is_bounded() {
        let fade = Crossfade {
            parent: TileKey::new(1, 2, 3, 4),
            child: TileKey::new(1, 1, 6, 8),
            started_frame: 10,
        };
        assert_eq!(fade.child_weight(10), 0.0);
        assert_eq!(fade.child_weight(16), 1.0);
        assert!(fade.is_complete(16));
    }

    #[test]
    fn spawn_validates_day_color_before_starting_worker() {
        let valid_path = temp_container(
            &synthetic_container(
                TextureChannel::DayColor,
                PixelFormat::Bc7,
                LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
            ),
            "valid",
        );
        let streamer = VirtualTextureStreamer::spawn(&valid_path, 1_000_000).unwrap();
        assert_eq!(
            streamer
                .layer_for_channel(TextureChannel::DayColor)
                .unwrap()
                .format,
            PixelFormat::Bc7
        );
        fs::remove_file(&valid_path).unwrap();

        for (label, channel, format, flags) in [
            ("missing", TextureChannel::CloudDensity, PixelFormat::Bc4, 0),
            (
                "flags",
                TextureChannel::DayColor,
                PixelFormat::Bc7,
                LAYER_FLAG_WRAP_X,
            ),
            (
                "format",
                TextureChannel::DayColor,
                PixelFormat::Bc5,
                LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
            ),
        ] {
            let path = temp_container(&synthetic_container(channel, format, flags), label);
            assert!(VirtualTextureStreamer::spawn(&path, 1_000_000).is_err());
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn backpressure_preserves_visible_pending_requests() {
        let (mut streamer, receiver) = VirtualTextureStreamer::new_for_test(1);
        let first = TileKey::new(1, 0, 0, 0);
        let second = TileKey::new(1, 0, 1, 0);
        streamer.submit_feedback(
            0,
            vec![
                TileRequest::visible(first, 10, 0),
                TileRequest::visible(second, 20, 0),
            ],
            false,
        );
        assert_eq!(streamer.dispatch_feedback(1).unwrap(), 1);
        let first_sent = receiver.recv().unwrap();
        assert_eq!(first_sent.key, second);
        assert_eq!(streamer.dispatch_feedback(2).unwrap(), 1);
        assert_eq!(receiver.recv().unwrap().key, first);
    }
    #[test]
    fn stale_unaffirmed_requests_expire_from_pending() {
        let (mut streamer, receiver) = VirtualTextureStreamer::new_for_test(1);
        let blocker = TileKey::new(1, 0, 2, 0);
        streamer.submit_feedback(0, vec![TileRequest::visible(blocker, 100, 0)], false);
        streamer.dispatch_feedback(1).unwrap();
        let old = TileKey::new(1, 0, 0, 0);
        let live = TileKey::new(1, 0, 1, 0);
        streamer.submit_feedback(
            1,
            vec![
                TileRequest::visible(old, 10, 1),
                TileRequest::visible(live, 10, 1),
            ],
            false,
        );
        assert_eq!(streamer.dispatch_feedback(2).unwrap(), 0);
        assert_eq!(streamer.pending.len(), 2);
        assert_eq!(receiver.recv().unwrap().key, blocker);
        streamer.submit_feedback(32, vec![TileRequest::visible(live, 10, 32)], false);
        assert_eq!(streamer.dispatch_feedback(33).unwrap(), 1);
        assert_eq!(receiver.recv().unwrap().key, live);
        assert!(streamer.pending.is_empty());
        assert!(!streamer.in_flight.contains_key(&old));
    }

    #[test]
    fn in_flight_duplicate_is_not_resent() {
        let (mut streamer, receiver) = VirtualTextureStreamer::new_for_test(2);
        let key = TileKey::new(1, 0, 0, 0);
        streamer.submit_feedback(0, vec![TileRequest::prefetch(key, 1, 0)], false);
        assert_eq!(streamer.dispatch_feedback(1).unwrap(), 1);
        assert_eq!(receiver.recv().unwrap().key, key);
        streamer.submit_feedback(1, vec![TileRequest::visible(key, 100, 1)], false);
        assert_eq!(streamer.dispatch_feedback(2).unwrap(), 0);
        assert!(receiver.try_recv().is_err());
        assert_eq!(
            streamer.in_flight.get(&key).unwrap().kind,
            RequestKind::Visible
        );
    }

    #[test]
    fn stale_completion_is_dropped_but_reaffirmed_tile_is_uploaded() {
        let (request_sender, _requests) = mpsc::sync_channel(4);
        let (completion_sender, completed) = mpsc::sync_channel(4);
        let layer = valid_layer();
        let mut streamer = VirtualTextureStreamer::from_channels(
            request_sender, completed, vec![layer], u64::MAX);
        let old = TileKey::new(layer.id, 0, 0, 0);
        let live = TileKey::new(layer.id, 0, 1, 0);
        streamer.submit_feedback(0, vec![
            TileRequest::visible(old, 10, 0), TileRequest::visible(live, 10, 0)], false);
        assert_eq!(streamer.dispatch_feedback(1).unwrap(), 2);
        streamer.submit_feedback(32, vec![TileRequest::visible(live, 10, 32)], false);
        assert_eq!(streamer.dispatch_feedback(33).unwrap(), 0);
        for key in [old, live] {
            completion_sender.send((key, Ok(UploadJob {
                key, parent: None, layer, format: PixelFormat::Bc7,
                content_hash: 1, payload: vec![1],
            }))).unwrap();
        }
        assert_eq!(streamer.poll_one().unwrap().unwrap().key, live);
        assert!(streamer.poll_one().unwrap().is_none());
        assert!(streamer.in_flight.is_empty());
    }

    #[test]
    fn shared_parent_pin_releases_after_both_crossfades() {
        let (mut streamer, _receiver) = VirtualTextureStreamer::new_for_test(1);
        let now = Instant::now();
        let parent = TileKey::new(1, 8, 0, 0);
        streamer
            .residency_mut()
            .mark_resident(parent, 1, now, false)
            .unwrap();
        let layer = valid_layer();
        let first = UploadJob {
            key: TileKey::new(1, 7, 0, 0),
            parent: Some(parent),
            layer,
            format: PixelFormat::Bc7,
            content_hash: 1,
            payload: vec![1],
        };
        let second = UploadJob {
            key: TileKey::new(1, 7, 1, 0),
            parent: Some(parent),
            layer,
            format: PixelFormat::Bc7,
            content_hash: 2,
            payload: vec![1],
        };
        streamer.mark_uploaded(&first, 10, now).unwrap();
        streamer.mark_uploaded(&second, 10, now).unwrap();
        assert!(streamer.residency().record(parent).unwrap().pinned);
        assert_eq!(streamer.crossfades(16).count(), 0);
        assert!(!streamer.residency().record(parent).unwrap().pinned);
    }

    #[test]
    fn pending_queue_keeps_visible_upgrade_and_dedupes_inflight() {
        let key = TileKey::new(2, 0, 0, 0);
        let mut queue = TileRequestQueue::default();
        queue.enqueue(TileRequest::prefetch(key, 1, 10));
        queue.enqueue(TileRequest::visible(key, 50, 11));
        let requests = queue.drain_prioritized();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].kind, crate::earthvt::RequestKind::Visible);
        assert_eq!(requests[0].priority, 50);
    }
}
