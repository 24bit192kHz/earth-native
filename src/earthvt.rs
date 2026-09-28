#![allow(dead_code)] // The virtual-texture API is wired by the staged streamer.

//! Versioned, memory-map-friendly storage for Earth virtual textures.
//!
//! Every integer in an `.earthvt` file is little endian. The directory is
//! deliberately fixed width so a runtime can borrow a memory mapping and copy
//! compressed tile payloads directly into a Vulkan staging buffer. There is no
//! image decoding in this format.
//!
//! Version 1 layout:
//!
//! ```text
//! header                 96 bytes
//! layer directory        layer_count * 48 bytes
//! tile index              index_count * 48 bytes
//! alignment padding       optional, up to the payload region
//! tile payloads           fixed-size GPU-ready blocks, 16-byte aligned
//! ```
//!
//! Tiles have a 256x256 interior and a four-pixel gutter on all sides. Each
//! stored tile is therefore 264x264 pixels. Edge tiles are padded by the asset
//! baker, which keeps every tile's block-compressed byte count identical.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    ops::Range,
    time::{Duration, Instant},
};

pub const MAGIC: [u8; 8] = *b"EARTHVT\0";
pub const VERSION: u16 = 1;
pub const HEADER_BYTES: usize = 96;
pub const LAYER_ENTRY_BYTES: usize = 48;
pub const INDEX_ENTRY_BYTES: usize = 48;

pub const TILE_SIZE: u16 = 256;
pub const GUTTER_SIZE: u16 = 4;
pub const PADDED_TILE_SIZE: u32 = TILE_SIZE as u32 + (GUTTER_SIZE as u32 * 2);
pub const PAYLOAD_ALIGNMENT: u64 = 16;
pub const MAX_LAYERS: usize = 64;

pub const HEADER_FLAGS_KNOWN: u32 = 0;
pub const TILE_FLAGS_KNOWN: u32 = 0;

/// The color-space and addressing metadata the sampler needs for a layer.
pub const LAYER_FLAG_SRGB: u32 = 1 << 0;
pub const LAYER_FLAG_WRAP_X: u32 = 1 << 1;
pub const LAYER_FLAG_CLAMP_Y: u32 = 1 << 2;
pub const LAYER_FLAGS_KNOWN: u32 = LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y;

/// The normal two-second eviction grace period prevents zoom oscillation.
pub const DEFAULT_EVICTION_HYSTERESIS: Duration = Duration::from_secs(2);

/// GPU format of an already encoded tile payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum PixelFormat {
    Bc1 = 1,
    Bc7 = 2,
    Bc5 = 3,
    Bc4 = 4,
    R16 = 5,
}

impl PixelFormat {
    pub const fn raw(self) -> u16 {
        self as u16
    }

    pub fn from_raw(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Bc1),
            2 => Some(Self::Bc7),
            3 => Some(Self::Bc5),
            4 => Some(Self::Bc4),
            5 => Some(Self::R16),
            _ => None,
        }
    }

    /// Exact byte count for one padded 264x264 tile.
    pub const fn encoded_tile_bytes(self) -> u64 {
        let extent = PADDED_TILE_SIZE as u64;
        match self {
            Self::Bc1 | Self::Bc4 => (extent / 4) * (extent / 4) * 8,
            Self::Bc7 | Self::Bc5 => (extent / 4) * (extent / 4) * 16,
            Self::R16 => extent * extent * 2,
        }
    }

    pub const fn is_block_compressed(self) -> bool {
        !matches!(self, Self::R16)
    }
}

/// Semantic channel names emitted by the offline material baker.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum TextureChannel {
    DayColor = 1,
    SurfaceNormal = 2,
    NightEmission = 3,
    WaterMask = 4,
    CloudDensity = 5,
    CloudNormal = 6,
    Height = 7,
    StarPanorama = 8,
    CityCloudGlow = 9,
    AuroraMask = 10,
    AuroraColor = 11,
    LightningMask = 12,
}

impl TextureChannel {
    pub const fn raw(self) -> u16 {
        self as u16
    }

    pub fn from_raw(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::DayColor),
            2 => Some(Self::SurfaceNormal),
            3 => Some(Self::NightEmission),
            4 => Some(Self::WaterMask),
            5 => Some(Self::CloudDensity),
            6 => Some(Self::CloudNormal),
            7 => Some(Self::Height),
            8 => Some(Self::StarPanorama),
            9 => Some(Self::CityCloudGlow),
            10 => Some(Self::AuroraMask),
            11 => Some(Self::AuroraColor),
            12 => Some(Self::LightningMask),
            _ => None,
        }
    }

    /// Restrict channel encodings to the formats selected by the renderer.
    pub const fn accepts_format(self, format: PixelFormat) -> bool {
        match self {
            Self::DayColor | Self::CityCloudGlow | Self::AuroraColor => {
                matches!(format, PixelFormat::Bc7)
            }
            // City lights are grayscale: BC4, or BC7 for coloured sources.
            Self::NightEmission => matches!(format, PixelFormat::Bc7 | PixelFormat::Bc4),
            Self::SurfaceNormal | Self::CloudNormal => matches!(format, PixelFormat::Bc5),
            Self::WaterMask | Self::CloudDensity | Self::AuroraMask | Self::LightningMask => {
                matches!(format, PixelFormat::Bc4)
            }
            Self::Height => matches!(format, PixelFormat::R16),
            Self::StarPanorama => matches!(format, PixelFormat::Bc1 | PixelFormat::Bc7),
        }
    }
}

/// Parsed contents of the fixed-size file header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Header {
    pub version: u16,
    pub flags: u32,
    pub tile_size: u16,
    pub gutter_size: u16,
    pub layer_count: u16,
    pub index_count: u32,
    pub layer_table_offset: u64,
    pub index_offset: u64,
    pub payload_offset: u64,
    pub file_bytes: u64,
}

/// One texture channel and the contiguous range of its tile index entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerDescriptor {
    pub id: u16,
    pub channel: TextureChannel,
    pub format: PixelFormat,
    pub mip_count: u16,
    pub base_width: u32,
    pub base_height: u32,
    pub first_index: u32,
    pub index_count: u32,
    pub flags: u32,
}

impl LayerDescriptor {
    /// Pixel dimensions for a mip, where mip zero is the full source image.
    pub fn mip_dimensions(self, mip: u16) -> Option<(u32, u32)> {
        if mip >= self.mip_count || self.base_width == 0 || self.base_height == 0 {
            return None;
        }
        Some((
            mip_dimension(self.base_width, mip),
            mip_dimension(self.base_height, mip),
        ))
    }

    /// Number of 256x256 interiors required for a mip.
    pub fn tile_grid(self, mip: u16) -> Option<(u32, u32)> {
        let (width, height) = self.mip_dimensions(mip)?;
        Some((
            ceil_div(width, TILE_SIZE as u32),
            ceil_div(height, TILE_SIZE as u32),
        ))
    }

    pub fn contains_key(self, key: TileKey) -> bool {
        if key.layer != self.id {
            return false;
        }
        let Some((tiles_x, tiles_y)) = self.tile_grid(key.mip) else {
            return false;
        };
        key.x < tiles_x && key.y < tiles_y
    }

    pub fn parent_key(self, key: TileKey) -> Option<TileKey> {
        if !self.contains_key(key) {
            return None;
        }
        let parent = key.parent()?;
        self.contains_key(parent).then_some(parent)
    }

    /// The total number of entries in a complete, non-sparse mip pyramid.
    pub fn expected_tile_count(self) -> Option<u32> {
        let mut total = 0_u64;
        for mip in 0..self.mip_count {
            let (tiles_x, tiles_y) = self.tile_grid(mip)?;
            let count = u64::from(tiles_x).checked_mul(u64::from(tiles_y))?;
            total = total.checked_add(count)?;
        }
        u32::try_from(total).ok()
    }
}

/// The logical location of one virtual texture tile.
///
/// Keys sort in canonical on-disk order: layer, mip, row, then column.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TileKey {
    pub layer: u16,
    pub mip: u16,
    pub x: u32,
    pub y: u32,
}

impl TileKey {
    pub const fn new(layer: u16, mip: u16, x: u32, y: u32) -> Self {
        Self { layer, mip, x, y }
    }

    /// Return the next coarser tile, without checking a layer's mip bounds.
    pub fn parent(self) -> Option<Self> {
        Some(Self {
            layer: self.layer,
            mip: self.mip.checked_add(1)?,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// Return an ancestor after `levels` coarser mip transitions.
    pub fn ancestor(self, levels: u16) -> Option<Self> {
        let mut key = self;
        for _ in 0..levels {
            key = key.parent()?;
        }
        Some(key)
    }
}

impl Ord for TileKey {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.layer, self.mip, self.y, self.x).cmp(&(other.layer, other.mip, other.y, other.x))
    }
}

impl PartialOrd for TileKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// An index entry. `payload_offset` is absolute from the beginning of the file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileIndexEntry {
    pub key: TileKey,
    pub flags: u32,
    pub payload_offset: u64,
    pub payload_bytes: u64,
    /// Opaque baker-provided content fingerprint. A zero value means unspecified.
    pub content_hash: u32,
}

/// Borrowed payload paired with its validated index entry.
#[derive(Clone, Copy, Debug)]
pub struct TileRef<'a> {
    pub entry: TileIndexEntry,
    pub payload: &'a [u8],
}

/// A resident tile selected for a requested tile or one of its coarser parents.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedTile<'a> {
    pub requested: TileKey,
    pub resolved: TileKey,
    pub levels_up: u16,
    pub tile: TileRef<'a>,
}

/// Errors returned while validating an `.earthvt` byte slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EarthVtError {
    TooSmall {
        needed: usize,
        available: usize,
    },
    InvalidMagic,
    UnsupportedVersion(u16),
    UnsupportedHeaderFlags(u32),
    UnknownPixelFormat(u16),
    UnknownTextureChannel(u16),
    InvalidHeader(&'static str),
    InvalidLayer {
        layer: usize,
        reason: &'static str,
    },
    InvalidIndex {
        index: u32,
        reason: &'static str,
    },
    Overflow(&'static str),
    OutOfBounds {
        section: &'static str,
        offset: u64,
        bytes: u64,
    },
}

impl fmt::Display for EarthVtError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooSmall { needed, available } => {
                write!(
                    formatter,
                    "file is too small: need {needed} bytes, have {available}"
                )
            }
            Self::InvalidMagic => formatter.write_str("not an EARTHVT container"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported EARTHVT version {version}")
            }
            Self::UnsupportedHeaderFlags(flags) => {
                write!(formatter, "unsupported EARTHVT header flags 0x{flags:08x}")
            }
            Self::UnknownPixelFormat(format) => {
                write!(formatter, "unknown EARTHVT pixel format {format}")
            }
            Self::UnknownTextureChannel(channel) => {
                write!(formatter, "unknown EARTHVT texture channel {channel}")
            }
            Self::InvalidHeader(reason) => write!(formatter, "invalid EARTHVT header: {reason}"),
            Self::InvalidLayer { layer, reason } => {
                write!(formatter, "invalid EARTHVT layer {layer}: {reason}")
            }
            Self::InvalidIndex { index, reason } => {
                write!(formatter, "invalid EARTHVT tile index {index}: {reason}")
            }
            Self::Overflow(section) => write!(formatter, "EARTHVT {section} arithmetic overflow"),
            Self::OutOfBounds {
                section,
                offset,
                bytes,
            } => write!(
                formatter,
                "EARTHVT {section} range at {offset} for {bytes} bytes is outside the file"
            ),
        }
    }
}

impl Error for EarthVtError {}

/// A parsed container. `bytes` may be an mmap-backed slice owned by the caller.
///
/// The parser only allocates the small layer directory. The tile index and all
/// payloads remain borrowed from `bytes` and are decoded on demand.
pub struct EarthVt<'a> {
    bytes: &'a [u8],
    header: Header,
    layers: Vec<LayerDescriptor>,
    index_range: Range<usize>,
}

impl<'a> EarthVt<'a> {
    /// Parse and fully validate a version 1 container without decoding payloads.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, EarthVtError> {
        let header = parse_header(bytes)?;
        let layer_bytes = u64::from(header.layer_count)
            .checked_mul(LAYER_ENTRY_BYTES as u64)
            .ok_or(EarthVtError::Overflow("layer directory"))?;
        let index_bytes = u64::from(header.index_count)
            .checked_mul(INDEX_ENTRY_BYTES as u64)
            .ok_or(EarthVtError::Overflow("tile index"))?;
        let layer_range = section_range(
            bytes,
            header.layer_table_offset,
            layer_bytes,
            "layer directory",
        )?;
        let index_range = section_range(bytes, header.index_offset, index_bytes, "tile index")?;

        validate_layout(&header, &layer_range, &index_range)?;

        let mut layers = Vec::with_capacity(usize::from(header.layer_count));
        let mut expected_first_index = 0_u32;
        for layer_number in 0..usize::from(header.layer_count) {
            let offset = layer_range.start + layer_number * LAYER_ENTRY_BYTES;
            let layer = parse_layer(bytes, offset)?;
            validate_layer(
                &layer,
                layer_number,
                layers.last().copied(),
                expected_first_index,
            )?;
            expected_first_index = expected_first_index
                .checked_add(layer.index_count)
                .ok_or(EarthVtError::Overflow("layer index range"))?;
            if layers.iter().any(|other| other.channel == layer.channel) {
                return Err(EarthVtError::InvalidLayer {
                    layer: layer_number,
                    reason: "texture channels must be unique",
                });
            }
            layers.push(layer);
        }
        if expected_first_index != header.index_count {
            return Err(EarthVtError::InvalidHeader(
                "layer index ranges do not cover the tile index",
            ));
        }

        for layer in layers.iter().copied() {
            validate_layer_index(bytes, &header, &index_range, layer)?;
        }

        Ok(Self {
            bytes,
            header,
            layers,
            index_range,
        })
    }

    pub fn header(&self) -> Header {
        self.header
    }

    pub fn layers(&self) -> &[LayerDescriptor] {
        &self.layers
    }

    pub fn layer(&self, id: u16) -> Option<LayerDescriptor> {
        self.layers
            .binary_search_by_key(&id, |layer| layer.id)
            .ok()
            .map(|index| self.layers[index])
    }

    pub fn layer_for_channel(&self, channel: TextureChannel) -> Option<LayerDescriptor> {
        self.layers
            .iter()
            .copied()
            .find(|layer| layer.channel == channel)
    }

    /// The file tail containing only aligned, direct-upload payloads.
    pub fn payload_region(&self) -> &'a [u8] {
        let start = usize::try_from(self.header.payload_offset)
            .expect("validated payload offset fits usize");
        &self.bytes[start..]
    }

    pub fn index_entry(&self, index: u32) -> Option<TileIndexEntry> {
        if index >= self.header.index_count {
            return None;
        }
        Some(self.index_entry_unchecked(index))
    }

    pub fn entries(&self) -> impl Iterator<Item = TileIndexEntry> + '_ {
        (0..self.header.index_count).map(|index| self.index_entry_unchecked(index))
    }

    /// Find a tile in the on-disk index. Complete mip pyramids guarantee that a
    /// valid key has an entry even when its GPU atlas slot is not resident.
    pub fn tile_entry(&self, key: TileKey) -> Option<TileIndexEntry> {
        let layer = self.layer(key.layer)?;
        if !layer.contains_key(key) {
            return None;
        }

        let mut low = layer.first_index;
        let mut high = layer.first_index.checked_add(layer.index_count)?;
        while low < high {
            let middle = low + (high - low) / 2;
            let candidate = self.index_entry_unchecked(middle);
            match candidate.key.cmp(&key) {
                Ordering::Less => low = middle + 1,
                Ordering::Greater => high = middle,
                Ordering::Equal => return Some(candidate),
            }
        }
        None
    }

    pub fn tile(&self, key: TileKey) -> Option<TileRef<'a>> {
        let entry = self.tile_entry(key)?;
        let payload = self.payload_for_entry(entry);
        Some(TileRef { entry, payload })
    }

    pub fn parent_key(&self, key: TileKey) -> Option<TileKey> {
        self.layer(key.layer)?.parent_key(key)
    }

    /// Resolve to the finest resident tile or a resident parent mip fallback.
    pub fn resolve_resident(
        &self,
        requested: TileKey,
        residency: &ResidencyTracker,
    ) -> Option<ResolvedTile<'a>> {
        let mut candidate = requested;
        let mut levels_up = 0_u16;
        loop {
            if residency.is_resident(candidate) {
                let tile = self.tile(candidate)?;
                return Some(ResolvedTile {
                    requested,
                    resolved: candidate,
                    levels_up,
                    tile,
                });
            }
            candidate = self.parent_key(candidate)?;
            levels_up = levels_up.checked_add(1)?;
        }
    }

    fn index_entry_unchecked(&self, index: u32) -> TileIndexEntry {
        let offset = self.index_range.start
            + usize::try_from(index).expect("u32 fits usize") * INDEX_ENTRY_BYTES;
        parse_index_entry(self.bytes, offset)
    }

    fn payload_for_entry(&self, entry: TileIndexEntry) -> &'a [u8] {
        let start =
            usize::try_from(entry.payload_offset).expect("validated payload offset fits usize");
        let bytes =
            usize::try_from(entry.payload_bytes).expect("validated payload byte count fits usize");
        let end = start
            .checked_add(bytes)
            .expect("validated payload range does not overflow");
        &self.bytes[start..end]
    }
}

/// Request priority class. Visible feedback always outranks prefetch feedback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum RequestKind {
    Prefetch = 0,
    Visible = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileRequest {
    pub key: TileKey,
    /// Larger values are serviced first, normally projected pixel coverage.
    pub priority: u32,
    pub kind: RequestKind,
    pub requested_frame: u64,
}

impl TileRequest {
    pub const fn visible(key: TileKey, priority: u32, requested_frame: u64) -> Self {
        Self {
            key,
            priority,
            kind: RequestKind::Visible,
            requested_frame,
        }
    }

    pub const fn prefetch(key: TileKey, priority: u32, requested_frame: u64) -> Self {
        Self {
            key,
            priority,
            kind: RequestKind::Prefetch,
            requested_frame,
        }
    }

    /// Merge a newer observation of the same tile into this request.
    pub fn merge(&mut self, request: Self) {
        debug_assert_eq!(self.key, request.key);
        self.priority = self.priority.max(request.priority);
        self.kind = self.kind.max(request.kind);
        self.requested_frame = self.requested_frame.max(request.requested_frame);
    }
}

/// Deduplicates one-frame-late GPU feedback without performing disk I/O.
#[derive(Default)]
pub struct TileRequestQueue {
    pending: BTreeMap<TileKey, TileRequest>,
}

impl TileRequestQueue {
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Add or merge a request. A visible request upgrades an earlier prefetch.
    pub fn enqueue(&mut self, request: TileRequest) {
        match self.pending.get_mut(&request.key) {
            Some(existing) => existing.merge(request),
            None => {
                self.pending.insert(request.key, request);
            }
        }
    }
    /// Drop requests older than `max_age` frames that the newest feedback
    /// batch no longer affirms. Reaffirmed keys (the settled view) are kept.
    pub fn retain_fresh(&mut self, current_frame: u64, max_age: u64, visible: &BTreeSet<TileKey>) {
        self.pending.retain(|key, request| {
            visible.contains(key)
                || current_frame.saturating_sub(request.requested_frame) <= max_age
        });
    }

    /// Consume all requests in stable service order. The caller owns I/O policy.
    pub fn drain_prioritized(&mut self) -> Vec<TileRequest> {
        let mut requests = std::mem::take(&mut self.pending)
            .into_values()
            .collect::<Vec<_>>();
        requests.sort_by(|left, right| {
            right
                .kind
                .cmp(&left.kind)
                .then_with(|| right.priority.cmp(&left.priority))
                .then_with(|| left.key.cmp(&right.key))
        });
        requests
    }
}

#[derive(Clone, Debug)]
pub struct ResidencyRecord {
    pub bytes: u64,
    pub last_used: Instant,
    pub pinned: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionPlan {
    Ready,
    Evict {
        keys: Vec<TileKey>,
        reclaimed_bytes: u64,
    },
    Deferred {
        needed_bytes: u64,
        reclaimable_bytes: u64,
    },
    Oversized {
        requested_bytes: u64,
        budget_bytes: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidencyError {
    AlreadyResident(TileKey),
    BudgetExceeded {
        requested_bytes: u64,
        available_bytes: u64,
    },
    AccountingOverflow,
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyResident(key) => write!(formatter, "tile {key:?} is already resident"),
            Self::BudgetExceeded {
                requested_bytes,
                available_bytes,
            } => write!(
                formatter,
                "cannot admit {requested_bytes} bytes with only {available_bytes} bytes available"
            ),
            Self::AccountingOverflow => formatter.write_str("residency byte accounting overflow"),
        }
    }
}

impl Error for ResidencyError {}

/// Fixed-budget LRU metadata. It models GPU residency only and owns no images,
/// threads, files, or Vulkan resources.
pub struct ResidencyTracker {
    budget_bytes: u64,
    resident_bytes: u64,
    eviction_hysteresis: Duration,
    records: BTreeMap<TileKey, ResidencyRecord>,
}

impl ResidencyTracker {
    pub fn new(budget_bytes: u64) -> Self {
        Self::with_hysteresis(budget_bytes, DEFAULT_EVICTION_HYSTERESIS)
    }

    pub fn with_hysteresis(budget_bytes: u64, eviction_hysteresis: Duration) -> Self {
        Self {
            budget_bytes,
            resident_bytes: 0,
            eviction_hysteresis,
            records: BTreeMap::new(),
        }
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn eviction_hysteresis(&self) -> Duration {
        self.eviction_hysteresis
    }

    pub fn is_resident(&self, key: TileKey) -> bool {
        self.records.contains_key(&key)
    }

    pub fn record(&self, key: TileKey) -> Option<&ResidencyRecord> {
        self.records.get(&key)
    }

    pub fn touch(&mut self, key: TileKey, now: Instant) -> bool {
        let Some(record) = self.records.get_mut(&key) else {
            return false;
        };
        record.last_used = now;
        true
    }

    pub fn set_pinned(&mut self, key: TileKey, pinned: bool) -> bool {
        let Some(record) = self.records.get_mut(&key) else {
            return false;
        };
        record.pinned = pinned;
        true
    }

    /// Plan evictions before scheduling an upload. Recent and pinned tiles are
    /// intentionally ineligible, so a caller can defer instead of causing pop-in.
    pub fn plan_admission(&self, requested_bytes: u64, now: Instant) -> AdmissionPlan {
        if requested_bytes > self.budget_bytes {
            return AdmissionPlan::Oversized {
                requested_bytes,
                budget_bytes: self.budget_bytes,
            };
        }

        let available = self.budget_bytes.saturating_sub(self.resident_bytes);
        if requested_bytes <= available {
            return AdmissionPlan::Ready;
        }
        let needed_bytes = requested_bytes - available;
        let mut candidates = self
            .records
            .iter()
            .filter_map(|(key, record)| {
                let age = now.checked_duration_since(record.last_used)?;
                (!record.pinned && age >= self.eviction_hysteresis).then_some((*key, record))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_key, left), (right_key, right)| {
            left.last_used
                .cmp(&right.last_used)
                .then_with(|| left_key.cmp(right_key))
        });

        let mut reclaimed_bytes = 0_u64;
        let mut keys = Vec::new();
        for (key, record) in candidates {
            reclaimed_bytes = reclaimed_bytes.saturating_add(record.bytes);
            keys.push(key);
            if reclaimed_bytes >= needed_bytes {
                return AdmissionPlan::Evict {
                    keys,
                    reclaimed_bytes,
                };
            }
        }
        AdmissionPlan::Deferred {
            needed_bytes,
            reclaimable_bytes: reclaimed_bytes,
        }
    }

    /// Commit a completed GPU upload after any planned evictions have been made.
    pub fn mark_resident(
        &mut self,
        key: TileKey,
        bytes: u64,
        now: Instant,
        pinned: bool,
    ) -> Result<(), ResidencyError> {
        if self.records.contains_key(&key) {
            return Err(ResidencyError::AlreadyResident(key));
        }
        let available = self.budget_bytes.saturating_sub(self.resident_bytes);
        if bytes > available {
            return Err(ResidencyError::BudgetExceeded {
                requested_bytes: bytes,
                available_bytes: available,
            });
        }
        self.resident_bytes = self
            .resident_bytes
            .checked_add(bytes)
            .ok_or(ResidencyError::AccountingOverflow)?;
        self.records.insert(
            key,
            ResidencyRecord {
                bytes,
                last_used: now,
                pinned,
            },
        );
        Ok(())
    }

    pub fn remove(&mut self, key: TileKey) -> Option<ResidencyRecord> {
        let record = self.records.remove(&key)?;
        self.resident_bytes = self.resident_bytes.saturating_sub(record.bytes);
        Some(record)
    }
}

fn parse_header(bytes: &[u8]) -> Result<Header, EarthVtError> {
    if bytes.len() < HEADER_BYTES {
        return Err(EarthVtError::TooSmall {
            needed: HEADER_BYTES,
            available: bytes.len(),
        });
    }
    if bytes[..8] != MAGIC {
        return Err(EarthVtError::InvalidMagic);
    }
    let version = read_u16(bytes, 8);
    if version != VERSION {
        return Err(EarthVtError::UnsupportedVersion(version));
    }
    if usize::from(read_u16(bytes, 10)) != HEADER_BYTES {
        return Err(EarthVtError::InvalidHeader("unexpected header size"));
    }
    let flags = read_u32(bytes, 12);
    if flags & !HEADER_FLAGS_KNOWN != 0 {
        return Err(EarthVtError::UnsupportedHeaderFlags(flags));
    }
    let tile_size = read_u16(bytes, 16);
    let gutter_size = read_u16(bytes, 18);
    if tile_size != TILE_SIZE || gutter_size != GUTTER_SIZE {
        return Err(EarthVtError::InvalidHeader(
            "version 1 requires 256-pixel tiles with 4-pixel gutters",
        ));
    }
    let layer_count = read_u16(bytes, 20);
    if layer_count == 0 || usize::from(layer_count) > MAX_LAYERS {
        return Err(EarthVtError::InvalidHeader("invalid layer count"));
    }
    if usize::from(read_u16(bytes, 22)) != LAYER_ENTRY_BYTES {
        return Err(EarthVtError::InvalidHeader("unexpected layer entry size"));
    }
    if usize::from(read_u16(bytes, 24)) != INDEX_ENTRY_BYTES {
        return Err(EarthVtError::InvalidHeader(
            "unexpected tile index entry size",
        ));
    }
    if read_u16(bytes, 26) != 0
        || read_u64(bytes, 64) != 0
        || read_u64(bytes, 72) != 0
        || read_u64(bytes, 80) != 0
        || read_u64(bytes, 88) != 0
    {
        return Err(EarthVtError::InvalidHeader("reserved bytes are nonzero"));
    }
    let index_count = read_u32(bytes, 28);
    if index_count == 0 {
        return Err(EarthVtError::InvalidHeader("tile index is empty"));
    }
    let file_bytes = read_u64(bytes, 56);
    if file_bytes != u64::try_from(bytes.len()).map_err(|_| EarthVtError::Overflow("file size"))? {
        return Err(EarthVtError::InvalidHeader(
            "declared file size does not match mapped byte length",
        ));
    }
    Ok(Header {
        version,
        flags,
        tile_size,
        gutter_size,
        layer_count,
        index_count,
        layer_table_offset: read_u64(bytes, 32),
        index_offset: read_u64(bytes, 40),
        payload_offset: read_u64(bytes, 48),
        file_bytes,
    })
}

fn parse_layer(bytes: &[u8], offset: usize) -> Result<LayerDescriptor, EarthVtError> {
    let channel_raw = read_u16(bytes, offset + 2);
    let format_raw = read_u16(bytes, offset + 4);
    let channel = TextureChannel::from_raw(channel_raw)
        .ok_or(EarthVtError::UnknownTextureChannel(channel_raw))?;
    let format =
        PixelFormat::from_raw(format_raw).ok_or(EarthVtError::UnknownPixelFormat(format_raw))?;
    Ok(LayerDescriptor {
        id: read_u16(bytes, offset),
        channel,
        format,
        mip_count: read_u16(bytes, offset + 6),
        base_width: read_u32(bytes, offset + 8),
        base_height: read_u32(bytes, offset + 12),
        first_index: read_u32(bytes, offset + 16),
        index_count: read_u32(bytes, offset + 20),
        flags: read_u32(bytes, offset + 24),
    })
}

fn parse_index_entry(bytes: &[u8], offset: usize) -> TileIndexEntry {
    TileIndexEntry {
        key: TileKey {
            layer: read_u16(bytes, offset),
            mip: read_u16(bytes, offset + 2),
            x: read_u32(bytes, offset + 4),
            y: read_u32(bytes, offset + 8),
        },
        flags: read_u32(bytes, offset + 12),
        payload_offset: read_u64(bytes, offset + 16),
        payload_bytes: read_u64(bytes, offset + 24),
        content_hash: read_u32(bytes, offset + 32),
    }
}

fn validate_layout(
    header: &Header,
    layer_range: &Range<usize>,
    index_range: &Range<usize>,
) -> Result<(), EarthVtError> {
    if header.layer_table_offset % 8 != 0 || header.index_offset % 8 != 0 {
        return Err(EarthVtError::InvalidHeader(
            "layer and index tables must be eight-byte aligned",
        ));
    }
    if header.payload_offset % PAYLOAD_ALIGNMENT != 0 {
        return Err(EarthVtError::InvalidHeader(
            "payload region must be sixteen-byte aligned",
        ));
    }
    if layer_range.start < HEADER_BYTES || index_range.start < HEADER_BYTES {
        return Err(EarthVtError::InvalidHeader("metadata overlaps the header"));
    }
    let payload_start = usize::try_from(header.payload_offset)
        .map_err(|_| EarthVtError::Overflow("payload offset"))?;
    if layer_range.end > payload_start || index_range.end > payload_start {
        return Err(EarthVtError::InvalidHeader(
            "metadata overlaps the payload region",
        ));
    }
    if ranges_overlap(layer_range, index_range) {
        return Err(EarthVtError::InvalidHeader(
            "layer and index tables overlap",
        ));
    }
    Ok(())
}

fn validate_layer(
    layer: &LayerDescriptor,
    layer_number: usize,
    previous: Option<LayerDescriptor>,
    expected_first_index: u32,
) -> Result<(), EarthVtError> {
    if let Some(previous) = previous {
        if layer.id <= previous.id {
            return Err(EarthVtError::InvalidLayer {
                layer: layer_number,
                reason: "layer ids must be strictly ascending",
            });
        }
    }
    if layer.base_width == 0 || layer.base_height == 0 {
        return Err(EarthVtError::InvalidLayer {
            layer: layer_number,
            reason: "base dimensions must be nonzero",
        });
    }
    let expected_mips =
        full_mip_count(layer.base_width, layer.base_height).ok_or(EarthVtError::InvalidLayer {
            layer: layer_number,
            reason: "invalid mip dimensions",
        })?;
    if layer.mip_count != expected_mips {
        return Err(EarthVtError::InvalidLayer {
            layer: layer_number,
            reason: "mip count is not a complete pyramid",
        });
    }
    if !layer.channel.accepts_format(layer.format) {
        return Err(EarthVtError::InvalidLayer {
            layer: layer_number,
            reason: "channel uses an unsupported pixel format",
        });
    }
    if layer.flags & !LAYER_FLAGS_KNOWN != 0 {
        return Err(EarthVtError::InvalidLayer {
            layer: layer_number,
            reason: "layer has unknown flags",
        });
    }
    if layer.first_index != expected_first_index {
        return Err(EarthVtError::InvalidLayer {
            layer: layer_number,
            reason: "layer index range is not contiguous",
        });
    }
    if layer.index_count
        != layer
            .expected_tile_count()
            .ok_or(EarthVtError::InvalidLayer {
                layer: layer_number,
                reason: "tile count overflows",
            })?
    {
        return Err(EarthVtError::InvalidLayer {
            layer: layer_number,
            reason: "index count does not describe a complete mip pyramid",
        });
    }
    Ok(())
}

fn validate_layer_index(
    bytes: &[u8],
    header: &Header,
    index_range: &Range<usize>,
    layer: LayerDescriptor,
) -> Result<(), EarthVtError> {
    let mut previous = None;
    let end = layer
        .first_index
        .checked_add(layer.index_count)
        .ok_or(EarthVtError::Overflow("layer index range"))?;
    for index in layer.first_index..end {
        let offset =
            index_range.start + usize::try_from(index).expect("u32 fits usize") * INDEX_ENTRY_BYTES;
        let entry = parse_index_entry(bytes, offset);
        if entry.key.layer != layer.id {
            return Err(EarthVtError::InvalidIndex {
                index,
                reason: "entry belongs to a different layer",
            });
        }
        if !layer.contains_key(entry.key) {
            return Err(EarthVtError::InvalidIndex {
                index,
                reason: "tile coordinates are outside its mip grid",
            });
        }
        if let Some(previous) = previous {
            if entry.key <= previous {
                return Err(EarthVtError::InvalidIndex {
                    index,
                    reason: "entries are not in canonical order",
                });
            }
        }
        previous = Some(entry.key);
        if entry.flags & !TILE_FLAGS_KNOWN != 0 {
            return Err(EarthVtError::InvalidIndex {
                index,
                reason: "tile has unknown flags",
            });
        }
        if read_u32(bytes, offset + 36) != 0 || read_u64(bytes, offset + 40) != 0 {
            return Err(EarthVtError::InvalidIndex {
                index,
                reason: "reserved bytes are nonzero",
            });
        }
        if entry.payload_offset < header.payload_offset {
            return Err(EarthVtError::InvalidIndex {
                index,
                reason: "payload precedes the payload region",
            });
        }
        if entry.payload_offset % PAYLOAD_ALIGNMENT != 0 {
            return Err(EarthVtError::InvalidIndex {
                index,
                reason: "payload offset is not sixteen-byte aligned",
            });
        }
        if entry.payload_bytes != layer.format.encoded_tile_bytes() {
            return Err(EarthVtError::InvalidIndex {
                index,
                reason: "payload byte length does not match the fixed tile format",
            });
        }
        let _ = section_range(
            bytes,
            entry.payload_offset,
            entry.payload_bytes,
            "tile payload",
        )?;
    }
    Ok(())
}

pub fn full_mip_count(width: u32, height: u32) -> Option<u16> {
    if width == 0 || height == 0 {
        return None;
    }
    let mut largest = width.max(height);
    let mut count = 1_u16;
    while largest > 1 {
        largest = largest / 2 + largest % 2;
        count = count.checked_add(1)?;
    }
    Some(count)
}

fn mip_dimension(base: u32, mip: u16) -> u32 {
    if mip >= 32 {
        return 1;
    }
    let divisor = 1_u64 << u32::from(mip);
    u64::from(base).div_ceil(divisor) as u32
}

fn ceil_div(value: u32, divisor: u32) -> u32 {
    value / divisor + u32::from(value % divisor != 0)
}

fn section_range(
    bytes: &[u8],
    offset: u64,
    length: u64,
    section: &'static str,
) -> Result<Range<usize>, EarthVtError> {
    let end = offset
        .checked_add(length)
        .ok_or(EarthVtError::Overflow(section))?;
    let file_bytes = u64::try_from(bytes.len()).map_err(|_| EarthVtError::Overflow("file size"))?;
    if end > file_bytes {
        return Err(EarthVtError::OutOfBounds {
            section,
            offset,
            bytes: length,
        });
    }
    let start = usize::try_from(offset).map_err(|_| EarthVtError::Overflow(section))?;
    let end = usize::try_from(end).map_err(|_| EarthVtError::Overflow(section))?;
    Ok(start..end)
}

fn ranges_overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct TestLayer {
        id: u16,
        channel: TextureChannel,
        format: PixelFormat,
        width: u32,
        height: u32,
        flags: u32,
    }

    #[derive(Clone, Copy)]
    struct TestEntry {
        key: TileKey,
        format: PixelFormat,
    }

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn align_16(value: usize) -> usize {
        (value + 15) & !15
    }

    fn make_container(test_layers: &[TestLayer]) -> Vec<u8> {
        let mut entries = Vec::new();
        let mut layer_counts = Vec::new();
        for test_layer in test_layers {
            let mip_count = full_mip_count(test_layer.width, test_layer.height).unwrap();
            let first = u32::try_from(entries.len()).unwrap();
            for mip in 0..mip_count {
                let width = mip_dimension(test_layer.width, mip);
                let height = mip_dimension(test_layer.height, mip);
                let tiles_x = ceil_div(width, TILE_SIZE as u32);
                let tiles_y = ceil_div(height, TILE_SIZE as u32);
                for y in 0..tiles_y {
                    for x in 0..tiles_x {
                        entries.push(TestEntry {
                            key: TileKey::new(test_layer.id, mip, x, y),
                            format: test_layer.format,
                        });
                    }
                }
            }
            layer_counts.push((
                first,
                u32::try_from(entries.len()).unwrap() - first,
                mip_count,
            ));
        }

        let layer_offset = HEADER_BYTES;
        let index_offset = layer_offset + test_layers.len() * LAYER_ENTRY_BYTES;
        let payload_offset = align_16(index_offset + entries.len() * INDEX_ENTRY_BYTES);
        let payload_bytes = entries
            .iter()
            .map(|entry| usize::try_from(entry.format.encoded_tile_bytes()).unwrap())
            .sum::<usize>();
        let file_bytes = payload_offset + payload_bytes;
        let mut bytes = vec![0_u8; file_bytes];

        bytes[..8].copy_from_slice(&MAGIC);
        put_u16(&mut bytes, 8, VERSION);
        put_u16(&mut bytes, 10, HEADER_BYTES as u16);
        put_u16(&mut bytes, 16, TILE_SIZE);
        put_u16(&mut bytes, 18, GUTTER_SIZE);
        put_u16(&mut bytes, 20, u16::try_from(test_layers.len()).unwrap());
        put_u16(&mut bytes, 22, LAYER_ENTRY_BYTES as u16);
        put_u16(&mut bytes, 24, INDEX_ENTRY_BYTES as u16);
        put_u32(&mut bytes, 28, u32::try_from(entries.len()).unwrap());
        put_u64(&mut bytes, 32, layer_offset as u64);
        put_u64(&mut bytes, 40, index_offset as u64);
        put_u64(&mut bytes, 48, payload_offset as u64);
        put_u64(&mut bytes, 56, file_bytes as u64);

        for (number, test_layer) in test_layers.iter().enumerate() {
            let offset = layer_offset + number * LAYER_ENTRY_BYTES;
            let (first_index, index_count, mip_count) = layer_counts[number];
            put_u16(&mut bytes, offset, test_layer.id);
            put_u16(&mut bytes, offset + 2, test_layer.channel.raw());
            put_u16(&mut bytes, offset + 4, test_layer.format.raw());
            put_u16(&mut bytes, offset + 6, mip_count);
            put_u32(&mut bytes, offset + 8, test_layer.width);
            put_u32(&mut bytes, offset + 12, test_layer.height);
            put_u32(&mut bytes, offset + 16, first_index);
            put_u32(&mut bytes, offset + 20, index_count);
            put_u32(&mut bytes, offset + 24, test_layer.flags);
        }

        let mut payload_cursor = payload_offset;
        for (number, entry) in entries.iter().enumerate() {
            let offset = index_offset + number * INDEX_ENTRY_BYTES;
            let payload_size = usize::try_from(entry.format.encoded_tile_bytes()).unwrap();
            put_u16(&mut bytes, offset, entry.key.layer);
            put_u16(&mut bytes, offset + 2, entry.key.mip);
            put_u32(&mut bytes, offset + 4, entry.key.x);
            put_u32(&mut bytes, offset + 8, entry.key.y);
            put_u64(&mut bytes, offset + 16, payload_cursor as u64);
            put_u64(&mut bytes, offset + 24, payload_size as u64);
            put_u32(&mut bytes, offset + 32, number as u32 + 1);
            bytes[payload_cursor] = (number as u8).wrapping_add(1);
            payload_cursor += payload_size;
        }
        bytes
    }

    fn single_layer() -> TestLayer {
        TestLayer {
            id: 7,
            channel: TextureChannel::CloudDensity,
            format: PixelFormat::Bc4,
            width: 512,
            height: 256,
            flags: LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
        }
    }

    #[test]
    fn fixed_tile_storage_matches_gpu_block_layouts() {
        assert_eq!(PADDED_TILE_SIZE, 264);
        assert_eq!(PixelFormat::Bc7.encoded_tile_bytes(), 69_696);
        assert_eq!(PixelFormat::Bc5.encoded_tile_bytes(), 69_696);
        assert_eq!(PixelFormat::Bc4.encoded_tile_bytes(), 34_848);
        assert_eq!(PixelFormat::R16.encoded_tile_bytes(), 139_392);
        assert_eq!(PixelFormat::Bc1.encoded_tile_bytes(), 34_848);
    }

    #[test]
    fn parses_complete_multi_channel_pyramid_without_copying_payloads() {
        let bytes = make_container(&[
            TestLayer {
                id: 2,
                channel: TextureChannel::DayColor,
                format: PixelFormat::Bc7,
                width: 256,
                height: 128,
                flags: LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
            },
            single_layer(),
        ]);
        let container = EarthVt::parse(&bytes).unwrap();

        assert_eq!(container.header().tile_size, TILE_SIZE);
        assert_eq!(container.layers().len(), 2);
        assert_eq!(
            container.layer(2).unwrap().channel,
            TextureChannel::DayColor
        );
        assert_eq!(
            container.entries().count(),
            container.header().index_count as usize
        );
        assert_eq!(
            container.payload_region().as_ptr(),
            bytes[container.header().payload_offset as usize..].as_ptr()
        );

        let tile = container.tile(TileKey::new(2, 0, 0, 0)).unwrap();
        assert_eq!(
            tile.payload.len(),
            PixelFormat::Bc7.encoded_tile_bytes() as usize
        );
        assert_eq!(tile.payload[0], 1);
    }

    #[test]
    fn parent_fallback_uses_a_resident_coarser_mip() {
        let bytes = make_container(&[single_layer()]);
        let container = EarthVt::parse(&bytes).unwrap();
        let requested = TileKey::new(7, 0, 1, 0);
        let parent = container.parent_key(requested).unwrap();
        assert_eq!(parent, TileKey::new(7, 1, 0, 0));

        let now = Instant::now();
        let mut residency = ResidencyTracker::new(1_000_000);
        let parent_tile = container.tile(parent).unwrap();
        residency
            .mark_resident(parent, parent_tile.entry.payload_bytes, now, false)
            .unwrap();

        let resolved = container.resolve_resident(requested, &residency).unwrap();
        assert_eq!(resolved.requested, requested);
        assert_eq!(resolved.resolved, parent);
        assert_eq!(resolved.levels_up, 1);
    }

    #[test]
    fn rejects_out_of_bounds_and_overflowing_offsets() {
        let mut bytes = make_container(&[single_layer()]);
        let index_offset = read_u64(&bytes, 40) as usize;
        let file_bytes = bytes.len() as u64;
        put_u64(&mut bytes, index_offset + 16, file_bytes);
        assert!(matches!(
            EarthVt::parse(&bytes),
            Err(EarthVtError::OutOfBounds {
                section: "tile payload",
                ..
            })
        ));

        let mut bytes = make_container(&[single_layer()]);
        put_u64(&mut bytes, 32, u64::MAX - 1);
        assert!(matches!(
            EarthVt::parse(&bytes),
            Err(EarthVtError::Overflow(_))
        ));
    }

    #[test]
    fn rejects_noncanonical_payload_size_and_incomplete_mips() {
        let mut bytes = make_container(&[single_layer()]);
        let index_offset = read_u64(&bytes, 40) as usize;
        put_u64(&mut bytes, index_offset + 24, 4);
        assert!(matches!(
            EarthVt::parse(&bytes),
            Err(EarthVtError::InvalidIndex {
                reason: "payload byte length does not match the fixed tile format",
                ..
            })
        ));

        let mut bytes = make_container(&[single_layer()]);
        let layer_offset = read_u64(&bytes, 32) as usize;
        put_u16(&mut bytes, layer_offset + 6, 1);
        assert!(matches!(
            EarthVt::parse(&bytes),
            Err(EarthVtError::InvalidLayer {
                reason: "mip count is not a complete pyramid",
                ..
            })
        ));
    }

    #[test]
    fn rejects_overlapping_metadata_and_duplicate_tile_keys() {
        let mut bytes = make_container(&[single_layer()]);
        let layer_offset = read_u64(&bytes, 32);
        put_u64(&mut bytes, 40, layer_offset);
        assert!(matches!(
            EarthVt::parse(&bytes),
            Err(EarthVtError::InvalidHeader(
                "layer and index tables overlap"
            ))
        ));

        let mut bytes = make_container(&[single_layer()]);
        let index_offset = read_u64(&bytes, 40) as usize;
        put_u32(&mut bytes, index_offset + INDEX_ENTRY_BYTES + 4, 0);
        assert!(matches!(
            EarthVt::parse(&bytes),
            Err(EarthVtError::InvalidIndex {
                reason: "entries are not in canonical order",
                ..
            })
        ));
    }

    #[test]
    fn request_queue_upgrades_prefetch_and_residency_obeys_hysteresis() {
        let key_a = TileKey::new(1, 0, 0, 0);
        let key_b = TileKey::new(1, 0, 1, 0);
        let mut queue = TileRequestQueue::default();
        queue.enqueue(TileRequest::prefetch(key_a, 20, 10));
        queue.enqueue(TileRequest::visible(key_a, 5, 11));
        queue.enqueue(TileRequest::visible(key_b, 30, 11));
        let requests = queue.drain_prioritized();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].key, key_b);
        assert_eq!(requests[1].key, key_a);
        assert_eq!(requests[1].kind, RequestKind::Visible);
        assert_eq!(requests[1].priority, 20);

        let now = Instant::now();
        let old = now.checked_sub(Duration::from_secs(3)).unwrap();
        let mut residency = ResidencyTracker::new(100);
        residency.mark_resident(key_a, 40, old, false).unwrap();
        residency.mark_resident(key_b, 40, now, false).unwrap();
        assert_eq!(
            residency.plan_admission(40, now),
            AdmissionPlan::Evict {
                keys: vec![key_a],
                reclaimed_bytes: 40,
            }
        );
    }
}
