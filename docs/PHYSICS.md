# Physics and verification

earth-native renders analytic spheres/ellipsoids with one fullscreen triangle
per pass (stars, then the body), so the cost is per pixel, not per vertex.
Everything below runs in the fragment shaders `shaders/earth_textured.frag`
(Earth), `shaders/earth.frag` (Moon and planets) and
`shaders/stars_textured.frag` (sky).

## Sky geometry

- **Ephemerides:** Astronomy Engine 2.1.19 (offline VSOP87/ELP) gives the
  Sun, Moon and planet positions, apparent sizes, lunar phase, Greenwich
  apparent sidereal time and the IAU rotation axes of every body.
- **Units:** WGS84 equatorial radius 6378.137 km, AU 149 597 870.7 km (IAU
  2012), solar radius 695 700 km, lunar radius 1737.4 km, NASA/JPL planetary
  radii and polar flattening.
- **Planet frames:** each body uses its IAU pole and prime meridian, so its
  subsolar point moves at its own solar-day rate (checked in tests: Jupiter
  ~36°/h, Uranus ~21°/h, Neptune ~22°/h, Venus and the Moon < 1°/h).
- **ISS:** the Earth camera can follow the ISS, propagated with SGP4 from
  CelesTrak elements.

Verified against JPL Horizons (2026-09-27 17:58:10 UTC):

| Quantity | earth-native | JPL Horizons |
| --- | --- | --- |
| Subsolar point | 91.82°W, 1.85°S | 91.82°W, 1.85°S (RA − GAST) |
| Sun distance | 149 933 000 km | 149 932 500 km |
| Sublunar point | 99.00°E, 10.64°N | 99.00°E, 10.64°N (geocentric) |
| Moon distance | 375 675 km | 375 683 km |
| Moon illuminated | 98.549 % | 98.550 % |

A pixel test at 2024-04-08 18:18 UTC puts the rendered equatorial terminator
within one 0.94° sample of the Horizons prediction, and the total-eclipse
lunar shadow falls over Mexico and Texas as in the DSCOVR/EPIC image of that day.

## Earth

- **Atmosphere:** physically based multiple scattering. `src/sky.rs` bakes
  the Bruneton & Neyret (2008) transmittance table (256×64), Hillaire's
  (2020) multiple-scattering table (32²) and a ground/cloud-top sky
  irradiance table (64×16) at startup; the shader marches every view ray
  through the 100 km shell, lighting each sample by the Sun and (at night)
  the Moon. Air: Rayleigh scattering (Bodhaine et al. 1999 dispersion,
  13.6e-3 km⁻¹ at 550 nm, 8 km scale height). Ozone: Chappuis absorption
  (Serdyuchenko/Gorshelev 2013 cross sections at 223 K) in a 25 km layer of
  ~300 DU. Aerosol: Cornette–Shanks g = 0.68, single-scattering albedo 0.94,
  1.8 km scale height. Units: a white Lambertian surface under a zenith Sun at
  1 AU is 1.
- **Colour bands, not wavelengths:** every coefficient is its spectrum
  averaged over one sRGB colour-matching function (CIE 1931 via the
  Wyman–Sloan–Shirley fit and the sRGB matrix) under a 5778 K Sun, which is
  exact for thin paths. With single wavelengths (650/550/450 nm) ozone
  absorbed green more than red, although the red band spans the 603 nm
  Chappuis peak, and long grazing paths turned magenta. With bands, a ray
  grazing 20 km up transmits blue > green > red (the blue ozone band above the
  orange sunset layer; Hulburt 1953), and the far haze turns from lavender to
  blue. Over the open ocean in the reference ISS video (linear red/green of
  the haze 0.53), the render went from 0.70 to 0.46.
- **Aerosol, live:** the NOAA GEFS-Aerosols analysis (GOCART, 0.25°, every
  6 h) gives the 550 nm optical depth and the 440–645 nm Ångström exponent
  (dust ≈ 0.2, smoke and pollution ≈ 1.5). Each ray takes the load where it
  meets the haze layer (its ground point, or the tangent point at the limb).
  Sunlight at the ground is dimmed and reddened by the local excess over the
  tables' climatology (τ = 0.18, α = 0.5), and what that excess scatters goes
  into the diffuse sky. Without the feed, the climatology is used.
- **Solar eclipses:** the exact area overlap of the solar and lunar discs as
  seen from every surface/cloud point dims direct light and skylight.
- **Surface:** NASA Blue Marble Next Generation at 500 m, the current
  month's composite (snow and vegetation follow the season; the nearest
  baked month stands in), streamed as a 65536×32768 BC7 virtual texture
  (256 px pages, GPU feedback). Relief is shaded from GEBCO 2026 15″ normals
  (BC5, 32768×16384) and exaggerated with distance (2.5 × 1.3^mip), as the
  eye reads it from orbit. The open ocean uses Case-1 water-leaving
  reflectance plus Cox–Munk sunglint and Fresnel sky reflection. Blue Marble
  has no sea ice, so the daily EUMETSAT OSI SAF concentration (10 km, both
  poles) lays pack ice over open water (albedo 0.70/0.75/0.80, no glint).
- **Clouds, live:** hourly NOAA GMGSI geostationary mosaics (visible by day,
  10.7 µm infrared by night, ~10 km). Each is compared with a decaying
  clear-sky composite of the same place, so deserts, snow and sea ice are
  not mistaken for cloud. GFS low cloud fills in warm low cloud at night, and
  GFS covers the poles beyond the mosaic. The observed cover (B-spline
  upsampled, so the 10 km grid leaves no straight edges) is read as a cloud
  fraction. Where inside it the cloud sits comes from a fractal projected
  triplanar from the sphere (isotropic, no lat/lon shear) mixed with the 1 km
  texture of the NASA cloud composite: that field is cut at its
  (1 − cover) quantile, measured over the textures, so the cloudy area keeps
  the observed fraction. Edges are sharp at ISS scale, one pixel's footprint
  wide at most (antialiased), and how far the field rises above the cut
  stands in for optical depth: thin fringes are grey (albedo 0.5), cores
  white (0.92). Once a pixel spans more than ~1.5-8 km (the globe view, the
  horizon), the mips have averaged the field toward its mean and the cut
  would switch at 50 % cover, so the observed cover is shown as the
  fraction it is instead, textured by the low-passed field, thin cover
  greyer. The shell is at 5.5 km with Sun-traced shadows. Without the
  feed, the NASA composite is shown as it is.
- **Night:** VIIRS Black Marble 2016 lights at 500 m (stored 32768×16384),
  coloured from intensity (sodium to white), city glow under low cloud,
  moonlight from Allen's lunar phase law and distance, and O(¹S) airglow
  integrated as limb columns at 95 km.
- **Aurora:** a volumetric march from 90 to 320 km (~25 km steps, 10-40).
  NOAA OVATION probabilities place the oval and set its activity. As in
  DMSP/VIIRS and ISS imagery, a patchy diffuse glow fills the equatorward
  half, and discrete arcs sit on the poleward flank at fixed probability
  contours (4.5-60 %), so quiet ovals show one or two and storms up to six.
  Brightness follows the local activity: diffuse ~1 kR, arcs from ~20 kR
  (IBC II) to ~130 kR (IBC III-IV) in storms, which seen along the limb
  outshine moonlit cloud, as in storm-time ISS footage. Each
  arc folds, breaks into segments and fades on its own, so they are not
  closed rings. Green 557.7 nm (peak ~110 km), red 630 nm (~240 km) and N₂⁺
  violet profiles; field-aligned rays make curtains at the limb. Emission
  fades out through nautical twilight at the emission point.
- **Lightning:** flashes cluster where NOAA GFS reports convective energy,
  precipitation and cloud water.

## Moon and planets

`earth.frag` intersects the true ellipsoid (polar/equatorial ratio per body)
and applies disk-resolved photometry: Lunar-Lambert with McEwen's phase
function for the airless Moon and Mercury, Minnaert limb darkening for Mars
(k = 0.75), Venus (0.80), Uranus/Neptune (0.85) and Jupiter/Saturn (0.90).
Saturn's rings are a plane with NASA/JPL radial boundaries, modelled optical
depths, gaps, and the planet's oblate shadow with a finite-Sun penumbra.
Exposure adapts to each body (as a camera would).

## Stars

Stars are geometry, not an image: every Hipparcos star with V ≤ 8.0
(41 394 stars, `assets/stars/`) is its own quad in a Vulkan draw
(`shaders/stars_points.vert/.frag`), shaded as a Gaussian point-spread
function. Each frame the vertex shader places every star for the current
UTC: Hipparcos proper motion from epoch J1991.25, then one J2000-to-world
matrix from Astronomy Engine (precession, nutation, apparent sidereal time,
the same frame as the Sun and Moon), then annual aberration from Earth's
barycentric velocity. Checked against the 2026-08-23 Sun–Regulus
conjunction: Regulus renders 0.55° from the Sun's centre, the true value.
Brightness steps 1.8× per magnitude (the eye's and a photograph's
compressive response), the PSF widens for bright stars, and colour follows
B−V through blackbody chromaticities; saturation acts on luminance so hues
survive.

NASA SVS Deep Star Maps 2020 (16K HDR, BC1) supplies only the unresolved
background: a 4-tap minimum filter removes its point stars, and it is shown
colourless (the eye sees the Milky Way with rods; the map's photographic
H-alpha red is not visible). It is rotated by sidereal time plus precession
in right ascension, within ~0.1° of the catalogue stars.

## Camera

The scene renders to an HDR (RGBA16F) target, then passes through a model of
a full-frame camera behind an ISS window (`shaders/post.frag`):

- **Point of view:** `camera live` rides the ISS (SGP4 position and velocity,
  level local-horizontal attitude), 78° horizontal lens, pitched so the
  horizon sits ~20 % from the top, 30 fps while riding.
- **Two looks** (`earth-native look realistic|cinematic`, remembered across
  restarts, or `EARTH_NATIVE_LOOK`). *Realistic* is one physical camera: a
  single exposure for the whole frame, so a Sun in frame (or the daylit
  Earth) hides the stars, and the Sun carries its full lens glare.
  *Cinematic* shows what the adapted eye would like on a desktop: the Milky
  Way and the catalogue stars keep a fixed night-series brightness (EV 17)
  whatever the exposure, so they show next to the daylit Earth and the Sun,
  and the Sun is a soft glow with faint rays (its glare capped at 3 % of the
  daylight glare, disc compressed like the Moon's).
- **Milky Way (cinematic):** baked at startup (`src/milky_way.rs`, ~0.7 s)
  from the star panorama: a ~0.09° mip is decoded, its point stars removed by
  a 3 × 3 morphological opening and the result blurred; its colour is not
  used (BC1's 565 endpoints turn dark regions green and magenta), the glow
  is tinted from bluish in the faint arms to warm in the bright band. The
  sky's median glow is its black point (only the band shows, empty sky
  stays black) and the band's 99.5th percentile sits at 0.05 linear through
  a contrast curve (power 1.6): a subtle glow behind the Earth, not a fog.
- **Auto exposure:** metered on a 64 px mip: the 85th-percentile Earth
  luminance, or a highlight rule when bright sunlit sky covers more than 3 %
  of the frame (the sunrise band seen from the ISS). Daylight keeps a "sunny
  16" exposure; at night the camera opens up for moonlight, city lights,
  aurora and stars and holds 1.6 stops under the meter for a night look. A
  thin lit limb around a night globe saturates like city lights. Realistic:
  the Sun's glare is metered over the whole frame (mean glare under 8 %,
  92 % of the frame under half white) and an open Sun in frame caps the
  exposure at daylight; a frame without the Earth is exposed as a starfield.
  Cinematic: a frame without the Earth keeps the current exposure. Light
  adaptation is fast (0.12 s, at most 1.5 stops over on the first frames),
  dark adaptation slow (0.9 s), so panning from the night sky back to the
  daylit Earth does not flash white. Cuts snap the exposure.
- **The Moon in the sky:** Lunar-Lambert photometry (bright to the limb at
  full Moon), the SVS map (stretched to a mean albedo of ~0.5) scaled to the
  real normal albedo of 0.12, and local adaptation: a single night exposure
  would clip the Moon ~13 stops over into a flat white disc, so the disc is
  compressed as the eye or an HDR merge sees it, with the maria readable, and
  its glare is scaled to match.
- **Optics:** bloom from the mip pyramid, each level softly compressed above
  16× white (a lit limb 2^10 over a night exposure stays a sharp line with a
  small glow instead of a white fog), an analytic veiling-glare point spread
  and an 18-ray diffraction starburst for the Sun and the Moon (with a
  lens-acceptance cut-off off-frame), ghosts, and 60 %-corrected vignetting.
- **Development:** log-space contrast around mid-grey, a ×1.3 saturation
  (the "vivid" picture style of processed Earth-observation frames), a filmic
  curve with a long shoulder, gain-dependent sensor grain and dithered 8-bit
  output.

Presents are motion-gated, so an idle desktop re-renders only when the scene
moves by at least half a pixel, or while the virtual texture is streaming.

## Remaining approximations

Spherical Earth surface (WGS84 equatorial radius), UTC ≈ UT1 (< 0.9 s),
three colour bands (exact only for thin paths), aerosol sampled once per ray
(the Sun's path uses the tables' climatology), a 2004 surface composite
(seasonal but not today's snow or vegetation), clouds as one 5.5 km shell,
empirical aurora structure, and camera colour rendering (the video's haze is
still more cyan than ours: blue/green 1.59 against 2.2).
