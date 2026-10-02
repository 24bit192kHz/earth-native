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

The Moon's face, checked the same way (2026-09-29 21:56:32 UTC): sub-Earth
point (libration) 2.414°W, 6.573°S against Horizons' 2.416°W, 6.575°S;
position angle of its north pole 343.70° (343.70°); phase angle 41.806°
(41.806°); illuminated 87.270 % (87.270 %); distance 58.113 Earth radii
(58.111).

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
  not mistaken for cloud. The mosaic joins five satellites along fixed
  meridians (178.2° W, 106.1° W, 37.7° W, 22.7° E, 93.1° E), and the one
  looking toward the Sun reads clear sea and haze up to 0.4 brighter than
  its neighbour (sunglint, forward scattering): a straight edge down the
  globe with false cloud on one side. The offset is measured across each
  seam (median over 1.6° of latitude, so real cloud cancels) and taken off
  the brighter side, easing out over 20° of longitude. GFS low cloud fills
  in warm low cloud at night, and GFS covers the poles beyond the mosaic.
  The observed cover (B-spline upsampled, so the 10 km grid leaves no
  straight edges) is read as a cloud
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
  greyer. The shell is at 5.5 km with Sun-traced shadows. Two things a
  flat shell lacks are restored without extra texture fetches: cloud tops
  are shaded as a height field (~1.5 km from edge to core, its slope from
  screen-space derivatives, with wrapped lighting since light diffuses
  inside a cloud), so they show lit and shaded sides under a low Sun; and
  clouds have sides, so a field of fraction f hides
  1 − (1 − f)^(1 + 0.25 tan θ) of the ground at view zenith angle θ and
  broken cloud closes up toward the horizon. The fractal's fine octave is
  read two mips down (rounded ~0.5 km elements instead of specks). Without
  the feed, the NASA composite is shown as it is.
- **Night:** VIIRS Black Marble 2016 lights at 500 m (stored 32768×16384),
  coloured from intensity (sodium to white), city glow under low cloud,
  moonlight from Allen's lunar phase law and distance, and O(¹S) airglow
  integrated as limb columns at 95 km.
- **Aurora:** a volumetric march from 90 to 320 km (~25 km steps, 10-40).
  NOAA OVATION probabilities place the oval and set its activity. As in
  DMSP/VIIRS and ISS imagery, a patchy diffuse glow fills the equatorward
  half, and discrete arcs sit on the poleward flank at fixed probability
  contours (4.5-60 %), so quiet ovals show one or two and storms up to six.
  Arcs are 14 km wide sheets with folds of up to ~100 km and field-aligned
  rays kept to footprints of 80-250 km; a display gain of 150 brings a
  quiet oval to the brightness of orbital time-lapses.
  Brightness follows the local activity: diffuse ~1 kR, arcs from ~20 kR
  (IBC II) to ~130 kR (IBC III-IV) in storms, which seen along the limb
  outshine moonlit cloud, as in storm-time ISS footage. Each
  arc folds, breaks into segments and fades on its own, so they are not
  closed rings. Green 557.7 nm (peak ~110 km), red 630 nm (~240 km) and N₂⁺
  violet profiles; field-aligned rays make curtains at the limb. Emission
  fades out through nautical twilight at the emission point.
- **Lightning:** flashes cluster where NOAA GFS reports convective energy,
  precipitation and cloud water; each takes the shape of the cloud it
  lights, brightest through the thick cores.

## Moon and planets

`earth.frag` intersects the true ellipsoid (polar/equatorial ratio per body)
and applies disk-resolved photometry: Lunar-Lambert with McEwen's phase
function for the airless Moon and Mercury, Minnaert limb darkening for Mars
(k = 0.75), Venus (0.80), Uranus/Neptune (0.85) and Jupiter/Saturn (0.90).
Saturn's rings are a plane with NASA/JPL radial boundaries, modelled optical
depths, gaps, and the planet's oblate shadow with a finite-Sun penumbra.
The rings cast their shadow on Saturn's globe (the slant optical depth of
the ring plane toward the Sun).

Every body goes through the same camera as the Earth: radiance in units of
the body's own sunlight (a white Lambertian surface under its zenith Sun is
1), the same metered auto exposure, the same Sun (a limb-darkened disc of
its true apparent size, 68.6' from Mercury to 1.07' from Neptune, with the
lens's glow), the same stars, bloom, film curve, grain and dither. The
photographic grade (contrast, saturation, colour rendering) is left off:
the planet maps are processed photographs already. Uranus and Neptune are
almost featureless, and their maps' faint gradients are below what BC1
blocks can carry: Uranus is read four mips down and Neptune two.

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
compressive response) on a display scale that does not follow the camera
exposure, so the stars show beside the daylit Earth and the Sun, as in a
composited space film; the PSF widens for bright stars, and colour follows
B−V through blackbody chromaticities; saturation acts on luminance so hues
survive.

NASA SVS Deep Star Maps 2020 (16K HDR, BC1) supplies only the unresolved
background: a 4-tap minimum filter removes its point stars, and it is shown
colourless (the eye sees the Milky Way with rods; the map's photographic
H-alpha red is not visible), as a night series (EV 17) records it at every
camera exposure. It is rotated by sidereal time plus precession
in right ascension, within ~0.1° of the catalogue stars.

## Camera

The scene renders to an HDR (RGBA16F) target, then passes through a model of
a full-frame camera behind an ISS window (`shaders/post.frag`):

- **Point of view:** `camera live` rides the ISS (SGP4 position and velocity,
  level local-horizontal attitude), 78° horizontal lens, pitched so the
  horizon sits ~20 % from the top, 20 fps while riding (the globe animates
  lightning and aurora at 8 fps and otherwise redraws only when it moves).
- **Auto exposure:** metered on a 64 px mip: the 85th-percentile luminance
  of the lit part of the body (anything under 1/500 of its brightest
  percent is left out, so a crescent is exposed for the crescent; the
  Earth's night side, which has its own exposure, is read at that
  exposure and kept as a subject, so a thin twilight arc over it blooms
  at the night exposure instead of holding the camera down), or a highlight rule when bright sunlit sky covers more than 3 %
  of the frame (the sunrise band seen from the ISS, held ~1.5 stops over
  the key so it keeps its colours). Daylight stays near a "sunny 16"
  exposure: a low Sun or open ocean is lifted by 0.6 of its deficit, at
  most 0.9 stops, as the footage's auto exposure does; at night the camera opens up for moonlight, city lights
  and aurora and holds 1.6 stops under the meter for a night look. The Sun
  in frame does not cap the exposure, and a frame without the Earth keeps
  the exposure it had. Light
  adaptation is fast (0.12 s, at most 1.5 stops over on the first frames),
  dark adaptation slow (0.9 s), so panning from the night sky back to the
  daylit Earth does not flash white. A cut to another view is shown at the
  daylight exposure until the new view has been metered (two frames), then
  snaps: a night exposure carried into a daylit view was a white flash.
- **The night side's own exposure:** city lights, moonlit cloud and ground
  (tinted slightly blue, the Purkinje shift), lightning, airglow and aurora
  are drawn as a night series (EV 16) records them whatever the camera
  exposure, eased in through twilight in stops (none with the Sun up, all
  of it 14° down). They stay visible beside the daylit globe and under a
  sunrise; at a night exposure the factor is one and the scene is physical.
- **Local adaptation:** above 0.5 the luminance is compressed toward 1.2
  with the hue kept (after metering), so a twilight arc 2^7 over a night
  exposure keeps its orange, white and blue layers and city lights stay
  golden.
- **The Sun in the sky:** like the Moon, a saturated disc with a soft glow
  of fixed display strength (12 % of its daylight-exposure glare), no
  diffraction rays or ghosts.
- **The Moon in the sky:** Lunar-Lambert photometry (bright to the limb at
  full Moon), the SVS map (stretched to a mean albedo of ~0.5) scaled to the
  real normal albedo of 0.12, and local adaptation: a single night exposure
  would clip the Moon ~13 stops over into a flat white disc, so the disc is
  compressed as the eye or an HDR merge sees it, with the maria readable, and
  its glare is scaled to match.
- **Optics:** bloom from the mip pyramid (`shaders/bloom.frag`: levels 1-2
  at half resolution, 3 and up at one eighth), each level softly compressed above
  16× white (a lit limb 2^10 over a night exposure stays a sharp line with a
  small glow instead of a white fog), an analytic veiling-glare point spread
  for the Sun and the Moon (with a lens-acceptance cut-off off-frame), and
  60 %-corrected vignetting.
- **Colour rendering:** a sensor's green channel reaches into the blue and
  the raw conversion only partly undoes it, so Rayleigh blue photographs
  azure and open water teal: green takes 20 % of blue and blue 10 % of green
  (rows sum to one, greys stay grey). Against the reference video the open
  ocean's blue/green went from 2.3 to 1.7 (video 1.4-2.0) and red/green
  from 0.51 to 0.40 (0.26-0.36); land moves by less than 0.05.
- **Development:** log-space contrast around mid-grey, a ×1.3 saturation
  (the "vivid" picture style of processed Earth-observation frames) that
  fades to none between 1 and 6× white, so clipped highlights run to white
  as a sensor's do, a filmic curve with a long shoulder, gain-dependent sensor grain and dithered 8-bit
  output.

Presents are motion-gated, so an idle desktop re-renders only when the scene
moves by at least half a pixel, or while the virtual texture is streaming.

## Remaining approximations

Spherical Earth surface (WGS84 equatorial radius), UTC ≈ UT1 (< 0.9 s),
three colour bands (exact only for thin paths), aerosol sampled once per ray
(the Sun's path uses the tables' climatology), a 2004 surface composite
(seasonal but not today's snow or vegetation), clouds as one 5.5 km shell,
empirical aurora structure, cloud relief from derivatives (no cast
self-shadows), and a two-parameter camera colour rendering.
