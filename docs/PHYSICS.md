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

- **Atmosphere:** single scattering on a 100 km shell with a precomputed
  molecular column table (512×1, exact integration of an 8.5 km scale height,
  < 0.05 % error at grazing angles). Rayleigh optical depths follow Bodhaine
  et al. (1999) at 680/550/440 nm (0.041/0.097/0.243). Sunlight reaching the
  scattering air is attenuated along its own path with the single-scattering
  average of sun- and view-path extinction, which gives the orange twilight
  band at the terminator while the full-disk rim stays blue.
- **Aerosols:** optical depth 0.05 (clean marine), Henyey–Greenstein g = 0.72
  + 15 % isotropic, single-scattering albedo 0.95; they also dim the surface.
- **Solar eclipses:** the exact area overlap of the solar and lunar discs as
  seen from every surface/cloud point dims direct light and skylight.
- **Surface:** NASA Blue Marble NG albedo (saturation × 0.72, calibrated
  against DSCOVR/EPIC true colour), relief normals from GEBCO, Beer–Lambert
  reddening of sunlight, Cox–Munk sunglint (7 m/s wind slope variance).
- **Clouds:** NASA Blue Marble cloud composite on a 5.5 km shell (global mean
  cloud-top height). Opacity saturates with brightness (thick decks opaque,
  thin cirrus translucent), with sun-traced cloud shadows.
- **Night:** VIIRS Black Marble lights coloured from intensity (sodium to
  white), moonlight from Allen's lunar phase law and distance with a
  night-adaptation gain (like a dark-adapted eye or an ISS night photo),
  airglow at 95 km integrated over the pixel footprint.
- **Aurora:** a 12-step volumetric march from 90 to 320 km. NOAA OVATION
  probabilities place the oval; arcs follow its contours; green 557.7 nm
  (peak ~110 km), red 630 nm (~240 km) and N₂⁺ violet profiles; rays and
  folds animate on seconds-to-minutes scales.
- **Lightning:** flashes cluster where NOAA GFS reports convective energy,
  precipitation and cloud water.

Calibration against a real DSCOVR/EPIC frame (2026-09-25 16:34:46 UTC, same
sub-observer point, area-weighted disk statistics):

| Statistic | EPIC | earth-native |
| --- | --- | --- |
| Clear-ocean sRGB | 39, 53, 70 | 47, 60, 78 |
| White-cloud area (L > 170) | 7.0 % | 6.6 % |
| Disk mean colour | 96, 101, 107 | within ~3 % |

## Moon and planets

`earth.frag` intersects the true ellipsoid (polar/equatorial ratio per body)
and applies disk-resolved photometry: Lunar-Lambert with McEwen's phase
function for the airless Moon and Mercury, Minnaert limb darkening for Mars
(k = 0.75), Venus (0.80), Uranus/Neptune (0.85) and Jupiter/Saturn (0.90).
Saturn's rings are a plane with NASA/JPL radial boundaries, modelled optical
depths, gaps, and the planet's oblate shadow with a finite-Sun penumbra.
Exposure adapts to each body (as a camera would).

## Stars

NASA SVS Deep Star Maps 2020 (Hipparcos-2, Tycho-2, Gaia DR2), 16K HDR,
tone-mapped so the brightest 0.002 % of texels reach white, BC1 with a full
mip chain, sampled in the J2000 frame rotated by the current sidereal time.

## Presentation

A filmic curve (small toe, exponential shoulder above 0.72) maps linear
radiance to the SDR swapchain. The Earth pass is scissored to the atmosphere
(or aurora) shell; presents are motion-gated so an idle desktop re-renders
only when the scene moves by at least half a pixel.

## Remaining approximations

Spherical Earth surface (WGS84 equatorial radius), UTC ≈ UT1 (< 0.9 s),
single scattering, static historical NASA cloud map, empirical aurora
structure, planet texture longitude origins as published by the source.
