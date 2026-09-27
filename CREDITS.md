# Credits and data licenses

The code is MIT licensed (see `LICENSE`). The texture pack shipped with
releases is built by `pipeline/data_pipeline.py` from the sources below; the
exact URLs and SHA-256 of every input are recorded in its `provenance.json`.
NASA does not endorse this project.

| Data | Source | License / terms |
| --- | --- | --- |
| Earth surface | NASA Earth Observatory, Blue Marble Next Generation (Sept. 2004, topography + bathymetry), Reto Stöckli | NASA imagery, public domain; credit NASA Earth Observatory |
| Night lights | NASA Earth Observatory / Suomi NPP VIIRS, Black Marble 2016 (grayscale) | Public domain; credit NASA Earth Observatory |
| Clouds | NASA Blue Marble cloud composite (Visible Earth) | Public domain; credit NASA |
| Relief | NASA Visible Earth, GEBCO 2008 elevation | GEBCO Compilation Group; free use with attribution |
| Moon | NASA Goddard SVS, CGI Moon Kit (LRO WAC/LOLA) | Public domain; credit NASA's Scientific Visualization Studio |
| Stars | NASA Goddard SVS, Deep Star Maps 2020 (Hipparcos-2, Tycho-2, Gaia DR2) | Public domain; credit NASA/Goddard SVS; Gaia data ESA/Gaia/DPAC (CC BY-SA 3.0 IGO) |
| Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune | Solar System Scope textures (based on NASA mission imagery) | CC BY 4.0, https://www.solarsystemscope.com/textures/ |
| Weather | NOAA NCEP GFS 0.25° analysis (AWS Open Data) | Public domain (NOAA) |
| Aurora | NOAA SWPC OVATION Prime forecast | Public domain (NOAA) |
| ISS orbit | CelesTrak general perturbations elements | Free for non-commercial use |

Software included or linked:

- `third_party/libsgp4` — SGP4 propagator by Daniel Warner, Apache-2.0 (license in that directory).
- Astronomy Engine by Don Cross (via `astronomy-engine-bindings`), MIT.
- Rust crates listed in `Cargo.lock`, under their respective licenses.
