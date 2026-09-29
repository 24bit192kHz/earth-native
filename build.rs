use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[allow(dead_code)]
#[path = "src/atmosphere.rs"]
mod atmosphere;

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/atmosphere.rs");
    fs::write(out_dir.join("atmosphere-column.bgra"), atmosphere::bake_column_lut())
        .expect("could not bake atmosphere lookup");
    let glslc = env::var_os("EARTH_NATIVE_GLSLC").unwrap_or_else(|| "glslc".into());

    // The explicit list below is authoritative: these seven shaders are the
    // complete set compiled to SPIR-V. Watching the directory re-runs this
    // script when a new source appears, and the scan after the loop warns if
    // it is not added to the list, so an 8th shader can never be silently
    // missed by the incremental cache.
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    let shaders = [
        ("earth.vert", "earth.vert.spv"),
        ("stars.frag", "stars.frag.spv"),
        ("stars_textured.frag", "stars_textured.frag.spv"),
        ("earth.frag", "earth.frag.spv"),
        ("earth_textured.frag", "earth_textured.frag.spv"),
        ("stars_points.vert", "stars_points.vert.spv"),
        ("stars_points.frag", "stars_points.frag.spv"),
        ("post.frag", "post.frag.spv"),
        ("bloom.frag", "bloom.frag.spv"),
    ];
    // Fold the compiler identity into the cache key so a glslc upgrade
    // rebuilds even when every source is unchanged.
    let glslc_version = Command::new(&glslc)
        .arg("--version")
        .output()
        .map(|output| output.stdout)
        .unwrap_or_default();
    let compiler_hash = fnv1a64(&glslc_version);
    for (source, output) in shaders {
        let source_path = shader_dir.join(source);
        let output_path = out_dir.join(output);
        println!("cargo:rerun-if-changed={}", source_path.display());
        println!("cargo:rerun-if-env-changed=EARTH_NATIVE_GLSLC");

        // Incremental cache: invoke glslc only when the output is missing or
        // small, older than the source, or the (source, compiler) hash differs.
        let source_bytes = fs::read(&source_path)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", source_path.display()));
        let cache_key = format!("{:016x}:{:016x}", fnv1a64(&source_bytes), compiler_hash);
        let hash_path = output_path.with_extension("spv.hash");
        let hash_matches = fs::read_to_string(&hash_path)
            .map(|cached| cached.trim() == cache_key)
            .unwrap_or(false);
        let output_len_ok = fs::metadata(&output_path)
            .map(|metadata| metadata.len() >= 20)
            .unwrap_or(false);
        let output_newer = match (fs::metadata(&output_path), fs::metadata(&source_path)) {
            (Ok(output_meta), Ok(source_meta)) => {
                match (output_meta.modified(), source_meta.modified()) {
                    (Ok(output_time), Ok(source_time)) => output_time >= source_time,
                    _ => false,
                }
            }
            _ => false,
        };
        if output_len_ok && output_newer && hash_matches {
            continue;
        }

        let status = Command::new(&glslc)
            .arg("--target-env=vulkan1.3")
            .arg("-O")
            .arg(&source_path)
            .arg("-o")
            .arg(&output_path)
            .status()
            .unwrap_or_else(|error| {
                panic!(
                    "could not execute {} while precompiling {}: {error}",
                    PathBuf::from(&glslc).display(),
                    source_path.display()
                )
            });

        assert!(
            status.success(),
            "glslc failed while compiling {}",
            source_path.display()
        );
        let metadata = fs::metadata(&output_path).expect("glslc did not create a SPIR-V module");
        assert!(
            metadata.len() >= 20,
            "generated SPIR-V module is unexpectedly small"
        );
        fs::write(&hash_path, &cache_key).expect("could not write shader cache hash");
    }

    if let Ok(entries) = fs::read_dir(&shader_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_shader = path.extension().is_some_and(|extension| {
                matches!(
                    extension.to_str(),
                    Some("vert" | "frag" | "comp" | "geom" | "tesc" | "tese" | "glsl")
                )
            });
            let listed = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| shaders.iter().any(|(source, _)| *source == name));
            if is_shader && !listed {
                println!(
                    "cargo:warning=shader {} is not in the build.rs shader list and will not be compiled",
                    path.display()
                );
            }
        }
    }

    compile_sgp4(&manifest_dir);
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn compile_sgp4(manifest_dir: &Path) {
    let wrapper_dir = manifest_dir.join("sgp4");
    println!("cargo:rerun-if-env-changed=EARTH_NATIVE_LIBSGP4_DIR");
    let vendored_dir = env::var_os("EARTH_NATIVE_LIBSGP4_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("third_party/libsgp4"));
    if !vendored_dir.join("CoordGeodetic.h").is_file() {
        panic!(
            "vendored libsgp4 is missing at {}; set EARTH_NATIVE_LIBSGP4_DIR to its directory",
            vendored_dir.display()
        );
    }
    let sources = [
        "CoordGeodetic.cpp",
        "CoordTopocentric.cpp",
        "DateTime.cpp",
        "DecayedException.cpp",
        "Eci.cpp",
        "Globals.cpp",
        "Observer.cpp",
        "OrbitalElements.cpp",
        "SGP4.cpp",
        "SatelliteException.cpp",
        "SolarPosition.cpp",
        "TimeSpan.cpp",
        "Tle.cpp",
        "TleException.cpp",
        "Util.cpp",
        "Vector.cpp",
    ];
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&wrapper_dir)
        .include(&vendored_dir)
        .file(wrapper_dir.join("earth_sgp4.cpp"));
    println!(
        "cargo:rerun-if-changed={}",
        wrapper_dir.join("earth_sgp4.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        wrapper_dir.join("earth_sgp4.h").display()
    );
    for source in sources {
        let path = vendored_dir.join(source);
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(path);
    }
    build.compile("earth_sgp4");
}
