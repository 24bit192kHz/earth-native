fn main() {
    // intel_tex_2 bundles C++ objects (ASTC path) that need the C++ runtime.
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
