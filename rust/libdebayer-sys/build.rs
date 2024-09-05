use anyhow::Result;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=src/lib.rs");

    let libdebayer_lib = pkg_config::probe_library("libdebayer")?;

    let include_args: Vec<String> = libdebayer_lib.include_paths
        .iter()
        .map(|path| format!("-I{}", path.to_string_lossy()))
        .collect();

    let link_args: Vec<String> = libdebayer_lib.link_paths
        .iter()
        .map(|path| format!("-L{}", path.clone().into_os_string().into_string().unwrap()))
        .collect();

    let libs_args: Vec<String> = libdebayer_lib.libs
        .iter()
        .map(|path| format!("-l{}", path.to_string()))
        .collect();

    let mut clang_args = Vec::new();
    clang_args.extend(include_args);
    clang_args.extend(link_args);
    clang_args.extend(libs_args);

    let bindings = bindgen::Builder::default()
        .clang_args(clang_args)
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .default_enum_style(bindgen::EnumVariation::ModuleConsts)
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    Ok(())
}
