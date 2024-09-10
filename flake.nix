{
  description = "libdebayer";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "aarch64-linux" "x86_64-linux" ] (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        libdebayer = pkgs.stdenv.mkDerivation {
          pname = "libdebayer";
          version = "0.1.0";
          src = ./c;
          nativeBuildInputs = [ pkgs.cmake ];
          buildInputs = [ pkgs.cudatoolkit ];
          preConfigure = ''
             export CUDA_PATH=${pkgs.cudatoolkit}
          '';
        };

        libdebayer_cpp = pkgs.stdenv.mkDerivation {
          pname = "libdebayer_cpp";
          version = "0.1.0";
          src = ./cpp;
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [ pkgs.cudatoolkit libdebayer ];
          preConfigure = ''
             export CUDA_PATH=${pkgs.cudatoolkit}
             export libdebayer_DIR=${libdebayer}/lib/cmake
          '';
        };

        fetch_kodak = pkgs.writeShellApplication {
          name = "fetch_kodak";
          runtimeInputs = [ pkgs.wget ];

          text = builtins.readFile ./benchmark/kodak/download_kodak.sh;
        };

        kodak_benchmark_cpp_unwrapped = pkgs.stdenv.mkDerivation {
          pname = "kodak_benchmark_cpp_unwrapped";
          version = "0.1.0";
          src = ./benchmark/cpp;
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [ pkgs.cudatoolkit pkgs.opencv libdebayer libdebayer_cpp ];
          preConfigure = ''
             export CUDA_PATH=${pkgs.cudatoolkit}
             export libdebayer_DIR=${libdebayer}/lib/cmake
             export libdebayercpp_DIR=${libdebayer_cpp}/lib/cmake
          '';
        };

        kodak_benchmark_cpp = pkgs.writeShellApplication {
          name = "kodak_benchmark_cpp";
          runtimeInputs = [ fetch_kodak kodak_benchmark_cpp_unwrapped ];
          text = ''
             tmpdir=$(mktemp -d)
             echo "using $tmpdir"
             cd "$tmpdir"
             fetch_kodak
             KODAK_FOLDER_PATH="$tmpdir" test_debayer
             rm -rf "$tmpdir"
          '';
        };
      in
      with pkgs;
      {
        packages.libdebayer = libdebayer;
        packages.libdebayer_cpp = libdebayer_cpp;
        packages.fetch_kodak = fetch_kodak;
        packages.kodak_benchmark_cpp_unwrapped = kodak_benchmark_cpp_unwrapped;
        packages.kodak_benchmark_cpp = kodak_benchmark_cpp;
        
        devShells.default = mkShell {
          nativeBuildInputs = with pkgs; [ pkg-config cmake clang ];
          buildInputs = with pkgs; [
            gitFull gitRepo gnupg autoconf curl
            procps gnumake util-linux m4 gperf unzip
            cudatoolkit linuxPackages.nvidia_x11
            libGLU libGL
            opencv
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib
            ncurses5 stdenv.cc binutils
            libdebayer
            libdebayer_cpp
            rust-bin.nightly.latest.default
            rust-analyzer
            imagemagick
          ];

          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/";


          shellHook = ''
             export CUDA_PATH=${pkgs.cudatoolkit}
             export libdebayer_DIR=${libdebayer}/lib/cmake
             export libdebayercpp_DIR=${libdebayer_cpp}/lib/cmake
             # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
             export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
             export EXTRA_CCFLAGS="-I/usr/include"
          '';
        };
      }
    ) // {
        overlays = {
          default = final: _prev: {
            libdebayer = self.packages.${final.system}.libdebayer;
            libdebayer_cpp = self.packages.${final.system}.libdebayer_cpp;
          };
        };
    };
}
