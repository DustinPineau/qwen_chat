{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python313                # Python 3.13 interpreter
    python313Packages.pip    # pip package manager
    pyright                  # Python LSP server
    stdenv.cc.cc.lib         # C++ standard library (libstdc++), required by compiled pip packages
    zlib                     # compression library (libz), required by compiled pip packages
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
  '';
}
