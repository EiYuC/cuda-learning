{
  description = "CUDA learning environment for GTX 1650 Ti";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }: {
    devShells.x86_64-linux.default =
      let
        pkgs = import nixpkgs {
          system = "x86_64-linux";
          config.allowUnfree = true;
        };
        cuda = pkgs.cudaPackages_12;
      in
      (pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
        packages = with pkgs; [ cmake ninja gdb ] ++ (with cuda; [
          cuda_nvcc cuda_gdb cuda_sanitizer_api
        ]);
        buildInputs = with cuda; [
          cuda_cudart cccl libcublas libcufft libcurand libcusolver libcusparse
        ];
        CUDA_PATH = "${cuda.cuda_nvcc}";
        CUDACXX = "${cuda.cuda_nvcc}/bin/nvcc";
        CUDAHOSTCXX = "${cuda.backendStdenv.cc}/bin/g++";
        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        '';
      };
  };
}
