fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "hyperscale")]
    {
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .compile_protos(&["proto/bramha.proto"], &["proto/"])?;
    }
    Ok(())
}
