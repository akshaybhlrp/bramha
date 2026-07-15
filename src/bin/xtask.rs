use std::env;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    let command = &args[1];
    match command.as_str() {
        "test" => {
            println!("⚙️ Running Bramha Neural Engine standardized Cargo test pipelines...");
            let status = Command::new("cargo")
                .arg("test")
                .status()
                .expect("Failed to execute cargo test command");
            if !status.success() {
                std::process::exit(1);
            }
        }
        "bench" => {
            println!("📊 Running Bramha Neural Engine Criterion benchmark checks...");
            let status = Command::new("cargo")
                .arg("bench")
                .status()
                .expect("Failed to execute cargo bench command");
            if !status.success() {
                std::process::exit(1);
            }
        }
        "build" => {
            println!("⚙️ Standardizing workspace binary production build release compilation...");
            let status = Command::new("cargo")
                .arg("build")
                .arg("--release")
                .status()
                .expect("Failed to execute cargo build command");
            if !status.success() {
                std::process::exit(1);
            }
        }
        _ => {
            println!("⚠️ Unknown xtask command: {}", command);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("Bramha Neural Engine Developer Workflow Automation Tool");
    println!("Usage: cargo xtask <command>");
    println!("\nAvailable Commands:");
    println!("  test    - Execute standardized process unit test suites");
    println!("  bench   - Run complete performance benchmark arrays");
    println!("  build   - Compile fully optimized production target bundles");
}
