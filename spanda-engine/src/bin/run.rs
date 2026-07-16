use spanda_engine::{EngineConfig, InferenceSession, GenerationParams};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("Usage: spanda-run <model.spanda> \"<prompt>\"");
        return Ok(());
    }

    let model_path = &args[1];
    let prompt = &args[2];

    println!("🚀 Starting SPANDA Inference Session...");
    println!("  Model: {}", model_path);
    println!("  Prompt: \"{}\"", prompt);

    let config = EngineConfig {
        model_path: model_path.clone(),
        max_vram_budget_mb: 2048,
        enable_l3_offload: true,
        enable_prefetch: true,
    };

    let mut session = InferenceSession::new(config)?;

    // Mock tokenization of prompt (just mapping characters to token IDs)
    let prompt_tokens: Vec<u32> = prompt.chars().map(|c| c as u32).collect();

    let params = GenerationParams {
        temperature: 0.7,
        top_p: 0.9,
        max_new_tokens: 32,
    };

    println!("\nGenerating response...");
    let mut completion_tokens = Vec::new();
    for token_res in session.generate(&prompt_tokens, &params) {
        match token_res {
            Ok(token) => {
                completion_tokens.push(token);
                // Print character representation of u32 (back to char)
                let c = std::char::from_u32(token).unwrap_or(' ');
                print!("{}", c);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Err(e) => {
                println!("Error generating token: {}", e);
                break;
            }
        }
    }
    println!("\n\n✅ Done! Generated {} tokens.", completion_tokens.len());

    Ok(())
}
