use tokenizers::Tokenizer;
fn main() {
    let t = Tokenizer::from_file("/home/akshay-bhalerao/tensor_data/qwen2.5-0.5b/tokenizer.json")
        .unwrap();
    let enc = t.encode("", false).unwrap();
    println!("Empty string: {:?}", enc.get_ids());
    let enc = t.encode("<|im_start|>system", false).unwrap();
    println!("<|im_start|>system: {:?}", enc.get_ids());
}
