use rand::rngs::StdRng;

pub trait Model: Sized {
    type ModelError;

    fn build_n_gram<P>(&mut self, pre_processor_chain: P, corpus: Vec<&str>) -> Result<(), String>
    where
        P: Fn(String) -> String;

    fn generate(&self, max_tokens: u32, seed: u64, top_k: f64, temperature: f64) -> String;

    fn predict_next_token(
        &self, context: &str, rng: &mut StdRng, top_k: f64, temperature: f64,
    ) -> &str;

    fn save(&self) -> Result<String, Self::ModelError>;

    fn load(path: &str) -> Result<Self, String>;
}
